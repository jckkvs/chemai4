"""
backend/chem/smiles_transformer.py

sklearn Pipeline に組み込める SMILES→記述子変換 Transformer。
学習時・推論時を通じて一貫した変換が保証される。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from backend.chem.rdkit_adapter import RDKitAdapter
from backend.chem.psmiles_adapter import PSmilesAdapter
from backend.chem.mix_rules import default_rules_manager

logger = logging.getLogger(__name__)


class SmilesDescriptorTransformer(BaseEstimator, TransformerMixin):
    """
    SMILES列を記述子に変換するsklearn互換Transformer。

    Parameters
    ----------
    smiles_col : str
        SMILESが入力されている列名。
    selected_descriptors : list[str] | None
        使用する記述子名のリスト。Noneの場合は全計算結果を使用。
    count_normalization : str
        数え上げ系記述子(原子数/環数/官能基数等)の正規化方式。
        "raw" = そのまま個数(デフォルト値)
        "density" = モル体積(cm3/mol)で割った密度(デフォルト)
        引用: van Krevelen 2009, 密度記述子は分子サイズの影響を除外
    """

    def __init__(
        self,
        smiles_col: str | list[dict] | None = None,
        selected_descriptors: list[str] | None = None,
        active_engines: list[str] | None = None,
        count_normalization: str = "density",
        fraction_type: str = "wt",
        # Morgan フィンガープリント オプション
        morgan_count: bool = False,
        morgan_radius: int = 2,
        morgan_bits: int = 2048,
        morgan_order_by_appearance: bool = False,
    ) -> None:
        self.smiles_col = smiles_col
        self.selected_descriptors = selected_descriptors
        self.active_engines = active_engines
        self.count_normalization = count_normalization  # "raw" or "density"
        self.fraction_type = fraction_type
        # Morgan フィンガープリント オプション
        self.morgan_count = morgan_count
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.morgan_order_by_appearance = morgan_order_by_appearance
        
        self.components = []
        if isinstance(smiles_col, str):
            self.components = [{"smiles_col": smiles_col, "fraction_col": None}]
        elif isinstance(smiles_col, list):
            self.components = smiles_col
        elif smiles_col is None:
            self.components = [{"smiles_col": "smiles", "fraction_col": None}]

        self._descriptor_cols: list[str] = []
        self._non_smiles_cols: list[str] = []

    def _compute_descriptors(self, smiles_list: list[str]) -> pd.DataFrame:
        """SMILESリストから記述子DataFrameを計算する。"""
        from backend.chem import RDKitAdapter, MordredAdapter
        from backend.chem.psmiles_adapter import PSmilesAdapter
        
        # ポリマーSMILES (PSMILES) の検出
        # リストの先頭50件を見て、1件でも '*' または '[*]' があればポリマーとみなす
        has_psmiles = any(PSmilesAdapter.is_psmiles(smi) for smi in smiles_list[:50] if isinstance(smi, str))
        
        # Streamlit セッションからの事前計算結果の再利用（存在する場合）
        try:
            import streamlit as st
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                precalc_df = st.session_state.get("precalc_smiles_df")
                orig_df = st.session_state.get("df")
                orig_smiles_col = st.session_state.get("smiles_col")
                
                if precalc_df is not None and not precalc_df.empty and orig_df is not None and orig_smiles_col:
                    if not hasattr(st.session_state, "_smiles_precalc_dict"):
                        mapping = {}
                        for idx, smi in zip(orig_df.index, orig_df[orig_smiles_col]):
                            if pd.notna(smi) and idx in precalc_df.index:
                                mapping[str(smi)] = precalc_df.loc[idx]
                        st.session_state["_smiles_precalc_dict"] = mapping
                    
                    smi_dict = st.session_state["_smiles_precalc_dict"]
                    
                    hit_rows = []
                    all_hit = True
                    for smi in smiles_list:
                        smi_str = str(smi)
                        if smi_str in smi_dict:
                            hit_rows.append(smi_dict[smi_str])
                        else:
                            all_hit = False
                            break
                            
                    if all_hit and hit_rows:
                        logger.info("事前計算された記述子キャッシュを再利用します。")
                        cached_df = pd.DataFrame(hit_rows)
                        cached_df.index = range(len(cached_df))
                        
                        if self.selected_descriptors and not has_psmiles:
                            valid_cols = [c for c in self.selected_descriptors if c in cached_df.columns]
                            if valid_cols:
                                return cached_df[valid_cols]
                        
                        return cached_df
        except Exception as e:
            logger.debug(f"キャッシュ再利用スキップ: {e}")

        adapters = []
        if has_psmiles:
            logger.info("PSMILESを検出しました。ポリマー用記述子抽出モード(PSmilesAdapter)に切り替えます。")
            adapters.append(PSmilesAdapter())
            # PSMILES は従来アダプタを使用
            desc_dfs: list[pd.DataFrame] = []
            for adapter in adapters:
                if adapter.is_available():
                    try:
                        res = adapter.compute(smiles_list)
                        desc_dfs.append(res.descriptors)
                    except Exception as e:
                        logger.warning(f"{adapter.name}: 計算エラー - {e}")
            if not desc_dfs:
                return pd.DataFrame()
            X_chem = pd.concat(desc_dfs, axis=1)
            X_chem = X_chem.loc[:, ~X_chem.columns.duplicated()]
            return X_chem
        else:
            # ── プラグインレジストリ経由で全記述子を計算 ──
            try:
                from backend.chem.descriptors import compute_all_descriptors, get_plugins_by_engine
                
                # エンジン名から必要なプラグイン名を特定
                plugin_names = None
                if self.active_engines is not None:
                    plugin_names = []
                    for e in self.active_engines:
                        # RDKitAdapter -> RDKit 等の変換
                        eng_map = {"RDKitAdapter": "RDKit", "XTBAdapter": "XTB", "MordredAdapter": "Mordred", 
                                "SkfpAdapter": "scikit-FP", "Mol2VecAdapter": "Mol2Vec", "GroupContribAdapter": "GroupContrib",
                                "MolAIAdapter": "MolAI", "UMAAdapter": "UMA", "PaDELAdapter": "PaDEL", 
                                "DescriptaStorusAdapter": "DescriptaStorus", "MolfeatAdapter": "Molfeat", 
                                "ChempropAdapter": "Chemprop", "CosmoAdapter": "COSMO", "UniPkaAdapter": "UniPKa"}
                        eng_str = eng_map.get(e, e)
                        for p in get_plugins_by_engine(eng_str):
                            plugin_names.append(p.name)
                            
                X_chem = compute_all_descriptors(smiles_list, plugin_names=plugin_names)
                if not X_chem.empty:
                    # 選択されている場合はフィルタリング
                    if self.selected_descriptors:
                        valid = [c for c in self.selected_descriptors if c in X_chem.columns]
                        if valid:
                            X_chem = X_chem[valid]
                        else:
                            logger.warning(
                                f"selected_descriptorsの記述子がいずれも計算結果に存在しません。"
                                f"フォールバックとして計算結果({X_chem.shape[1]}列)をそのまま使用します。"
                            )
                    return X_chem
                else:
                    logger.warning("プラグインレジストリから記述子が0件。従来アダプタにフォールバック。")
            except Exception as e:
                logger.warning(f"プラグインレジストリ経由の計算に失敗: {e}。従来アダプタにフォールバック。")

            # フォールバック: 従来のRDKit+Mordredアダプタ
            from backend.chem import RDKitAdapter, MordredAdapter
            adapters = [
                RDKitAdapter(
                    compute_fp=True,
                    morgan_count=self.morgan_count,
                    morgan_radius=self.morgan_radius,
                    morgan_bits=self.morgan_bits,
                    morgan_order_by_appearance=self.morgan_order_by_appearance,
                ),
                MordredAdapter(selected_only=True),
            ]
            desc_dfs = []
            for adapter in adapters:
                if adapter.is_available():
                    try:
                        res = adapter.compute(smiles_list)
                        desc_dfs.append(res.descriptors)
                    except Exception as e:
                        logger.warning(f"{adapter.name}: 計算エラー - {e}")
            if not desc_dfs:
                return pd.DataFrame()
            X_chem = pd.concat(desc_dfs, axis=1)
            X_chem = X_chem.loc[:, ~X_chem.columns.duplicated()]

        # 選択されている場合はフィルタリング
        if self.selected_descriptors:
            valid = [c for c in self.selected_descriptors if c in X_chem.columns]
            if valid:
                X_chem = X_chem[valid]
            else:
                logger.warning(
                    "selected_descriptorsの記述子がいずれも最終計算結果に存在しません。"
                    "全記述子を使用します (フォールバック)。"
                )
                
        return X_chem

    def _apply_count_normalization(
        self, X_chem: pd.DataFrame, smiles_list: list[str]
    ) -> pd.DataFrame:
        """
        数え上げ系記述子のモル体積密度正規化。

        count_normalization == "density" の場合、
        is_count=True の全列をモル体積(cm3/mol)で割る。

        Implements: van Krevelen 2009 - 密度正規化記述子
        引用: 分子サイズの影響を除外するため、カウント系記述子を
              モル体積で正規化。これにより異なるサイズの分子間で
              官能基密度を公平に比較可能。
        """
        if self.count_normalization != "density":
            return X_chem

        # 数え上げ系列名を特定
        count_cols = self._identify_count_columns(X_chem.columns.tolist())
        if not count_cols:
            return X_chem

        # モル体積を計算（RDKit MolWt / 推定密度, or AllChem）
        mol_volumes = self._compute_molar_volumes(smiles_list)
        if mol_volumes is None:
            return X_chem

        # 密度変換: count / V
        X_out = X_chem.copy()
        for col in count_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col] / mol_volumes
                # 列名にサフィックス追加（密度であることを明示）
                X_out.rename(columns={col: f"{col}_density"}, inplace=True)

        logger.info(
            f"数え上げ記述子を密度変換: {len(count_cols)}列 / "
            f"モル体積範囲: {mol_volumes.min():.1f}-{mol_volumes.max():.1f} cm3/mol"
        )
        return X_out

    @staticmethod
    def _identify_count_columns(columns: list[str]) -> list[str]:
        """数え上げ系の列名を特定する。"""
        count_cols = []
        for col in columns:
            # fr_系 (官能基フラグメントカウント)
            if col.startswith("fr_"):
                count_cols.append(col)
            # Num系 (原子数/結合数/環数)
            elif col.startswith("Num") or "Count" in col:
                count_cols.append(col)
            # NHOHCount, NOCount
            elif col in ("NHOHCount", "NOCount"):
                count_cols.append(col)
        return count_cols

    @staticmethod
    def _compute_molecular_weight(smiles_list: list[str]) -> "np.ndarray":
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            volumes = []
            for smi in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(str(smi))
                    if mol is not None:
                        mw = Descriptors.MolWt(mol)
                        volumes.append(mw)
                    else:
                        volumes.append(100.0)  # フォールバック
                except Exception:
                    volumes.append(100.0)
            return np.array(volumes)
        except ImportError:
            return np.full(len(smiles_list), 100.0)

    @staticmethod
    def _compute_molar_volumes(smiles_list: list[str]) -> "np.ndarray | None":
        """SMILESリストからモル体積を計算する。"""
        mws = SmilesDescriptorTransformer._compute_molecular_weight(smiles_list)
        return np.maximum(mws / 1.0, 10.0)

    def _transform_mixture(self, X: pd.DataFrame) -> pd.DataFrame:
        """複数SMILESコンポーネントからの加重平均記述子の計算"""
        n_samples = len(X)
        descs_list = []
        mw_list = []
        W_given_list = []
        
        for comp in self.components:
            s_col = comp.get("smiles_col")
            f_col = comp.get("fraction_col")
            
            smi_list = X[s_col].tolist() if s_col and s_col in X.columns else [""] * n_samples
            
            # デスクリプタ計算
            desc = self._compute_descriptors(smi_list)
            desc = self._apply_count_normalization(desc, smi_list)
            descs_list.append(desc)
            
            # 分子量計算
            mw = self._compute_molecular_weight(smi_list)
            mw_list.append(mw)
            
            # 割合取得
            if f_col and f_col in X.columns:
                w = pd.to_numeric(X[f_col], errors="coerce").fillna(0.0).values
            else:
                w = np.ones(n_samples)
            W_given_list.append(w)
            
        # wt/mol 分率の計算
        W_given_mat = np.column_stack(W_given_list)
        MW_mat = np.column_stack(mw_list)
        
        W_sum = W_given_mat.sum(axis=1, keepdims=True)
        W_sum[W_sum == 0] = 1.0
        W_given_mat = W_given_mat / W_sum # normalize
        
        if self.fraction_type == "wt":
            wt_frac = W_given_mat
            moles = W_given_mat / MW_mat
            mol_sum = moles.sum(axis=1, keepdims=True)
            mol_sum[mol_sum == 0] = 1.0
            mol_frac = moles / mol_sum
        else: # mol
            mol_frac = W_given_mat
            weights = W_given_mat * MW_mat
            wt_sum = weights.sum(axis=1, keepdims=True)
            wt_sum[wt_sum == 0] = 1.0
            wt_frac = weights / wt_sum
            
        # ベースとするカラムセットは最初の有効な成分のものとする
        ref_cols = descs_list[0].columns if descs_list else pd.Index([])
        
        res = pd.DataFrame(index=X.index, columns=ref_cols)
        
        for col in ref_cols:
            rule = default_rules_manager.get_rule(str(col))
            fracs = wt_frac if rule == "wt" else mol_frac
            
            val = np.zeros(n_samples)
            for i in range(len(self.components)):
                if col in descs_list[i].columns:
                    val += fracs[:, i] * descs_list[i][col].values
            res[col] = val
            
        return res

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SmilesDescriptorTransformer":
        for comp in self.components:
            if comp["smiles_col"] not in X.columns:
                raise ValueError(f"SMILES列 '{comp['smiles_col']}' がDataFrameに存在しません。")
                
        X_chem = self._transform_mixture(X)
        
        # 全NaN列の除去
        all_nan_cols = [c for c in X_chem.columns if X_chem[c].isna().all()]
        if all_nan_cols:
            X_chem = X_chem.drop(columns=all_nan_cols)
            
        self._descriptor_cols = X_chem.columns.tolist()
        
        # 削除すべきSMILES列を特定
        drop_cols = [c["smiles_col"] for c in self.components]
        self._non_smiles_cols = [c for c in X.columns if c not in drop_cols]
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        for comp in self.components:
            if comp["smiles_col"] not in X.columns:
                raise ValueError(f"SMILES列 '{comp['smiles_col']}' がDataFrameに存在しません。")
                
        X_chem = self._transform_mixture(X)

        for col in self._descriptor_cols:
            if col not in X_chem.columns:
                X_chem[col] = 0.0
        X_chem = X_chem[self._descriptor_cols].reset_index(drop=True)

        X_rest = X.reset_index(drop=True)
        drop_cols = []
        for c in self.components:
            if c.get("smiles_col"): drop_cols.append(c.get("smiles_col"))
            if c.get("fraction_col") and c.get("fraction_col") != "（なし）": drop_cols.append(c.get("fraction_col"))
        drop_cols.extend([c for c in self._descriptor_cols if c in X_rest.columns])
        X_rest = X_rest.drop(columns=[c for c in drop_cols if c in X_rest.columns], errors="ignore")

        return pd.concat([X_rest, X_chem], axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self._non_smiles_cols + self._descriptor_cols)

def progressive_precalculate(smiles_list: list[str], target_col_name: str = ""):
    """
    ユーザーの要求に応じ、優先順位をつけて事前計算を行い、進捗を yield するジェネレータ。
    Yields:
        (progress_float, status_message, current_df)
    """
    if not smiles_list:
        yield 1.0, "SMILES列が空です", pd.DataFrame()
        return

    # PSMILES check
    has_psmiles = any(PSmilesAdapter.is_psmiles(smi) for smi in smiles_list[:50] if isinstance(smi, str))
    
    if has_psmiles:
        yield 0.1, "PSMILES形式を検出しました。Polymer用モデルをロード中...", pd.DataFrame()
        adapter = PSmilesAdapter()
        try:
            res = adapter.compute(smiles_list)
            df_res = res.descriptors
            yield 1.0, "計算完了（PSMILES）", df_res
        except Exception as e:
            logger.error(f"PSMILES事前計算エラー: {e}")
            yield 1.0, f"エラー: {e}", pd.DataFrame()
        return

    # 通常のSMILES：3ステップで計算を完了させる
    from backend.chem.recommender import get_target_recommendation_by_name

    # --- ステップ1: 目的変数に対する推奨記述子 ---
    yield 0.3, f"目的変数「{target_col_name or '不明'}」に関連する推奨記述子を計算中...", pd.DataFrame()
    rec = get_target_recommendation_by_name(target_col_name)
    rec_names = [d.name for d in rec.descriptors] if rec else []

    rdkit_adapter = RDKitAdapter(
        compute_fp=False,
        morgan_count=self.morgan_count,
        morgan_radius=self.morgan_radius,
        morgan_bits=self.morgan_bits,
        morgan_order_by_appearance=self.morgan_order_by_appearance,
    )
    df_result = pd.DataFrame(index=range(len(smiles_list)))

    if rec_names and rdkit_adapter.is_available():
        try:
            df_rd_rec = rdkit_adapter.compute(smiles_list, selected_descriptors=rec_names).descriptors
            df_result = pd.concat([df_result, df_rd_rec], axis=1)
        except Exception:
            pass
    df_result = df_result.loc[:, ~df_result.columns.duplicated()]

    # --- ステップ2: 数え上げ系記述子 (is_count=True) ---
    yield 0.6, "数え上げ系記述子（原子数、環数等）を計算中...", df_result
    if rdkit_adapter.is_available():
        try:
            mdata = rdkit_adapter.get_descriptors_metadata()
            count_names = [m.name for m in mdata if m.is_count and m.name not in df_result.columns]
            if count_names:
                df_counts = rdkit_adapter.compute(smiles_list, selected_descriptors=count_names).descriptors
                df_result = pd.concat([df_result, df_counts], axis=1)
        except Exception:
            pass
    df_result = df_result.loc[:, ~df_result.columns.duplicated()]

    # --- ステップ3: 意味のある主要記述子 (厳選12個) ---
    CURATED_DESCRIPTORS = [
        "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
        "NumRotatableBonds", "RingCount", "NumAromaticRings",
        "FractionCSP3", "HeavyAtomCount", "MolMR", "HallKierAlpha",
    ]
    yield 0.9, "主要な物理化学記述子（分子量・LogP・TPSA等）を計算中...", df_result

    curated = [c for c in CURATED_DESCRIPTORS if c not in df_result.columns]
    if curated and rdkit_adapter.is_available():
        try:
            df_curated = rdkit_adapter.compute(smiles_list, selected_descriptors=curated).descriptors
            df_result = pd.concat([df_result, df_curated], axis=1)
        except Exception:
            pass

    df_result = df_result.loc[:, ~df_result.columns.duplicated()]
    yield 1.0, f"完了 — {len(df_result.columns)}個の主要記述子を抽出しました", df_result


def precalculate_all_descriptors(
    smiles_list: list[str],
    target_col_name: str = "",
    engine_flags: dict[str, bool] | None = None,
    molai_n_components: int = 32,
    progress_callback: Any = None,
) -> tuple[pd.DataFrame, dict | None]:
    """
    全記述子を一括計算する関数。app.py のインラインコードを統一。

    Args:
        smiles_list: 有効なSMILESのリスト
        target_col_name: 目的変数名（推奨記述子の選定に使用）
        engine_flags: {"use_mordred": True, ...} 各エンジンのON/OFF
        molai_n_components: MolAI PCA次元数
        progress_callback: (step, total, message) を呼ぶコールバック（省略可）

    Returns:
        (df_result, molai_variance_info)
        - df_result: 記述子DataFrame
        - molai_variance_info: MolAI PCA寄与率情報 or None
    """
    if engine_flags is None:
        engine_flags = {}

    def _progress(step: int, total: int, msg: str) -> None:
        if progress_callback:
            progress_callback(step, total, msg)

    n = len(smiles_list)
    if n == 0:
        return pd.DataFrame(), None

    # --- ステップ1: RDKit基本記述子（推奨+数え上げ+主要物理化学） ---
    _progress(1, 5, "推奨記述子を計算中...")
    df_result = pd.DataFrame(index=range(n))

    from backend.chem.recommender import get_target_recommendation_by_name
    rdkit_adapter = RDKitAdapter(
        compute_fp=False,
        morgan_count=self.morgan_count,
        morgan_radius=self.morgan_radius,
        morgan_bits=self.morgan_bits,
        morgan_order_by_appearance=self.morgan_order_by_appearance,
    )

    rec = get_target_recommendation_by_name(target_col_name)
    rec_names = [d.name for d in rec.descriptors] if rec else []
    if rec_names and rdkit_adapter.is_available():
        try:
            df_tmp = rdkit_adapter.compute(smiles_list, selected_descriptors=rec_names).descriptors
            df_result = pd.concat([df_result, df_tmp], axis=1)
        except Exception:
            pass

    _progress(2, 5, "数え上げ系記述子を計算中...")
    if rdkit_adapter.is_available():
        try:
            mdata = rdkit_adapter.get_descriptors_metadata()
            count_names = [m.name for m in mdata if m.is_count and m.name not in df_result.columns]
            if count_names:
                df_tmp = rdkit_adapter.compute(smiles_list, selected_descriptors=count_names).descriptors
                df_result = pd.concat([df_result, df_tmp], axis=1)
        except Exception:
            pass

    CURATED = [
        "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
        "RingCount", "NumAromaticRings", "FractionCSP3",
        "HeavyAtomCount", "MolMR", "HallKierAlpha",
    ]
    _progress(3, 5, "主要物理化学記述子を計算中...")
    curated = [c for c in CURATED if c not in df_result.columns]
    if curated and rdkit_adapter.is_available():
        try:
            df_tmp = rdkit_adapter.compute(smiles_list, selected_descriptors=curated).descriptors
            df_result = pd.concat([df_result, df_tmp], axis=1)
        except Exception:
            pass

    df_result = df_result.loc[:, ~df_result.columns.duplicated()]
    df_result = df_result.apply(pd.to_numeric, errors="coerce").convert_dtypes()

    # --- ステップ2: 追加エンジン ---
    _progress(4, 5, "追加エンジンの記述子を計算中...")
    _engine_adapters = {
        "use_mordred":  ("Mordred",      "backend.chem.mordred_adapter",       "MordredAdapter",      {"selected_only": True}),
        "use_xtb":      ("XTB",          "backend.chem.xtb_adapter",           "XTBAdapter",          {}),
        "use_cosmo":    ("COSMO-RS",     "backend.chem.cosmo_adapter",         "CosmoAdapter",        {}),
        "use_unipka":   ("UniPKa",       "backend.chem.unipka_adapter",        "UniPkaAdapter",       {}),
        "use_contrib":  ("GroupContrib", "backend.chem.group_contrib_adapter", "GroupContribAdapter", {}),
        "use_uma":      ("UMA",          "backend.chem.uma_adapter",           "UMAAdapter",          {}),
        "use_skfp":     ("scikit-FP",    "backend.chem.skfp_adapter",          "SkfpAdapter",         {}),
        "use_padel":    ("PaDEL",        "backend.chem.padel_adapter",         "PaDELAdapter",        {}),
        "use_ds":       ("DescriptaStorus","backend.chem.descriptastorus_adapter","DescriptaStorusAdapter",{}),
        "use_mol2vec":  ("Mol2Vec",      "backend.chem.mol2vec_adapter",       "Mol2VecAdapter",      {}),
        "use_molfeat":  ("Molfeat",      "backend.chem.molfeat_adapter",       "MolfeatAdapter",      {}),
        "use_chemprop": ("Chemprop",     "backend.chem.chemprop_adapter",      "ChempropAdapter",     {}),
    }
    extra_results: list[tuple[str, int]] = []  # [(name, n_new_cols), ...]

    for ekey, (ename, module_path, class_name, kwargs) in _engine_adapters.items():
        if engine_flags.get(ekey, False):
            try:
                mod = __import__(module_path, fromlist=[class_name])
                adapter_cls = getattr(mod, class_name)
                adapter = adapter_cls(**kwargs)
                if adapter.is_available():
                    eres = adapter.compute(smiles_list)
                    edf = eres.descriptors
                    new_cols = [c for c in edf.columns if c not in df_result.columns]
                    if new_cols:
                        edf_new = edf[new_cols].copy()
                        edf_new.index = df_result.index[:len(edf_new)]
                        df_result = pd.concat([df_result, edf_new], axis=1)
                        extra_results.append((ename, len(new_cols)))
            except Exception as e:
                logger.warning(f"{ename} スキップ: {e}")

    # --- ステップ3: MolAI ---
    molai_variance = None
    if engine_flags.get("use_molai", False):
        _progress(5, 5, "MolAI CNN Encoder + PCA を計算中...")
        try:
            from backend.chem.molai_adapter import MolAIAdapter
            molai_adp = MolAIAdapter(n_components=molai_n_components)
            if molai_adp.is_available():
                molai_adp.compute(smiles_list)
                if molai_adp._pca is not None:
                    evr = molai_adp._pca.explained_variance_ratio_
                    molai_variance = {
                        "ratio": evr.tolist(),
                        "cumulative": evr.cumsum().tolist(),
                        "n_components": molai_n_components,
                    }
        except Exception as e:
            logger.warning(f"MolAI スキップ: {e}")

    _progress(5, 5, f"完了 — {len(df_result.columns)}個の記述子を抽出")
    return df_result, molai_variance
