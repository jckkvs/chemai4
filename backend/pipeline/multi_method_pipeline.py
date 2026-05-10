"""
backend/pipeline/multi_method_pipeline.py

マルチスケール・マルチメソッド 階層的計算パイプライン。

3層のフィルタリング戦略で計算リソースを効率配分:

Layer 1 (高速スクリーニング): RDKit 2D記述子 + 簡易ML予測
  → 閾値基準で有望分子のみ Layer 2 へ
Layer 2 (中精度): xTB最適化 + 基本特徴量
  → 不確実性評価で信頼度低い分子を識別
Layer 3 (高精度, オプション): 追加計算（将来拡張用）

既存モジュールへの影響: なし（完全新規）
既存アダプターを呼び出すが、変更は一切しない。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LayerResult:
    """各計算層の結果。"""
    layer_name: str
    n_input: int
    n_output: int
    descriptors: pd.DataFrame
    passed_indices: list[int]
    failed_indices: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """マルチメソッドパイプラインの設定。"""
    # Layer 1: RDKit
    enable_layer1_rdkit: bool = True
    layer1_filter: Callable[[pd.DataFrame], list[int]] | None = None
    # 上記がNoneの場合、全分子をLayer 2に通す

    # Layer 2: xTB
    enable_layer2_xtb: bool = True
    xtb_calc_type: str = "opt"
    xtb_kwargs: dict[str, Any] = field(default_factory=dict)

    # Layer 2+: xTB ML派生特徴量
    enable_ml_features: bool = True

    # Layer 2+: 不確実性評価
    enable_uncertainty: bool = True
    min_confidence: float = 0.0  # この信頼度未満の分子を警告

    # 進捗コールバック
    progress_callback: Callable[[str, int, int], None] | None = None


class MultiMethodPipeline:
    """
    マルチスケール階層的計算パイプライン。

    大規模仮想スクリーニングに対応するため、高速な2D記述子で
    事前フィルタリングし、有望分子のみに高コストなxTB計算を適用する。

    使い方::

        pipeline = MultiMethodPipeline()
        result = pipeline.run(
            smiles_list=["CCO", "c1ccccc1", ...],
            config=PipelineConfig(
                enable_layer2_xtb=True,
                xtb_calc_type="sp",
            ),
        )
        print(result.shape)  # (n_molecules, n_features)
    """

    def run(
        self,
        smiles_list: list[str],
        config: PipelineConfig | None = None,
        charge_config_store: Any | None = None,
    ) -> pd.DataFrame:
        """
        パイプラインを実行し、統合特徴量DataFrameを返す。

        Args:
            smiles_list: 入力SMILESリスト。
            config: パイプライン設定。
            charge_config_store: 電荷設定ストア。

        Returns:
            行=分子（入力順）、列=特徴量 の DataFrame。
            計算されなかった分子の行はNaN埋め。
        """
        cfg = config or PipelineConfig()
        n_total = len(smiles_list)
        results: dict[str, LayerResult] = {}

        # 全分子のインデックス管理
        active_indices = list(range(n_total))

        # ── Layer 1: RDKit 2D記述子（高速）──
        if cfg.enable_layer1_rdkit:
            layer1 = self._run_layer1_rdkit(
                smiles_list, active_indices, cfg, charge_config_store,
            )
            results["rdkit_2d"] = layer1

            # フィルタリング
            if cfg.layer1_filter is not None and not layer1.descriptors.empty:
                passed = cfg.layer1_filter(layer1.descriptors)
                active_indices = [
                    layer1.passed_indices[i]
                    for i in passed
                    if i < len(layer1.passed_indices)
                ]
                logger.info(
                    "Layer 1 フィルタ: %d/%d 分子が通過",
                    len(active_indices), layer1.n_input,
                )
            else:
                active_indices = layer1.passed_indices

        # ── Layer 2: xTB計算（中精度）──
        if cfg.enable_layer2_xtb and active_indices:
            layer2_smiles = [smiles_list[i] for i in active_indices]
            layer2 = self._run_layer2_xtb(
                layer2_smiles, active_indices, cfg, charge_config_store,
            )
            results["xtb"] = layer2

            # ML派生特徴量
            if cfg.enable_ml_features and not layer2.descriptors.empty:
                ml_layer = self._run_ml_features(layer2, cfg)
                results["xtb_ml"] = ml_layer

            # 不確実性評価
            if cfg.enable_uncertainty and not layer2.descriptors.empty:
                unc_layer = self._run_uncertainty(layer2, cfg)
                results["uncertainty"] = unc_layer

        # ── 結果の統合 ──
        combined = self._merge_results(results, n_total)

        logger.info(
            "パイプライン完了: %d分子 × %d特徴量",
            combined.shape[0], combined.shape[1],
        )
        return combined

    # ────────────────────────────────────────────────────────
    # Layer 実装
    # ────────────────────────────────────────────────────────

    def _run_layer1_rdkit(
        self,
        smiles_list: list[str],
        indices: list[int],
        cfg: PipelineConfig,
        charge_config_store: Any,
    ) -> LayerResult:
        """Layer 1: RDKit 2D記述子。"""
        if cfg.progress_callback:
            cfg.progress_callback("Layer 1: RDKit 2D記述子", 0, len(indices))

        t0 = time.time()
        try:
            from backend.chem.rdkit_adapter import RDKitAdapter

            adapter = RDKitAdapter(compute_fp=False, compute_gasteiger=True)
            if not adapter.is_available():
                logger.warning("RDKitが利用不可: Layer 1スキップ")
                return LayerResult(
                    "rdkit_2d", len(indices), 0,
                    pd.DataFrame(), indices,
                )

            result = adapter.compute(
                smiles_list,
                charge_config_store=charge_config_store,
            )

            elapsed = time.time() - t0
            if cfg.progress_callback:
                cfg.progress_callback("Layer 1: RDKit 完了", len(indices), len(indices))

            return LayerResult(
                layer_name="rdkit_2d",
                n_input=len(indices),
                n_output=result.descriptors.shape[0],
                descriptors=result.descriptors,
                passed_indices=indices,
                failed_indices=result.failed_indices,
                elapsed_seconds=elapsed,
            )
        except Exception as e:
            logger.error("Layer 1 失敗: %s", e)
            return LayerResult(
                "rdkit_2d", len(indices), 0,
                pd.DataFrame(), indices,
                elapsed_seconds=time.time() - t0,
            )

    def _run_layer2_xtb(
        self,
        smiles_list: list[str],
        original_indices: list[int],
        cfg: PipelineConfig,
        charge_config_store: Any,
    ) -> LayerResult:
        """Layer 2: xTB計算。"""
        if cfg.progress_callback:
            cfg.progress_callback("Layer 2: xTB計算", 0, len(smiles_list))

        t0 = time.time()
        try:
            from backend.chem.xtb_adapter import XTBAdapter

            xtb_kwargs = dict(cfg.xtb_kwargs)
            if "calc_type" not in xtb_kwargs:
                xtb_kwargs["calc_type"] = cfg.xtb_calc_type

            adapter = XTBAdapter(**xtb_kwargs)
            if not adapter.is_available():
                logger.warning("xTBが利用不可: Layer 2スキップ")
                return LayerResult(
                    "xtb", len(smiles_list), 0,
                    pd.DataFrame(), original_indices,
                )

            result = adapter.compute(
                smiles_list,
                charge_config_store=charge_config_store,
            )

            elapsed = time.time() - t0
            if cfg.progress_callback:
                cfg.progress_callback(
                    "Layer 2: xTB完了", len(smiles_list), len(smiles_list),
                )

            return LayerResult(
                layer_name="xtb",
                n_input=len(smiles_list),
                n_output=result.descriptors.shape[0],
                descriptors=result.descriptors,
                passed_indices=original_indices,
                failed_indices=[
                    original_indices[i]
                    for i in result.failed_indices
                    if i < len(original_indices)
                ],
                elapsed_seconds=elapsed,
                metadata=result.metadata,
            )
        except Exception as e:
            logger.error("Layer 2 失敗: %s", e)
            return LayerResult(
                "xtb", len(smiles_list), 0,
                pd.DataFrame(), original_indices,
                elapsed_seconds=time.time() - t0,
            )

    @staticmethod
    def _run_ml_features(
        xtb_layer: LayerResult,
        cfg: PipelineConfig,
    ) -> LayerResult:
        """xTB結果から派生ML特徴量を抽出する。"""
        t0 = time.time()
        try:
            from backend.chem.xtb_ml_features import XTBMLFeatureExtractor

            extractor = XTBMLFeatureExtractor()
            records = xtb_layer.descriptors.to_dict("records")

            # 座標情報の取得
            coords_list = xtb_layer.metadata.get("optimized_coords", [])
            xyz_list = []
            atoms_list = []
            for ci in coords_list:
                if ci and "coords" in ci:
                    xyz_list.append(ci["coords"])
                    atoms_list.append(ci.get("atomic_numbers"))
                else:
                    xyz_list.append(None)
                    atoms_list.append(None)

            while len(xyz_list) < len(records):
                xyz_list.append(None)
                atoms_list.append(None)

            df = extractor.batch_extract(records, xyz_list, atoms_list)

            return LayerResult(
                layer_name="xtb_ml",
                n_input=len(records),
                n_output=df.shape[0],
                descriptors=df,
                passed_indices=xtb_layer.passed_indices,
                elapsed_seconds=time.time() - t0,
            )
        except Exception as e:
            logger.error("ML特徴量抽出失敗: %s", e)
            return LayerResult(
                "xtb_ml", 0, 0,
                pd.DataFrame(), xtb_layer.passed_indices,
                elapsed_seconds=time.time() - t0,
            )

    @staticmethod
    def _run_uncertainty(
        xtb_layer: LayerResult,
        cfg: PipelineConfig,
    ) -> LayerResult:
        """不確実性評価を実行する。"""
        t0 = time.time()
        try:
            from backend.chem.uncertainty_estimator import UncertaintyEstimator

            estimator = UncertaintyEstimator()
            records = xtb_layer.descriptors.to_dict("records")
            reports = estimator.batch_evaluate(records)

            # 信頼度スコアをDataFrame化
            features_list = [r.to_features() for r in reports]
            df = pd.DataFrame(features_list)

            # 低信頼度の警告
            low_conf = [
                i for i, r in enumerate(reports)
                if r.overall_confidence < cfg.min_confidence
            ]
            if low_conf:
                logger.warning(
                    "%d分子の信頼度が閾値 %.2f 未満",
                    len(low_conf), cfg.min_confidence,
                )

            return LayerResult(
                layer_name="uncertainty",
                n_input=len(records),
                n_output=df.shape[0],
                descriptors=df,
                passed_indices=xtb_layer.passed_indices,
                elapsed_seconds=time.time() - t0,
                metadata={"low_confidence_indices": low_conf},
            )
        except Exception as e:
            logger.error("不確実性評価失敗: %s", e)
            return LayerResult(
                "uncertainty", 0, 0,
                pd.DataFrame(), xtb_layer.passed_indices,
                elapsed_seconds=time.time() - t0,
            )

    @staticmethod
    def _merge_results(
        results: dict[str, LayerResult],
        n_total: int,
    ) -> pd.DataFrame:
        """全レイヤーの結果を分子インデックスで結合する。"""
        combined = pd.DataFrame(index=range(n_total))

        for layer_name, layer in results.items():
            if layer.descriptors.empty:
                continue

            # レイヤーの結果をオリジナルインデックスに配置
            df_aligned = pd.DataFrame(
                index=range(n_total),
                columns=layer.descriptors.columns,
                dtype=float,
            )

            for row_idx, orig_idx in enumerate(layer.passed_indices):
                if row_idx < len(layer.descriptors):
                    df_aligned.iloc[orig_idx] = layer.descriptors.iloc[row_idx]

            # カラム名の衝突回避
            existing_cols = set(combined.columns)
            new_cols = {}
            for col in df_aligned.columns:
                if col in existing_cols:
                    new_cols[col] = f"{layer_name}_{col}"
            if new_cols:
                df_aligned = df_aligned.rename(columns=new_cols)

            combined = pd.concat(
                [combined, df_aligned],
                axis=1,
            )

        return combined
