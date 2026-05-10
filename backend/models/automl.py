"""
backend/models/automl.py

AutoML エンジン。非専門家がワンボタンで機械学習を実行できるエンジン。
データ型自動判定 → 前処理 → 複数モデル学習 → 自動選択 → 結果返却。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.base import clone
from typing import Dict, Any

def _execute_prediction_pipeline(
    estimator, X_train: np.ndarray, y_arr: np.ndarray, cv_obj
) -> Dict[str, Any]:
    """
    予測値生成・指標計算の安全実行ラッパー。
    同期呼び出しを想定（フロントエンドで run.io_bound に委譲）。
    """
    if len(y_arr) == 0:
        raise ValueError("目的変数が空です")
    if np.ptp(y_arr) < 1e-9:
        return {
            "train_preds": getattr(estimator, "predict", lambda x: np.zeros(len(x)))(X_train),
            "oof_preds": np.full_like(y_arr, np.nan, dtype=float),
            "train_r2": float('nan'), "cv_r2": float('nan'), "gap": float('nan'),
            "y_true": y_arr
        }
    if X_train.shape[0] != len(y_arr):
        raise ValueError(f"特徴量行列と目的変数の行数が不一致: X={X_train.shape[0]}, y={len(y_arr)}")

    from sklearn.model_selection import cross_val_predict
    try:
        oof_preds = cross_val_predict(
            estimator, X_train, y_arr, cv=cv_obj, n_jobs=1,
            method="predict"
        )
    except Exception as e:
        raise RuntimeError(f"CV予測に失敗: {e}")

    full_pipe = clone(estimator)
    full_pipe.fit(X_train, y_arr)
    train_preds = full_pipe.predict(X_train)

    train_r2 = r2_score(y_arr, train_preds)
    cv_r2 = r2_score(y_arr, oof_preds)
    gap = train_r2 - cv_r2

    return {
        "train_preds": train_preds,
        "oof_preds": oof_preds,
        "train_r2": float(train_r2),
        "cv_r2": float(cv_r2),
        "gap": float(gap),
        "y_true": y_arr,
        "best_pipeline": full_pipe
    }

from backend.data.type_detector import TypeDetector, DetectionResult
from backend.data.preprocessor import Preprocessor, PreprocessConfig, build_full_pipeline
from backend.models.factory import get_model, get_default_automl_models
from backend.models.cv_manager import CVConfig, run_cross_validation
from backend.chem.rdkit_adapter import RDKitAdapter
from backend.chem.smiles_transformer import SmilesDescriptorTransformer
from backend.utils.config import RANDOM_STATE, AUTOML_CV_FOLDS

logger = logging.getLogger(__name__)


@dataclass
class AutoMLResult:
    """AutoML実行結果を保持するデータクラス。"""
    task: str                          # "regression" | "classification"
    best_model_key: str
    best_pipeline: Pipeline
    best_score: float
    scoring: str
    model_scores: dict[str, float]     # {model_key: cv_mean_score}
    model_details: dict[str, dict]     # {model_key: {mean, std, fit_time}}
    detection_result: DetectionResult
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)
    processed_X: pd.DataFrame | None = None
    # SHAP解析・評価用: パイプライン適用前の特徴量と目的変数
    X_train: pd.DataFrame | None = None
    y_train: np.ndarray | None = None
    # CV の Out-Of-Fold 予測 (全データに対するCVの予測値)
    oof_predictions: np.ndarray | None = None
    oof_true: np.ndarray | None = None
    # Holdout (train/test split) の予測
    holdout_true: np.ndarray | None = None
    # Trainデータの予測 (全データでの学習後の予測値)
    train_predictions: np.ndarray | None = None
    train_true: np.ndarray | None = None
    # SMILES相関係数とTransformerの保持
    smiles_correlations: dict[str, float] = field(default_factory=dict)
    smiles_transformer: Any | None = None
    # 自動解決された単調性制約の保持
    resolved_constraints: dict[str, int] = field(default_factory=dict)
    # 不確実性と適用領域 (Applicability Domain) の保持
    ad_distance_cv: np.ndarray | None = None
    in_domain_cv: np.ndarray | None = None
    uncertainty_cv: np.ndarray | None = None
    # Preprocessing Transparency Report
    preprocess_report: Any | None = None

class AutoMLEngine:
    """
    AutoMLエンジン。

    Implements: 要件定義書 §3.11 AutoMLモード

    Args:
        task: "auto" | "regression" | "classification"
        cv_folds: CV分割数
        model_keys: 試すモデルのキーリスト（None時はデフォルトを使用）
        timeout_seconds: 全体のタイムアウト（秒）
        progress_callback: 進捗コールバック (step, total, message) -> None
    """

    def __init__(
        self,
        task: str = "auto",
        cv_folds: int = AUTOML_CV_FOLDS,
        cv_key: str = "auto",  # "auto" = kfold(regression) / stratified_kfold(classification)
        cv_groups_col: str | None = None,  # GroupKFold等で使うグループ列名
        model_keys: list[str] | None = None,
        model_params: dict[str, dict[str, Any]] | None = None,  # {model_key: {param: val}}
        preprocess_params: dict[str, Any] | None = None,  # PreprocessConfigの上書き
        timeout_seconds: int = 600,
        progress_callback: Callable[[int, int, str], None] | None = None,
        selected_descriptors: list[str] | None = None,
        active_engines: list[str] | None = None,
        monotonic_constraints_dict: dict[str, int] | None = None,
        column_meta_dict: dict | None = None,
        count_normalization: str = "density",
        auto_feature_selection: bool = False,
        # Morgan フィンガープリント オプション
        morgan_count: bool = False,
        morgan_radius: int = 2,
        morgan_bits: int = 2048,
        morgan_order_by_appearance: bool = False,
    ) -> None:
        self.task = task
        self.cv_folds = cv_folds
        self.cv_key = cv_key
        self.cv_groups_col = cv_groups_col
        self.model_keys = model_keys
        self.model_params = model_params or {}
        self.preprocess_params = preprocess_params or {}
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback or (lambda s, t, m: None)
        self.selected_descriptors = selected_descriptors
        self.active_engines = active_engines
        self.count_normalization = count_normalization
        self.monotonic_constraints_dict = monotonic_constraints_dict or {}
        self.column_meta_dict = column_meta_dict or {}  # ColumnMeta 辭書
        self.auto_feature_selection = auto_feature_selection
        # Morgan フィンガープリント オプション
        self.morgan_count = morgan_count
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.morgan_order_by_appearance = morgan_order_by_appearance

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        smiles_col: str | list[dict] | None = None,
        fraction_type: str = "wt",
        group_col: str | None = None,
        preprocess_config: PreprocessConfig | None = None,
        cv_extra_params: dict[str, Any] | None = None,
    ) -> AutoMLResult:
        """
        AutoML全フローを実行する。

        Args:
            df: 入力DataFrame
            target_col: 目的変数の列名
            smiles_col: SMILES列名（化合物データの場合）
            group_col: グループ列名（GroupKFold等で使用）
            preprocess_config: 前処理設定（省略時はデフォルト）
            cv_extra_params: CVスプリッタに渡す追加引数

        Returns:
            AutoMLResult インスタンス
        """
        start = time.time()
        warnings: list[str] = []
        total_steps = 6
        cv_extra_params = cv_extra_params or {}

        # Step 1: データ品質チェック
        self.progress_callback(1, total_steps, "データ品質チェック中...")
        self._check_data_quality(df, target_col, warnings)

        # 目的変数の欠損行を除去
        if df[target_col].isna().any():
            initial_len = len(df)
            df = df.dropna(subset=[target_col]).copy()
            logger.info(f"目的変数の欠損により {initial_len - len(df)} 行を除去しました。")

        # Step 2: 変数型判定（SMILES等を検出）
        self.progress_callback(2, total_steps, "変数型を自動判定中...")
        detector = TypeDetector()
        _detect_cols_to_drop = [target_col]
        if group_col and group_col in df.columns:
            _detect_cols_to_drop.append(group_col)
        if self.cv_groups_col and self.cv_groups_col in df.columns and self.cv_groups_col not in _detect_cols_to_drop:
            _detect_cols_to_drop.append(self.cv_groups_col)
        
        comps = []
        if isinstance(smiles_col, str):
            comps = [{"smiles_col": smiles_col}]
        elif isinstance(smiles_col, list):
            comps = smiles_col
            
        _smiles_cols_present = False
        for c in comps:
            s_col = c.get("smiles_col")
            if s_col and s_col in df.columns:
                _smiles_cols_present = True
                _detect_cols_to_drop.append(s_col)
                
        detection_result = detector.detect(df.drop(columns=_detect_cols_to_drop))

        # Step 3: タスク判定
        self.progress_callback(3, total_steps, "タスク種別を判定中...")
        task = self._infer_task(df[target_col]) if self.task == "auto" else self.task
        logger.info(f"タスク: {task}")

        # Step 4: 目的変数・特徴量の準備
        y = df[target_col].values
        _drop_cols = [target_col]

        groups = df[group_col].values if group_col and group_col in df.columns else None
        # cv_groups_col が指定されている場合はそちらを優先
        if self.cv_groups_col and self.cv_groups_col in df.columns:
            groups = df[self.cv_groups_col].values

        # グループ列を特徴量から除外（_leakage_group等がfeatureに混入するのを防止）
        if group_col and group_col in df.columns:
            _drop_cols.append(group_col)
        if self.cv_groups_col and self.cv_groups_col in df.columns and self.cv_groups_col not in _drop_cols:
            _drop_cols.append(self.cv_groups_col)

        X = df.drop(columns=_drop_cols)

        # 特徴量が1つも残っていない場合のチェック
        if X.shape[1] == 0:
            raise ValueError("学習に使用できる特徴量がありません。目的変数以外の列が存在するか確認してください。")

        # 自動特徴量選択 (auto_feature_selection)
        if getattr(self, "auto_feature_selection", False) and X.shape[1] > 2:
            self.progress_callback(3.5, total_steps, "特徴量の事前選択を実行中...")
            try:
                from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                from backend.utils.config import default_config
                score_func = f_classif if task == "classification" else f_regression
                
                # 事前選択は数値変数のみに対して適用
                num_cols = X.select_dtypes(include="number").columns.tolist()
                cat_cols = [c for c in X.columns if c not in num_cols]
                
                # ユーザー指定の最大記述子数などを加味した上限設定
                upper_limit = default_config.max_descriptor_selection
                
                if num_cols and len(num_cols) > upper_limit:
                    k_best = upper_limit
                    selector = SelectKBest(score_func=score_func, k=k_best)
                    # NaNがあるとfitでエラーになるため簡易補完してスコア計算
                    X_num_filled = X[num_cols].fillna(X[num_cols].median())
                    # 全てNaNの列はmedian()もNaNになるため0補完
                    X_num_filled = X_num_filled.fillna(0.0)
                    selector.fit(X_num_filled, y)
                    selected_num_cols = np.array(num_cols)[selector.get_support()].tolist()
                    
                    dropped_cols = set(num_cols) - set(selected_num_cols)
                    if dropped_cols:
                        logger.info(f"auto_feature_selection: {len(dropped_cols)}列の数値変数を事前除外。")
                        X = X[selected_num_cols + cat_cols]
            except Exception as e:
                logger.warning(f"自動特徴量選択に失敗しました: {e}")

        # Step 5: モデル学習
        self.progress_callback(4, total_steps, "複数モデルで学習中...")
        model_keys = self.model_keys if self.model_keys else get_default_automl_models(task)
        if not model_keys:
             raise ValueError("学習に使用するモデルが指定されておらず、デフォルトも取得できませんでした。")

        scoring = self._get_scoring(task)
        # cv_key 自動決定: ユーザー指定全優先、"auto"の場合はタスクに応じて自動選択
        if self.cv_key == "auto":
            cv_key = "stratified_kfold" if task == "classification" else "kfold"
        else:
            cv_key = self.cv_key

        if cv_key == "scaffold":
            _chosen_smiles_col = None
            if isinstance(smiles_col, str):
                _chosen_smiles_col = smiles_col
            elif isinstance(smiles_col, list) and len(smiles_col) > 0:
                _chosen_smiles_col = smiles_col[0].get("smiles_col")
            
            if _chosen_smiles_col and _chosen_smiles_col in df.columns:
                try:
                    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
                    def get_scaffold(s):
                        try:
                            if pd.isna(s): return "unknown"
                            return MurckoScaffoldSmiles(str(s))
                        except Exception:
                            return "unknown"
                    groups = df[_chosen_smiles_col].apply(get_scaffold).values
                    logger.info(f"Scaffold Split groups computed from {_chosen_smiles_col}.")
                except ImportError:
                    logger.warning("RDKit is not installed. Falling back to KFold.")
                    cv_key = "stratified_kfold" if task == "classification" else "kfold"
            else:
                logger.warning("SMILES column not found for Scaffold Split. Falling back to KFold.")
                cv_key = "stratified_kfold" if task == "classification" else "kfold"

        # GroupKFold系の場合: グループ数 >= n_splits のバリデーション
        if cv_key in ("group_kfold", "leave_one_group_out", "scaffold") and groups is not None:
            n_unique_groups = len(np.unique(groups))
            if n_unique_groups < self.cv_folds:
                if n_unique_groups >= 2:
                    logger.warning(
                        f"{cv_key}: グループ数({n_unique_groups}) < n_splits({self.cv_folds})。"
                        f"n_splitsを{n_unique_groups}に自動調整します。"
                    )
                    self.cv_folds = n_unique_groups
                else:
                    logger.warning(
                        f"{cv_key}: グループ数({n_unique_groups})が不足。通常KFoldにフォールバックします。"
                    )
                    cv_key = "stratified_kfold" if task == "classification" else "kfold"
                    groups = None

        model_scores: dict[str, float] = {}
        model_details: dict[str, dict[str, Any]] = {}
        best_key = ""
        best_score = float("-inf")
        preprocess_cfg = preprocess_config or PreprocessConfig()
        deadline = start + self.timeout_seconds

        # ── SMILES列が存在する場合: 先にfit_transformして記述子DFを取得。
        # ── これによりTypeDetector・build_full_pipelineが記述子列を正しく認識できる。
        # ── 従来の「SmilesTransformer先頭挿入→変換前DFのdetection_result使用」という
        # ── 設計は TypeDetector が変換後の記述子列を認識できずColumnTransformerが空になるバグがあった。
        _smiles_transformer_for_cv: SmilesDescriptorTransformer | None = None
        X_train = X.copy()

        if _smiles_cols_present:
            logger.info(f"SMILES成分を事前変換して記述子DFを構築します: {comps}")
            _smiles_transformer_for_cv = SmilesDescriptorTransformer(
                smiles_col=comps,
                selected_descriptors=self.selected_descriptors,
                count_normalization=self.count_normalization,
                fraction_type=fraction_type,
                morgan_count=self.morgan_count,
                morgan_radius=self.morgan_radius,
                morgan_bits=self.morgan_bits,
                morgan_order_by_appearance=self.morgan_order_by_appearance,
            )
            try:
                X_train = _smiles_transformer_for_cv.fit_transform(X_train)
                logger.info(f"SMILES変換後のDF: {X_train.shape[1]}列")
                # 変換後のDFで TypeDetector を再実行 → detection_resultを更新
                detector_post = TypeDetector()
                detection_result = detector_post.detect(X_train)
                logger.info(f"SMILES変換後のTypeDetection結果: "
                           f"numeric={len(detection_result.numeric_columns)}列, "
                           f"categorical={len(detection_result.categorical_columns)}列")
            except Exception as _e:
                logger.warning(f"SMILES事前変換に失敗: {_e}。元のDFで続行します。")
                _smiles_transformer_for_cv = None
                X_train = X.copy()

        for i, mkey in enumerate(model_keys):
            if time.time() > deadline:
                warnings.append(f"タイムアウトにより {mkey} 以降のモデルをスキップしました。")
                break

            self.progress_callback(
                4, total_steps,
                f"学習中... ({i + 1}/{len(model_keys)}: {mkey})"
            )
            try:
                model_inst = get_model(mkey, task=task, **self.model_params.get(mkey, {}))
                
                # パイプラインを先に構築し、前処理器を fit() してS4特徴量空間を取得する
                pipeline_base = build_full_pipeline(
                    detection_result, model_inst,
                    target_col=target_col,
                    config=preprocess_cfg,
                )

                if self.monotonic_constraints_dict:
                    try:
                        from backend.models.monotonicity_adapter import apply_monotonicity_constraints
                        
                        preprocessor_step = pipeline_base.named_steps["preprocess"]
                        preprocessor_step.fit(X_train)

                        constrained_model = apply_monotonicity_constraints(
                            estimator=model_inst,
                            pipeline=preprocessor_step,
                            constraints_dict=self.monotonic_constraints_dict
                        )
                        # パイプラインの最終step (model) を置換
                        pipeline_base.steps[-1] = ("model", constrained_model)

                    except Exception as _e:
                        logger.warning(f"単調性制約適用をスキップ ({mkey}): {_e}", exc_info=True)
                # SMILES列があった場合: CV実行は変換済みX_trainで行う（smiles_varsなし）。
                # ── 理由: fold毎にSMILS変換すると、fold内の分子セットによって生成される記述子列が
                # ── 変わり「A given column is not a column of the dataframe」KeyErrorが発生する。
                # ── 事前変換済みX_trainはfitと同じ列セットを保証する。
                pipeline = pipeline_base  # smiles_varsなしでCV実行
                X_for_cv = X_train        # 変換済みDF
                cv_config = CVConfig(
                    cv_key=cv_key,
                    n_splits=self.cv_folds,
                    extra_params=cv_extra_params
                )
                result = run_cross_validation(
                    pipeline, X_for_cv, y, cv_config,
                    scoring=scoring,
                    groups=groups,
                    n_jobs=1,
                )
                score_key = f"test_{scoring}"
                if score_key in result:
                    mean_s = float(np.mean(result[score_key]))
                    std_s = float(np.std(result[score_key]))
                else:
                    mean_s = result.get("mean_test_score", 0.0)
                    std_s = result.get("std_test_score", 0.0)

                model_scores[mkey] = mean_s
                fold_scores_list = result[score_key].tolist() if score_key in result else []
                model_details[mkey] = {
                    "mean": mean_s,
                    "std": std_s,
                    "fit_time": float(np.mean(result.get("fit_time", [0]))),
                    "fold_scores": fold_scores_list,
                }
                if mean_s > best_score:
                    best_score = mean_s
                    best_key = mkey
                logger.info(f"  {mkey}: {mean_s:.4f} ± {std_s:.4f}")
            except Exception as e:
                msg = f"{mkey} の学習中にエラー: {str(e)}"
                logger.warning(msg)
                warnings.append(msg)
                import traceback
                logger.debug(traceback.format_exc())

        if not best_key:
            err_details = "\n".join(warnings[-min(len(warnings), 5):])
            raise RuntimeError(f"全モデルの学習に失敗しました（特徴量が全て除去された可能性があります）。詳細:\n{err_details}")

        # Step 6: 最良モデルを全データで再学習
        self.progress_callback(5, total_steps, f"最良モデル({best_key})を全データで学習中...")
        best_model = get_model(best_key, task=task, **self.model_params.get(best_key, {}))
        # 最良モデルにも単調性制約を適用
        if self.monotonic_constraints_dict or self.column_meta_dict:
            try:
                from backend.pipeline.column_selector import ColumnMeta
                from backend.pipeline.pipeline_builder import apply_monotonic_constraints
                _col_meta_best: dict[str, ColumnMeta] = {}
                for col in X_train.columns:
                    mono_val = self.monotonic_constraints_dict.get(col, 0)
                    _col_meta_best[col] = ColumnMeta(monotonic=mono_val)
                for col, meta in self.column_meta_dict.items():
                    if col in X_train.columns:
                        if isinstance(meta, ColumnMeta):
                            _col_meta_best[col] = meta
                        elif isinstance(meta, dict):
                            _col_meta_best[col] = ColumnMeta.from_dict(meta)
                best_model = apply_monotonic_constraints(
                    best_model, _col_meta_best,
                    feature_names=list(X_train.columns)
                )
            except Exception as _e:
                logger.warning(f"最良モデルへの単調性制約適用をスキップ ({best_key}): {_e}")
        best_pipeline_base = build_full_pipeline(
            detection_result, best_model,
            target_col=target_col,
            config=preprocess_cfg,
        )
        # SMILES変換済みTransformerがある場合はbest_pipelineにも先頭挿入
        if _smiles_transformer_for_cv is not None:
            from sklearn.base import clone as sklearn_clone
            best_pipeline = Pipeline([
                ("smiles_vars", sklearn_clone(_smiles_transformer_for_cv)),
                ("main_pipe", best_pipeline_base)
            ])
            best_pipeline.fit(X, y)   # 元のDF（SMILES列含む）で全データ学習
            X_for_eda = X_train       # EDA用は変換後DFを使用
        else:
            best_pipeline = best_pipeline_base
            best_pipeline.fit(X_train, y)
            X_for_eda = X_train

        # パイプラインの前処理部分(estimator以外)でtransformし、
        # 「実際にモデルに入力された最終データ」を取得する
        processed_X_final: pd.DataFrame | None = None
        preproc_report = None
        try:
            from backend.preprocessing.report import PreprocessingReport
            preproc_report = PreprocessingReport()
            preproc_report.original_features_count = X_for_eda.shape[1]
            if preprocess_cfg:
                preproc_report.scaler_used = preprocess_cfg.numeric_scaler
                
            # Pipeline[-1]がestimator。Pipeline[:-1]が前処理ステップ群。
            preprocessor_steps = best_pipeline[:-1]
            X_transformed = preprocessor_steps.transform(X_for_eda)
            # 特徴量名の取得
            try:
                feat_names = preprocessor_steps.get_feature_names_out().tolist()
            except Exception:
                n_cols = X_transformed.shape[1] if hasattr(X_transformed, "shape") else len(X_transformed[0])
                feat_names = [f"feature_{i}" for i in range(n_cols)]
            # sparse → dense変換
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            processed_X_final = pd.DataFrame(
                X_transformed, columns=feat_names, index=X_for_eda.index
            )
            preproc_report.final_features_count = processed_X_final.shape[1]
            
            # Simple difference heuristic
            dropped = set(X_for_eda.columns) - set(processed_X_final.columns)
            preproc_report.dropped_cols = list(dropped)
            num_cols = X_for_eda.select_dtypes(include='number').columns
            if X_for_eda[num_cols].isna().any().any():
                preproc_report.missing_value_handled = True
                preproc_report.imputations = {c: preprocess_cfg.numerical_imputer for c in num_cols if X_for_eda[c].isna().any()}

        except Exception as e:
            logger.warning(f"前処理後データの取得に失敗: {e}")
            processed_X_final = X_for_eda  # フォールバック: 変換済データ

        # OOF予測と全データ(Train)予測の計算
        oof_preds: np.ndarray | None = None
        train_preds: np.ndarray | None = None
        try:
            from sklearn.model_selection import cross_val_predict
            from backend.models.cv_manager import get_cv
            _cv_splitter = get_cv(CVConfig(cv_key=cv_key, n_splits=self.cv_folds, extra_params=cv_extra_params))
            _cv_method = "predict_proba" if task == "classification" and hasattr(best_pipeline, "predict_proba") else "predict"
            
            # 1. CV (OOF) 予測
            oof_preds = cross_val_predict(
                best_pipeline, X_for_eda if _smiles_transformer_for_cv is None else X, y,
                cv=_cv_splitter, method=_cv_method, n_jobs=1,
                groups=groups,
            )
            if _cv_method == "predict_proba" and oof_preds.ndim == 2:
                oof_preds = oof_preds.argmax(axis=1)

            # 2. 全データ (Train) 予測
            try:
                # すでに全データで学習済み(fit)の best_pipeline を使用
                predict_func = getattr(best_pipeline, _cv_method)
                train_preds = predict_func(X_for_eda if _smiles_transformer_for_cv is None else X)
                if _cv_method == "predict_proba" and train_preds.ndim == 2:
                    train_preds = train_preds.argmax(axis=1)
            except Exception as e:
                logger.warning(f"全データ予測の計算に失敗: {e}")

        except Exception as e:
            logger.warning(f"OOF予測の計算に失敗: {e}")
            oof_preds = None

        # ── AD (適用領域) と不確実性の計算 ──
        ad_distance_cv: np.ndarray | None = None
        in_domain_cv: np.ndarray | None = None
        uncertainty_cv: np.ndarray | None = None
        if processed_X_final is not None and oof_preds is not None:
            try:
                from backend.metrics.uncertainty import compute_ad_and_uncertainty
                X_mat = processed_X_final.values
                # 今回はCV全データに対して、自身を学習データとしてADを計算する (PCA Mahalanobis)
                ad_res = compute_ad_and_uncertainty(X_new=X_mat, X_train=X_mat, predictions=oof_preds)
                ad_distance_cv = ad_res.get("ad_distance")
                in_domain_cv = ad_res.get("in_domain")
                uncertainty_cv = ad_res.get("uncertainty")
            except Exception as e:
                logger.warning(f"AD/Uncertainty計算に失敗: {e}")

        self.progress_callback(6, total_steps, "完了!")
        elapsed = time.time() - start

        logger.info(
            f"AutoML完了: {elapsed:.1f}秒 / 最良モデル={best_key} / score={best_score:.4f}"
        )

        # SMILES記述子と目的変数の相関を計算
        _smiles_correlations: dict[str, float] = {}
        if _smiles_transformer_for_cv is not None:
            try:
                _target_series = pd.Series(y, index=X_train.index)
                _num_cols = X_train.select_dtypes(include="number").columns.tolist()
                if _num_cols and pd.api.types.is_numeric_dtype(_target_series):
                    _corr = X_train[_num_cols].corrwith(_target_series).dropna()
                    _smiles_correlations = _corr.to_dict()
            except Exception as _e:
                logger.debug(f"SMILES相関計算失敗: {_e}")

        # 最良モデルから、自動検出済みの単調性制約を抽出
        resolved_constraints: dict[str, int] = {}
        try:
            best_estimator = best_pipeline.steps[-1][1]
            if hasattr(best_estimator, "resolved_constraints_"):
                res = best_estimator.resolved_constraints_
                if res and isinstance(res, tuple):
                    feat_names_final = processed_X_final.columns.tolist() if processed_X_final is not None else []
                    if len(feat_names_final) == len(res):
                        for col, val in zip(feat_names_final, res):
                            if val in (1, -1):
                                resolved_constraints[col] = val
        except Exception as _e:
            logger.debug(f"自動検出済みの単調性制約抽出に失敗: {_e}")

        return AutoMLResult(
            task=task,
            best_model_key=best_key,
            best_pipeline=best_pipeline,
            best_score=best_score,
            scoring=scoring,
            model_scores=model_scores,
            model_details=model_details,
            detection_result=detection_result,
            elapsed_seconds=elapsed,
            warnings=warnings,
            processed_X=processed_X_final,
            X_train=X_train,
            y_train=y,
            oof_predictions=oof_preds,
            oof_true=y if oof_preds is not None else None,
            train_predictions=train_preds,
            train_true=y if train_preds is not None else None,
            smiles_transformer=_smiles_transformer_for_cv,
            smiles_correlations=_smiles_correlations,
            resolved_constraints=resolved_constraints,
            preprocess_report=preproc_report,
            ad_distance_cv=ad_distance_cv,
            in_domain_cv=in_domain_cv,
            uncertainty_cv=uncertainty_cv,
        )

    def run_multi_feature_sets(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_sets: list[dict],
        smiles_col: str | None = None,
        group_col: str | None = None,
        cv_extra_params: dict[str, Any] | None = None,
        progress_callback_outer: Callable[[int, int, str], None] | None = None,
    ) -> list["AutoMLResult"]:
        """
        複数の特徴量セット × パイプライン を順に実行して結果リストを返す。

        Args:
            df: 入力DataFrame（全記述子列を含む）
            target_col: 目的変数列名
            feature_sets: 特徴量セット定義リスト。各要素:
                {
                    "id": str,
                    "name": str,
                    "descriptors": list[str],  # 使用する記述子列名
                    "pipeline": "normal" | "highdim",
                    "rp_eps": float,         # JL歪み許容誤差（highdimのみ）
                }
            smiles_col: SMILES列名
            group_col: グループ列名
            cv_extra_params: CVスプリッタ追加引数
            progress_callback_outer: 外部進捗コールバック (set_idx, total_sets, msg)

        Returns:
            AutoMLResult のリスト（各feature_setに対応）
            ※ AutoMLResult.warnings[0] に feature_set名を付与する。
        """
        from backend.data.preprocessor import PreprocessConfig

        outer_cb = progress_callback_outer or (lambda s, t, m: None)
        results: list[AutoMLResult] = []
        n_sets = len(feature_sets)

        for idx, fs in enumerate(feature_sets):
            fs_name = fs.get("name", f"セット{idx+1}")
            outer_cb(idx + 1, n_sets, f"[{idx+1}/{n_sets}] {fs_name} を解析中...")
            logger.info(f"=== feature_set [{idx+1}/{n_sets}]: {fs_name} ===")

            # 記述子フィルタリング: fs["descriptors"] に含まれる列だけを使用
            desc_cols = fs.get("descriptors", [])
            if desc_cols:
                # 存在する列だけ使用（SMILES列・目的変数は別途処理）
                keep = set(desc_cols)
                available = set(df.columns) - {target_col}
                
                comps = []
                if isinstance(smiles_col, str):
                    comps = [{"smiles_col": smiles_col}]
                elif isinstance(smiles_col, list):
                    comps = smiles_col
                    
                for c in comps:
                    s_col = c.get("smiles_col")
                    if s_col:
                        available -= {s_col}
                        
                if group_col:
                    available -= {group_col}
                valid_desc = [c for c in desc_cols if c in available]
                # 有効な記述子 + 目的変数 + その他必要列でサブDF構築
                base_cols = [target_col]
                for c in comps:
                    s_col = c.get("smiles_col")
                    if s_col and s_col in df.columns:
                        base_cols.append(s_col)
                    f_col = c.get("fraction_col")
                    if f_col and f_col in df.columns:
                        base_cols.append(f_col)
                        
                if group_col and group_col in df.columns:
                    base_cols.append(group_col)
                use_cols = base_cols + valid_desc
                df_sub = df[[c for c in use_cols if c in df.columns]].copy()
            else:
                df_sub = df.copy()

            # パイプライン設定
            pipeline_type = fs.get("pipeline", "normal")
            rp_eps = float(fs.get("rp_eps", 0.1))
            preprocess_cfg = PreprocessConfig(
                random_projection_enable=(pipeline_type == "highdim"),
                random_projection_eps=rp_eps,
                random_projection_method="auto",
            )

            # 内部進捗コールバック（外部コールバックにラップ）
            def _inner_cb(step, total, msg, fn=fs_name, oi=idx, on=n_sets):
                outer_cb(oi + 1, on, f"[{oi+1}/{on}] {fn}: {msg}")

            # selected_descriptors を一時的にセット固有のものに変更
            orig_desc = self.selected_descriptors
            if desc_cols:
                self.selected_descriptors = valid_desc if valid_desc else None

            try:
                result = self.run(
                    df=df_sub,
                    target_col=target_col,
                    smiles_col=smiles_col,
                    group_col=group_col,
                    preprocess_config=preprocess_cfg,
                    cv_extra_params=cv_extra_params,
                )
                # セット情報をwarningsに付与（後でUIが参照）
                result.warnings.insert(0, f"__feature_set_name__:{fs_name}")
                result.warnings.insert(1, f"__feature_set_pipeline__:{pipeline_type}")
                results.append(result)
                logger.info(
                    f"  {fs_name} 完了: best={result.best_model_key} "
                    f"score={result.best_score:.4f}"
                )
            except Exception as e:
                logger.error(f"  {fs_name} 失敗: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # 失敗セットはダミー結果として記録
            finally:
                self.selected_descriptors = orig_desc

        outer_cb(n_sets, n_sets, f"全{n_sets}セット完了")
        return results

    @staticmethod
    def _infer_task(y_series: pd.Series) -> str:
        """目的変数から回帰/分類を自動判定する。"""
        if pd.api.types.is_float_dtype(y_series):
            return "regression"
        if pd.api.types.is_integer_dtype(y_series):
            n_unique = y_series.nunique()
            threshold = max(10, int(0.05 * len(y_series)))
            return "classification" if n_unique <= threshold else "regression"
        # 文字列/カテゴリ → 分類
        return "classification"

    @staticmethod
    def _get_scoring(task: str) -> str:
        """タスク種別に応じたデフォルトscoring文字列を返す。"""
        if task == "regression":
            return "neg_root_mean_squared_error"
        return "f1_weighted"

    @staticmethod
    def _check_data_quality(
        df: pd.DataFrame,
        target_col: str,
        warnings: list[str],
    ) -> None:
        """データ品質の基本チェックを実施して警告リストに追記する。"""
        if len(df) < 10:
            raise ValueError(f"データが少なすぎます（{len(df)}行）。最低10行必要です。")
        if target_col not in df.columns:
            raise ValueError(f"目的変数列 '{target_col}' が存在しません。")

        null_rate = df[target_col].isna().mean()
        if null_rate > 0:
            warnings.append(f"目的変数 '{target_col}' に欠損値が {null_rate:.1%} あります。欠損行を除外します。")

        dup_rate = df.duplicated().mean()
        if dup_rate > 0.05:
            warnings.append(f"重複行が {dup_rate:.1%} あります。")
