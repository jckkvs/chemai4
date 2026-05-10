"""
backend/pipeline/__init__.py

ML パイプラインパッケージ。
5段階パイプライン（列選択 → 前処理 → 特徴量生成 → 特徴量選択 → 推定器）
を提供する。

Example:
    >>> from backend.pipeline import build_pipeline, PipelineConfig, ColumnMeta
    >>> config = PipelineConfig(
    ...     task="regression",
    ...     col_select_mode="exclude",
    ...     col_select_columns=["id", "date"],
    ...     column_meta={"age": ColumnMeta(monotonic=1, group="demographic")},
    ...     estimator_key="xgb",
    ...     apply_monotonic=True,
    ... )
    >>> pipe = build_pipeline(config)
    >>> pipe.fit(X_train, y_train)
    >>> predictions = pipe.predict(X_test)
"""
from backend.pipeline.column_selector import ColumnMeta, ColumnSelectorWrapper
from backend.pipeline.col_preprocessor import ColPreprocessConfig, ColPreprocessor
from backend.pipeline.feature_generator import FeatureGenConfig, FeatureGenerator
from backend.pipeline.feature_selector import FeatureSelectorConfig, FeatureSelector
from backend.pipeline.pipeline_builder import (
    PipelineConfig,
    build_pipeline,
    apply_monotonic_constraints,
    extract_group_array,
)
from backend.pipeline.pipeline_grid import (
    PipelineGridConfig,
    PipelineCombination,
    generate_pipeline_grid,
    count_combinations,
)

__all__ = [
    # 設定クラス
    "PipelineConfig",
    "ColPreprocessConfig",
    "FeatureGenConfig",
    "FeatureSelectorConfig",
    "ColumnMeta",
    # Transformer
    "ColumnSelectorWrapper",
    "ColPreprocessor",
    "FeatureGenerator",
    "FeatureSelector",
    # ビルダー関数
    "build_pipeline",
    "apply_monotonic_constraints",
    "extract_group_array",
    # グリッド展開
    "PipelineGridConfig",
    "PipelineCombination",
    "generate_pipeline_grid",
    "count_combinations",
]
