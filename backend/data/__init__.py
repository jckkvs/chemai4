"""backend/data/__init__.py"""
from backend.data.loader import load_file, load_from_bytes, save_dataframe
from backend.data.type_detector import TypeDetector, ColumnType, ColumnInfo, DetectionResult
from backend.data.preprocessor import Preprocessor, PreprocessConfig, build_full_pipeline
from backend.data.feature_engineer import (
    InteractionTransformer,
    GroupAggTransformer,
    DatetimeFeatureExtractor,
    LagRollingTransformer,
    FeatureEngineeringConfig,
    build_feature_engineering_pipeline,
)
from backend.data.eda import (
    ColumnStats,
    compute_column_stats,
    summarize_dataframe,
    compute_correlation,
    OutlierResult,
    detect_outliers,
    compute_distribution,
    analyze_target,
)
from backend.data.dim_reduction import (
    DimReductionConfig,
    DimReducer,
    run_pca,
    run_tsne,
    run_umap,
)
from backend.data.benchmark import (
    ModelScore,
    BenchmarkResult,
    evaluate_regression,
    evaluate_classification,
    compute_learning_curve,
    benchmark_models,
)

__all__ = [
    "load_file",
    "load_from_bytes",
    "save_dataframe",
    "TypeDetector",
    "ColumnType",
    "ColumnInfo",
    "DetectionResult",
    "Preprocessor",
    "PreprocessConfig",
    "build_full_pipeline",
    "InteractionTransformer",
    "GroupAggTransformer",
    "DatetimeFeatureExtractor",
    "LagRollingTransformer",
    "FeatureEngineeringConfig",
    "build_feature_engineering_pipeline",
    "ColumnStats",
    "compute_column_stats",
    "summarize_dataframe",
    "compute_correlation",
    "OutlierResult",
    "detect_outliers",
    "compute_distribution",
    "analyze_target",
    "DimReductionConfig",
    "DimReducer",
    "run_pca",
    "run_tsne",
    "run_umap",
    "ModelScore",
    "BenchmarkResult",
    "evaluate_regression",
    "evaluate_classification",
    "compute_learning_curve",
    "benchmark_models",
]
