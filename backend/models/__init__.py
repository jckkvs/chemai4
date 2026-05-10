"""backend/models/__init__.py"""
from backend.models.factory import get_model, list_models, get_default_automl_models, get_model_registry
from backend.models.cv_manager import (
    get_cv, list_cv_methods, run_cross_validation, CVConfig, WalkForwardSplit
)

# フルスクラッチ実装モデル
try:
    from backend.models.linear_tree import (
        LinearTreeRegressor, LinearTreeClassifier,
        LinearForestRegressor, LinearForestClassifier,
        LinearBoostRegressor, LinearBoostClassifier,
        RidgeTreeRegressor, RidgeTreeClassifier,
    )
except ImportError:
    pass

try:
    from backend.models.rgf import RGFRegressor, RGFClassifier
except ImportError:
    pass

try:
    from backend.models.monotonic_kernel import (
        MonotonicKernelWrapper,
        MonotonicKernelClassifierWrapper,
        wrap_with_soft_monotonic,
        is_soft_monotonic_candidate,
    )
except ImportError:
    pass

__all__ = [
    "get_model",
    "list_models",
    "get_model_registry",
    "get_default_automl_models",
    "get_cv",
    "list_cv_methods",
    "run_cross_validation",
    "CVConfig",
    "WalkForwardSplit",
    # フルスクラッチ
    "LinearTreeRegressor", "LinearTreeClassifier",
    "LinearForestRegressor", "LinearForestClassifier",
    "LinearBoostRegressor", "LinearBoostClassifier",
    "RidgeTreeRegressor", "RidgeTreeClassifier",
    "RGFRegressor", "RGFClassifier",
    "MonotonicKernelWrapper", "MonotonicKernelClassifierWrapper",
    "wrap_with_soft_monotonic", "is_soft_monotonic_candidate",
]
