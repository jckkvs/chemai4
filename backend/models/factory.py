"""
backend/models/factory.py

機械学習モデルのファクトリーモジュール。
新モデルの追加はこのファイルの辞書への登録のみで完結する設計（Open/Closed原則）。
"""
from __future__ import annotations

import logging
from typing import Any

from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, Lasso, LassoCV,
    ElasticNet, ElasticNetCV, BayesianRidge,
    HuberRegressor, TheilSenRegressor, RANSACRegressor,
    ARDRegression,
    LogisticRegression,
)
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    BaggingRegressor, BaggingClassifier,
)
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    ConstantKernel,
    DotProduct,
    WhiteKernel,
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cross_decomposition import PLSRegression

from backend.utils.config import RANDOM_STATE
from backend.utils.optional_import import safe_import

logger = logging.getLogger(__name__)

# オプショナルライブラリ
_xgb         = safe_import("xgboost", "xgboost")
_lgb         = safe_import("lightgbm", "lightgbm")
_cat         = safe_import("catboost", "catboost")
_imodels     = safe_import("imodels", "imodels")

# フルスクラッチモデル（常時利用可能）
try:
    from backend.models.linear_tree import (
        LinearTreeRegressor, LinearTreeClassifier,
        LinearForestRegressor, LinearForestClassifier,
        LinearBoostRegressor, LinearBoostClassifier,
        RidgeTreeRegressor, RidgeTreeClassifier,
    )
    _linear_tree_available = True
except ImportError as e:
    _linear_tree_available = False
    logger.warning(f"linear_tree モジュールロード失敗: {e}")

try:
    from backend.models.rgf import RGFRegressor, RGFClassifier
    _rgf_available = True
except (ImportError, Exception) as e:
    _rgf_available = False
    RGFRegressor = None  # type: ignore[assignment,misc]
    RGFClassifier = None  # type: ignore[assignment,misc]
    logger.warning(f"rgf モジュールロード失敗: {e}")


def _xgb_regressor(**kw: Any) -> Any:
    from xgboost import XGBRegressor  # type: ignore
    return XGBRegressor(random_state=RANDOM_STATE, eval_metric="rmse", **kw)


def _xgb_classifier(**kw: Any) -> Any:
    from xgboost import XGBClassifier  # type: ignore
    return XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", **kw)


def _xgbrf_regressor(**kw: Any) -> Any:
    from xgboost import XGBRFRegressor  # type: ignore
    return XGBRFRegressor(random_state=RANDOM_STATE, eval_metric="rmse", **kw)


def _xgbrf_classifier(**kw: Any) -> Any:
    from xgboost import XGBRFClassifier  # type: ignore
    return XGBRFClassifier(random_state=RANDOM_STATE, eval_metric="logloss", **kw)


def _lgb_regressor(**kw: Any) -> Any:
    from lightgbm import LGBMRegressor  # type: ignore
    return LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **kw)


def _lgb_classifier(**kw: Any) -> Any:
    from lightgbm import LGBMClassifier  # type: ignore
    return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, **kw)


def _cat_regressor(**kw: Any) -> Any:
    from catboost import CatBoostRegressor  # type: ignore
    return CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, **kw)


def _cat_classifier(**kw: Any) -> Any:
    from catboost import CatBoostClassifier  # type: ignore
    return CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **kw)


# imodels ファクトリー関数
def _figs_regressor(**kw: Any) -> Any:
    from imodels import FIGSRegressor  # type: ignore
    return FIGSRegressor(**kw)


def _figs_classifier(**kw: Any) -> Any:
    from imodels import FIGSClassifier  # type: ignore
    return FIGSClassifier(**kw)


def _hsr_regressor(**kw: Any) -> Any:
    from imodels import HSTreeRegressorCV  # type: ignore
    return HSTreeRegressorCV(**kw)


def _hsr_classifier(**kw: Any) -> Any:
    from imodels import HSTreeClassifierCV  # type: ignore
    return HSTreeClassifierCV(**kw)


def _rulefit_regressor(**kw: Any) -> Any:
    from imodels import RuleFitRegressor  # type: ignore
    return RuleFitRegressor(**kw)


def _rulefit_classifier(**kw: Any) -> Any:
    from imodels import RuleFitClassifier  # type: ignore
    return RuleFitClassifier(**kw)


def _skoperules_classifier(**kw: Any) -> Any:
    from imodels import SkopeRulesClassifier  # type: ignore
    return SkopeRulesClassifier(**kw)


def _greedytree_regressor(**kw: Any) -> Any:
    from imodels import GreedyTreeRegressor  # type: ignore
    return GreedyTreeRegressor(**kw)


def _greedytree_classifier(**kw: Any) -> Any:
    from imodels import GreedyTreeClassifier  # type: ignore
    return GreedyTreeClassifier(**kw)


# LinearTree ファクトリー関数 (機能別に base_estimator をエクスポーズ)
def _linear_tree_regressor(**kw: Any) -> Any:
    from sklearn.linear_model import Ridge
    base = kw.pop("base_estimator", Ridge(alpha=kw.pop("alpha", 1.0)))
    return LinearTreeRegressor(base_estimator=base, **kw)


def _linear_tree_classifier(**kw: Any) -> Any:
    from sklearn.linear_model import LogisticRegression
    base = kw.pop("base_estimator", LogisticRegression(max_iter=500))
    return LinearTreeClassifier(base_estimator=base, **kw)


def _linear_forest_regressor(**kw: Any) -> Any:
    from sklearn.linear_model import Ridge
    base = kw.pop("base_estimator", Ridge(alpha=kw.pop("alpha", 1.0)))
    return LinearForestRegressor(base_estimator=base, **kw)


def _linear_forest_classifier(**kw: Any) -> Any:
    from sklearn.linear_model import LogisticRegression
    base = kw.pop("base_estimator", LogisticRegression(max_iter=500))
    return LinearForestClassifier(base_estimator=base, **kw)


def _linear_boost_regressor(**kw: Any) -> Any:
    from sklearn.linear_model import Ridge
    base = kw.pop("base_estimator", Ridge(alpha=kw.pop("alpha", 1.0)))
    return LinearBoostRegressor(base_estimator=base, **kw)


def _linear_boost_classifier(**kw: Any) -> Any:
    from sklearn.linear_model import LogisticRegression
    base = kw.pop("base_estimator", LogisticRegression(max_iter=500))
    return LinearBoostClassifier(base_estimator=base, **kw)


# ============================================================
# ★★★ NEW: GPR / GPC カーネル別ファクトリー関数 ★★★
# ============================================================
# 設計方針:
#   カーネルの種類はハイパーパラメータではなく、異なるモデルとして扱う。
#   各カーネルに対して個別のファクトリー関数とレジストリエントリを用意する。
#   全カーネルに WhiteKernel を加算して観測ノイズを自動推定する。

def _gpr_rbf(**kw: Any) -> GaussianProcessRegressor:
    """GPR with RBF kernel (+ WhiteKernel for noise estimation)."""
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_restarts_optimizer", 3)
    kw.setdefault("normalize_y", True)
    return GaussianProcessRegressor(kernel=kernel, **kw)


def _gpr_matern(**kw: Any) -> GaussianProcessRegressor:
    """GPR with Matérn kernel (nu=2.5, + WhiteKernel)."""
    nu = kw.pop("nu", 2.5)
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=nu) + WhiteKernel(noise_level=1.0)
    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_restarts_optimizer", 3)
    kw.setdefault("normalize_y", True)
    return GaussianProcessRegressor(kernel=kernel, **kw)


def _gpr_ard(**kw: Any) -> GaussianProcessRegressor:
    """
    GPR with ARD (Automatic Relevance Determination) kernel.
    ARD は RBF の length_scale を特徴量ごとに独立に持たせることで実現する。
    n_features が未知の場合は length_scale=1.0 (スカラー) で初期化し、
    fit 時に sklearn が自動で ARD 化する。
    """
    n_features = kw.pop("n_features", 1)
    length_scale_init = kw.pop("length_scale", [1.0] * n_features)
    kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale_init) + WhiteKernel(noise_level=1.0)
    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_restarts_optimizer", 3)
    kw.setdefault("normalize_y", True)
    return GaussianProcessRegressor(kernel=kernel, **kw)


def _gpr_constant_rbf(**kw: Any) -> GaussianProcessRegressor:
    """GPR with ConstantKernel + RBF (amplitude + length_scale, + WhiteKernel)."""
    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_restarts_optimizer", 3)
    kw.setdefault("normalize_y", True)
    return GaussianProcessRegressor(kernel=kernel, **kw)


def _gpr_dotproduct(**kw: Any) -> GaussianProcessRegressor:
    """GPR with DotProduct kernel (linear-like, + WhiteKernel)."""
    sigma_0 = kw.pop("sigma_0", 1.0)
    kernel = ConstantKernel(1.0) * DotProduct(sigma_0=sigma_0) + WhiteKernel(noise_level=1.0)
    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_restarts_optimizer", 3)
    kw.setdefault("normalize_y", True)
    return GaussianProcessRegressor(kernel=kernel, **kw)


def _gpc_rbf(**kw: Any) -> GaussianProcessClassifier:
    """GPC with RBF kernel."""
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_restarts_optimizer", 3)
    return GaussianProcessClassifier(kernel=kernel, **kw)


def _gpc_matern(**kw: Any) -> GaussianProcessClassifier:
    """GPC with Matérn kernel (nu=2.5)."""
    nu = kw.pop("nu", 2.5)
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=nu)
    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_restarts_optimizer", 3)
    return GaussianProcessClassifier(kernel=kernel, **kw)


def _gpc_ard(**kw: Any) -> GaussianProcessClassifier:
    """GPC with ARD kernel (feature-wise length_scale)."""
    n_features = kw.pop("n_features", 1)
    length_scale_init = kw.pop("length_scale", [1.0] * n_features)
    kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale_init)
    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_restarts_optimizer", 3)
    return GaussianProcessClassifier(kernel=kernel, **kw)


def _gpc_constant_rbf(**kw: Any) -> GaussianProcessClassifier:
    """GPC with ConstantKernel + RBF."""
    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0)
    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_restarts_optimizer", 3)
    return GaussianProcessClassifier(kernel=kernel, **kw)


def _gpc_dotproduct(**kw: Any) -> GaussianProcessClassifier:
    """GPC with DotProduct kernel."""
    sigma_0 = kw.pop("sigma_0", 1.0)
    kernel = ConstantKernel(1.0) * DotProduct(sigma_0=sigma_0)
    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_restarts_optimizer", 3)
    return GaussianProcessClassifier(kernel=kernel, **kw)


# ============================================================
# 回帰モデルのレジストリ
# ============================================================
_REGRESSION_REGISTRY: dict[str, dict[str, Any]] = {
    "linear": {
        "name": "Linear Regression",
        "class": LinearRegression,
        "default_params": {},
        "available": True,
        "tags": ["linear", "interpretable"],
    },
    "ridge": {
        "name": "Ridge Regression",
        "class": Ridge,
        "default_params": {"alpha": 1.0, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "regularized"],
    },
    "ridge_cv": {
        "name": "Ridge CV",
        "class": RidgeCV,
        "default_params": {},
        "available": True,
        "tags": ["linear", "regularized", "cv"],
    },
    "lasso": {
        "name": "Lasso",
        "class": Lasso,
        "default_params": {"alpha": 1.0, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "regularized", "sparse"],
    },
    "lasso_cv": {
        "name": "Lasso CV",
        "class": LassoCV,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "regularized", "cv"],
    },
    "elasticnet": {
        "name": "ElasticNet",
        "class": ElasticNet,
        "default_params": {"alpha": 1.0, "l1_ratio": 0.5, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "regularized"],
    },
    "elasticnet_cv": {
        "name": "ElasticNet CV",
        "class": ElasticNetCV,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "regularized", "cv"],
    },
    "bayesian_ridge": {
        "name": "Bayesian Ridge",
        "class": BayesianRidge,
        "default_params": {},
        "available": True,
        "tags": ["linear", "bayesian"],
    },
    "ard": {
        "name": "ARD Regression",
        "class": ARDRegression,
        "default_params": {},
        "available": True,
        "tags": ["linear", "bayesian", "sparse"],
    },
    "huber": {
        "name": "Huber Regressor",
        "class": HuberRegressor,
        "default_params": {},
        "available": True,
        "tags": ["linear", "robust"],
    },
    "theilsen": {
        "name": "TheilSen Regressor",
        "class": TheilSenRegressor,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "robust"],
    },
    "ransac": {
        "name": "RANSAC Regressor",
        "class": RANSACRegressor,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear", "robust"],
    },
    "svr_rbf": {
        "name": "SVR (RBF)",
        "class": SVR,
        "default_params": {"kernel": "rbf", "C": 1.0},
        "available": True,
        "tags": ["kernel", "nonlinear"],
    },
    "svr_linear": {
        "name": "SVR (Linear)",
        "class": SVR,
        "default_params": {"kernel": "linear", "C": 1.0},
        "available": True,
        "tags": ["kernel", "linear"],
    },
    "knn": {
        "name": "KNN Regressor",
        "class": KNeighborsRegressor,
        "default_params": {"n_neighbors": 5},
        "available": True,
        "tags": ["instance-based"],
    },
    "dt": {
        "name": "Decision Tree",
        "class": DecisionTreeRegressor,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["tree", "interpretable"],
    },
    "rf": {
        "name": "Random Forest",
        "class": RandomForestRegressor,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble", "tree"],
    },
    "et": {
        "name": "Extra Trees",
        "class": ExtraTreesRegressor,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble", "tree"],
    },
    "gbm": {
        "name": "Gradient Boosting",
        "class": GradientBoostingRegressor,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting"],
    },
    "hgbm": {
        "name": "HistGradientBoosting",
        "class": HistGradientBoostingRegressor,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting", "missing_ok"],
    },
    "adaboost": {
        "name": "AdaBoost",
        "class": AdaBoostRegressor,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting"],
    },
    "bagging": {
        "name": "Bagging Regressor",
        "class": BaggingRegressor,
        "default_params": {"n_estimators": 10, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble"],
    },
    "xgb": {
        "name": "XGBoost",
        "factory": _xgb_regressor,
        "default_params": {"n_estimators": 100},
        "available": bool(_xgb),
        "tags": ["ensemble", "boosting", "native_monotonic"],
    },
    "lgbm": {
        "name": "LightGBM",
        "factory": _lgb_regressor,
        "default_params": {"n_estimators": 100},
        "available": bool(_lgb),
        "tags": ["ensemble", "boosting", "native_monotonic"],
    },
    "catboost": {
        "name": "CatBoost",
        "factory": _cat_regressor,
        "default_params": {"iterations": 100},
        "available": bool(_cat),
        "tags": ["ensemble", "boosting"],
    },
    "mlp": {
        "name": "MLP Regressor",
        "class": MLPRegressor,
        "default_params": {
            "hidden_layer_sizes": (100, 50),
            "max_iter": 500,
            "random_state": RANDOM_STATE,
        },
        "available": True,
        "tags": ["neural_network"],
    },
    "gp": {
        "name": "GPR (RBF)",
        "factory": _gpr_rbf,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel"],
    },
    "gpr_rbf": {
        "name": "GPR (RBF)",
        "factory": _gpr_rbf,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel"],
    },
    "gpr_matern": {
        "name": "GPR (Matérn)",
        "factory": _gpr_matern,
        "default_params": {"nu": 2.5},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel"],
    },
    "gpr_ard": {
        "name": "GPR (ARD)",
        "factory": _gpr_ard,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel", "feature_selection"],
    },
    "gpr_constant_rbf": {
        "name": "GPR (Constant+RBF)",
        "factory": _gpr_constant_rbf,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel"],
    },
    "gpr_dotproduct": {
        "name": "GPR (DotProduct)",
        "factory": _gpr_dotproduct,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel", "linear"],
    },
    "pls": {
        "name": "PLS Regression",
        "class": PLSRegression,
        "default_params": {"n_components": 2},
        "available": True,
        "tags": ["linear", "dimensionality"],
    },
    "xgbrf": {
        "name": "XGBoost Random Forest (XGBRFRegressor)",
        "factory": _xgbrf_regressor,
        "default_params": {"n_estimators": 100},
        "available": bool(_xgb),
        "tags": ["ensemble", "tree", "rf", "native_monotonic"],
    },
    # ─── Linear Tree / Forest / Boost (フルスクラッチ) ───
    "lineartree": {
        "name": "Linear Tree Regressor (scratch)",
        "factory": _linear_tree_regressor,
        "default_params": {"max_depth": 5},
        "available": _linear_tree_available,
        "tags": ["tree", "linear", "interpretable"],
    },
    "linearforest": {
        "name": "Linear Forest Regressor (scratch)",
        "factory": _linear_forest_regressor,
        "default_params": {"n_estimators": 100, "max_depth": 5},
        "available": _linear_tree_available,
        "tags": ["ensemble", "tree", "linear"],
    },
    "linearboost": {
        "name": "Linear Boost Regressor (scratch)",
        "factory": _linear_boost_regressor,
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        "available": _linear_tree_available,
        "tags": ["ensemble", "boosting", "linear"],
    },
    "ridgetree": {
        "name": "Ridge Tree Regressor",
        "factory": lambda **kw: RidgeTreeRegressor(**kw),
        "default_params": {"max_depth": 5, "alpha": 1.0},
        "available": _linear_tree_available,
        "tags": ["tree", "linear", "interpretable"],
    },
}

# ─── RGF （フルスクラッチ、利用可能時のみレジストリに追加） ───
if _rgf_available:
    _REGRESSION_REGISTRY["rgf"] = {
        "name": "RGF Regressor (Regularized Greedy Forest, scratch)",
        "class": RGFRegressor,
        "default_params": {"n_estimators": 100, "max_leaf_nodes": 32, "lambda_l2": 1.0},
        "available": True,
        "tags": ["ensemble", "tree", "regularized"],
    }

# ─── imodels (回帰) ───
_REGRESSION_REGISTRY.update({
    "figs": {
        "name": "FIGS Regressor (imodels)",
        "factory": _figs_regressor,
        "default_params": {"max_rules": 12},
        "available": bool(_imodels),
        "tags": ["tree", "interpretable", "sparse"],
    },
    "hstree": {
        "name": "Hierarchical Shrinkage Tree CV (imodels)",
        "factory": _hsr_regressor,
        "default_params": {},
        "available": bool(_imodels),
        "tags": ["tree", "interpretable", "regularized"],
    },
    "rulefit": {
        "name": "RuleFit Regressor (imodels)",
        "factory": _rulefit_regressor,
        "default_params": {},
        "available": bool(_imodels),
        "tags": ["rule", "interpretable", "linear"],
    },
    "greedytree": {
        "name": "Greedy Tree Regressor (imodels)",
        "factory": _greedytree_regressor,
        "default_params": {"max_depth": 5},
        "available": bool(_imodels),
        "tags": ["tree", "interpretable"],
    },
})

# ============================================================
# 分類モデルのレジストリ
# ============================================================
_CLASSIFICATION_REGISTRY: dict[str, dict[str, Any]] = {
    "logistic": {
        "name": "Logistic Regression",
        "class": LogisticRegression,
        "default_params": {
            "max_iter": 1000,
            "solver": "liblinear",
            "random_state": RANDOM_STATE,
        },
        "available": True,
        "tags": ["linear", "interpretable"],
    },
    "svc_rbf": {
        "name": "SVC (RBF)",
        "class": SVC,
        "default_params": {
            "kernel": "rbf",
            "C": 1.0,
            "probability": True,
            "random_state": RANDOM_STATE,
        },
        "available": True,
        "tags": ["kernel", "nonlinear"],
    },
    "linearsvc": {
        "name": "LinearSVC",
        "class": LinearSVC,
        "default_params": {"C": 1.0, "max_iter": 2000, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["linear"],
    },
    "knn_c": {
        "name": "KNN Classifier",
        "class": KNeighborsClassifier,
        "default_params": {"n_neighbors": 5},
        "available": True,
        "tags": ["instance-based"],
    },
    "dt_c": {
        "name": "Decision Tree",
        "class": DecisionTreeClassifier,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["tree", "interpretable"],
    },
    "rf_c": {
        "name": "Random Forest",
        "class": RandomForestClassifier,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble", "tree"],
    },
    "et_c": {
        "name": "Extra Trees",
        "class": ExtraTreesClassifier,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble", "tree"],
    },
    "gbm_c": {
        "name": "Gradient Boosting",
        "class": GradientBoostingClassifier,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting"],
    },
    "hgbm_c": {
        "name": "HistGradientBoosting",
        "class": HistGradientBoostingClassifier,
        "default_params": {"random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting", "missing_ok"],
    },
    "adaboost_c": {
        "name": "AdaBoost",
        "class": AdaBoostClassifier,
        "default_params": {"n_estimators": 100, "random_state": RANDOM_STATE},
        "available": True,
        "tags": ["ensemble", "boosting"],
    },
    "bagging_c": {
        "name": "Bagging Classifier",
        "class": BaggingClassifier,
        "default_params": {"n_estimators": 10, "random_state": RANDOM_STATE, "n_jobs": -1},
        "available": True,
        "tags": ["ensemble"],
    },
    "xgb_c": {
        "name": "XGBoost",
        "factory": _xgb_classifier,
        "default_params": {"n_estimators": 100},
        "available": bool(_xgb),
        "tags": ["ensemble", "boosting"],
    },
    "lgbm_c": {
        "name": "LightGBM",
        "factory": _lgb_classifier,
        "default_params": {"n_estimators": 100},
        "available": bool(_lgb),
        "tags": ["ensemble", "boosting"],
    },
    "catboost_c": {
        "name": "CatBoost",
        "factory": _cat_classifier,
        "default_params": {"iterations": 100},
        "available": bool(_cat),
        "tags": ["ensemble", "boosting"],
    },
    "mlp_c": {
        "name": "MLP Classifier",
        "class": MLPClassifier,
        "default_params": {
            "hidden_layer_sizes": (100, 50),
            "max_iter": 500,
            "random_state": RANDOM_STATE,
        },
        "available": True,
        "tags": ["neural_network"],
    },
    "gnb": {
        "name": "Gaussian NB",
        "class": GaussianNB,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic"],
    },
    "bnb": {
        "name": "Bernoulli NB",
        "class": BernoulliNB,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic"],
    },
    "gp_c": {
        "name": "GPC (RBF)",
        "factory": _gpc_rbf,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel"],
    },
    "gpc_rbf": {
        "name": "GPC (RBF)",
        "factory": _gpc_rbf,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel"],
    },
    "gpc_matern": {
        "name": "GPC (Matérn)",
        "factory": _gpc_matern,
        "default_params": {"nu": 2.5},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel"],
    },
    "gpc_ard": {
        "name": "GPC (ARD)",
        "factory": _gpc_ard,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel", "feature_selection"],
    },
    "gpc_constant_rbf": {
        "name": "GPC (Constant+RBF)",
        "factory": _gpc_constant_rbf,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel"],
    },
    "gpc_dotproduct": {
        "name": "GPC (DotProduct)",
        "factory": _gpc_dotproduct,
        "default_params": {},
        "available": True,
        "tags": ["probabilistic", "gaussian_process", "kernel", "linear"],
    },
    "xgbrf_c": {
        "name": "XGBoost Random Forest (XGBRFClassifier)",
        "factory": _xgbrf_classifier,
        "default_params": {"n_estimators": 100},
        "available": bool(_xgb),
        "tags": ["ensemble", "tree", "rf", "native_monotonic"],
    },
    # ─── Linear Tree / Forest / Boost (フルスクラッチ) ───
    "lineartree_c": {
        "name": "Linear Tree Classifier (scratch)",
        "factory": _linear_tree_classifier,
        "default_params": {"max_depth": 5},
        "available": _linear_tree_available,
        "tags": ["tree", "linear", "interpretable"],
    },
    "linearforest_c": {
        "name": "Linear Forest Classifier (scratch)",
        "factory": _linear_forest_classifier,
        "default_params": {"n_estimators": 100, "max_depth": 5},
        "available": _linear_tree_available,
        "tags": ["ensemble", "tree", "linear"],
    },
    "linearboost_c": {
        "name": "Linear Boost Classifier (scratch)",
        "factory": _linear_boost_classifier,
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        "available": _linear_tree_available,
        "tags": ["ensemble", "boosting", "linear"],
    },
    "ridgetree_c": {
        "name": "Ridge Tree Classifier",
        "factory": lambda **kw: _linear_tree_classifier(**kw),
        "default_params": {"max_depth": 5},
        "available": _linear_tree_available,
        "tags": ["tree", "linear", "interpretable"],
    },
    # ─── imodels ───
    "figs_c": {
        "name": "FIGS Classifier (imodels)",
        "factory": _figs_classifier,
        "default_params": {"max_rules": 12},
        "available": bool(_imodels),
        "tags": ["tree", "interpretable", "sparse"],
    },
    "hstree_c": {
        "name": "Hierarchical Shrinkage Tree CV (imodels)",
        "factory": _hsr_classifier,
        "default_params": {},
        "available": bool(_imodels),
        "tags": ["tree", "interpretable", "regularized"],
    },
    "rulefit_c": {
        "name": "RuleFit Classifier (imodels)",
        "factory": _rulefit_classifier,
        "default_params": {},
        "available": bool(_imodels),
        "tags": ["rule", "interpretable", "linear"],
    },
    "skoperules_c": {
        "name": "SkopeRules Classifier (imodels)",
        "factory": _skoperules_classifier,
        "default_params": {},
        "available": bool(_imodels),
        "tags": ["rule", "interpretable", "sparse"],
    },
    "greedytree_c": {
        "name": "Greedy Tree Classifier (imodels)",
        "factory": _greedytree_classifier,
        "default_params": {"max_depth": 5},
        "available": bool(_imodels),
        "tags": ["tree", "interpretable"],
    },
}

# ─── RGF （フルスクラッチ、利用可能時のみレジストリに追加） ───
if _rgf_available:
    _CLASSIFICATION_REGISTRY["rgf_c"] = {
        "name": "RGF Classifier (Regularized Greedy Forest, scratch)",
        "class": RGFClassifier,
        "default_params": {"n_estimators": 100, "max_leaf_nodes": 32, "lambda_l2": 1.0},
        "available": True,
        "tags": ["ensemble", "tree", "regularized"],
    }


# ============================================================
# Factory API
# ============================================================

def get_model(
    model_key: str,
    task: str = "regression",
    **override_params: Any,
) -> Any:
    """
    指定された model_key に対応する学習済みモデルインスタンスを返す。

    Args:
        model_key: モデルのキー（例: "rf", "xgb", "logistic"）
        task: "regression" or "classification"
        **override_params: デフォルトパラメータを上書きするキーワード引数

    Returns:
        sklearn互換の推定器インスタンス

    Raises:
        ValueError: 未知のmodel_key、またはライブラリ未インストール
    """
    registry = _get_registry(task)
    if model_key not in registry:
        raise ValueError(
            f"未知のモデルキー '{model_key}' (task={task})。"
            f"利用可能: {list(registry.keys())}"
        )

    entry = registry[model_key]
    if not entry.get("available", True):
        raise ValueError(
            f"モデル '{model_key}' ({entry['name']}) は現在の環境で"
            f"ライブラリがインストールされていません。"
        )

    params = {**entry.get("default_params", {}), **override_params}

    if "factory" in entry:
        return entry["factory"](**params)
    else:
        return entry["class"](**params)


def list_models(
    task: str = "regression",
    available_only: bool = True,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    利用可能なモデルの一覧を返す。

    Args:
        task: "regression" or "classification"
        available_only: True の場合はインストール済みのみ返す
        tags: 指定した場合、そのタグを持つモデルのみ返す

    Returns:
        [{key, name, tags, available}] のリスト
    """
    registry = _get_registry(task)
    results = []
    for key, entry in registry.items():
        if available_only and not entry.get("available", True):
            continue
        if tags:
            model_tags = entry.get("tags", [])
            if not any(t in model_tags for t in tags):
                continue
        results.append({
            "key": key,
            "name": entry["name"],
            "class": entry.get("class"),
            "tags": entry.get("tags", []),
            "available": entry.get("available", True),
        })
    return results


def get_default_automl_models(task: str = "regression") -> list[str]:
    """
    AutoMLモードで使用するデフォルトモデルキーのリストを返す。
    使用可能なモデルから代表的なものを選択する。
    """
    regression_defaults = ["linear", "ridge_cv", "lasso_cv", "rf", "et", "gbm", "hgbm",
                           "xgb", "lgbm", "svr_rbf", "gpr_rbf"]
    classification_defaults = ["logistic", "rf_c", "et_c", "gbm_c", "hgbm_c",
                                "xgb_c", "lgbm_c", "svc_rbf", "knn_c", "dt_c", "gpc_rbf"]

    registry = _get_registry(task)
    defaults = regression_defaults if task == "regression" else classification_defaults
    return [k for k in defaults if k in registry and registry[k].get("available", True)]


def _get_registry(task: str) -> dict[str, dict[str, Any]]:
    """タスク種別に応じてレジストリを返す。"""
    if task == "regression":
        return _REGRESSION_REGISTRY
    elif task == "classification":
        return _CLASSIFICATION_REGISTRY
    else:
        raise ValueError(f"未知のタスク '{task}'。'regression' または 'classification' を指定してください。")


def get_model_registry(task: str = "regression") -> dict[str, dict[str, Any]]:
    """
    モデルのレジストリ（メタデータ全体）を返す。
    フロントエンドでの動的UI生成などに使用する。
    """
    return _get_registry(task)
