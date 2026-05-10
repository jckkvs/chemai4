"""
backend/models/tuner.py

ハイパーパラメータ最適化モジュール。
GridSearch, RandomSearch, HalvingSearch, Optuna, BayesSearchCV に対応。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)

from backend.utils.config import RANDOM_STATE
from backend.utils.optional_import import safe_import

logger = logging.getLogger(__name__)

_optuna = safe_import("optuna", "optuna")
_skopt = safe_import("skopt", "scikit-optimize")

# HalvingSearchCV (sklearn 0.24+)
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
    _halving_available = True
except ImportError:
    _halving_available = False


def _convert_optuna_grid_for_random(param_grid: dict[str, Any]) -> dict[str, Any]:
    """Optuna用のdict形式param_gridをRandomizedSearchCV用に変換する。

    Optuna形式: {"alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True}}
    → RandomSearch形式: {"alpha": loguniform(0.01, 10.0)}
    """
    from scipy.stats import uniform, randint

    converted: dict[str, Any] = {}
    for name, spec in param_grid.items():
        if isinstance(spec, dict):
            suggest_type = spec.get("type", "float")
            low = spec.get("low", 0)
            high = spec.get("high", 1)
            if suggest_type == "float":
                if spec.get("log", False):
                    try:
                        from scipy.stats import loguniform
                        converted[name] = loguniform(low, high)
                    except ImportError:
                        # scipy < 1.6: loguniform unavailable
                        converted[name] = uniform(low, high - low)
                else:
                    converted[name] = uniform(low, high - low)
            elif suggest_type == "int":
                spec.get("step", 1)
                converted[name] = randint(low, high + 1)
            elif suggest_type == "categorical":
                converted[name] = spec.get("choices", [])
            else:
                converted[name] = spec
        else:
            # すでにリストや分布の場合はそのまま
            converted[name] = spec
    return converted


@dataclass
class TunerConfig:
    """チューニング設定。"""
    method: str = "random"
    # "grid" | "random" | "halving_grid" | "halving_random" | "optuna" | "bayes"
    param_grid: dict[str, Any] = field(default_factory=dict)
    n_iter: int = 50              # RandomSearch / Optuna の試行回数
    cv: int = 5                   # 内部CV folds
    scoring: str = "neg_root_mean_squared_error"
    n_jobs: int = -1
    verbose: int = 0
    refit: bool = True
    optuna_direction: str = "maximize"
    optuna_timeout: int | None = None
    random_state: int = RANDOM_STATE


# ============================================================
# Pipeline プレフィックスユーティリティ
# ============================================================

def _detect_pipeline_step_name(model: Any) -> str | None:
    """
    modelがsklearn Pipelineの場合、estimator（推定器）への
    パスを '__' 結合で返す。
    Pipelineでなければ None を返す。

    ネストされたPipelineにも対応:
        Pipeline([("preprocess", ...), ("model", Lasso())])
        → "model"

        Pipeline([("smiles", ...), ("main_pipe", Pipeline([("prep", ...), ("model", Ridge())]))])
        → "main_pipe__model"
    """
    try:
        from sklearn.pipeline import Pipeline
        if not isinstance(model, Pipeline) or not hasattr(model, "steps") or len(model.steps) == 0:
            return None

        last_name, last_step = model.steps[-1]

        # 最終ステップがさらにPipelineの場合、再帰的に探索
        if isinstance(last_step, Pipeline) and hasattr(last_step, "steps") and len(last_step.steps) > 0:
            inner = _detect_pipeline_step_name(last_step)
            if inner is not None:
                return f"{last_name}__{inner}"

        return last_name
    except ImportError:
        pass
    return None


def _prefix_param_grid(
    param_grid: dict[str, Any],
    step_name: str,
) -> dict[str, Any]:
    """
    param_gridの全キーに '{step_name}__' プレフィックスを付与する。

    既にプレフィックス付きのキーはスキップする。

    例:
        {"alpha": [0.1, 1.0]} → {"estimator__alpha": [0.1, 1.0]}
    """
    prefixed: dict[str, Any] = {}
    prefix = f"{step_name}__"
    for key, value in param_grid.items():
        if key.startswith(prefix) or "__" in key:
            # 既にプレフィックス付き、またはネストされたパラメータ
            prefixed[key] = value
        else:
            prefixed[f"{prefix}{key}"] = value
    return prefixed


def _strip_prefix_from_params(
    params: dict[str, Any],
    step_name: str,
) -> dict[str, Any]:
    """
    best_paramsから '{step_name}__' プレフィックスを除去して返す。

    ユーザーには 'alpha' のようなクリーンなパラメータ名を見せる。
    """
    prefix = f"{step_name}__"
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if key.startswith(prefix):
            cleaned[key[len(prefix):]] = value
        else:
            cleaned[key] = value
    return cleaned


def tune(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    config: TunerConfig,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    指定された手法でハイパーパラメータを最適化して結果を返す。

    modelがsklearn Pipelineの場合、param_gridのキーに
    自動で最終ステップ名（例: 'estimator__'）プレフィックスを付与する。
    結果のbest_paramsからはプレフィックスを除去して返す。

    Args:
        model: sklearn互換の推定器 or Pipeline
        X: 特徴量行列
        y: 目的変数
        config: TunerConfig インスタンス
        groups: グループラベル（GroupKFold等で必要）

    Returns:
        {
            "best_estimator": 最良モデル,
            "best_params": 最良パラメータ（プレフィックスなし）,
            "best_score": 最良スコア,
            "cv_results": 全試行の結果（DataFrameに変換可能）
        }
    """
    # Pipeline検出 & param_gridにプレフィックスを自動付与
    step_name = _detect_pipeline_step_name(model)
    if step_name is not None:
        prefixed_grid = _prefix_param_grid(config.param_grid, step_name)
        logger.info(
            f"Pipeline検出: ステップ名='{step_name}', "
            f"param_grid キーを '{step_name}__' プレフィックス付きに変換"
        )
        config = TunerConfig(
            method=config.method,
            param_grid=prefixed_grid,
            n_iter=config.n_iter,
            cv=config.cv,
            scoring=config.scoring,
            n_jobs=config.n_jobs,
            verbose=config.verbose,
            refit=config.refit,
            optuna_direction=config.optuna_direction,
            optuna_timeout=config.optuna_timeout,
            random_state=config.random_state,
        )

    method = config.method.lower()

    if method == "grid":
        result = _run_grid(model, X, y, config, groups)
    elif method == "random":
        result = _run_random(model, X, y, config, groups)
    elif method == "halving_grid":
        result = _run_halving_grid(model, X, y, config, groups)
    elif method == "halving_random":
        result = _run_halving_random(model, X, y, config, groups)
    elif method == "optuna":
        result = _run_optuna(model, X, y, config, groups, step_name=step_name)
    elif method == "bayes":
        result = _run_bayes(model, X, y, config, groups)
    else:
        raise ValueError(
            f"未知のチューニング手法 '{method}'。"
            f"利用可能: grid, random, halving_grid, halving_random, optuna, bayes"
        )

    # best_paramsからプレフィックスを除去（ユーザー向け）
    if step_name is not None and "best_params" in result:
        result["best_params"] = _strip_prefix_from_params(
            result["best_params"], step_name
        )

    return result


def _run_grid(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """GridSearchCV による全探索。"""
    gs = GridSearchCV(
        model,
        cfg.param_grid,
        scoring=cfg.scoring,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        verbose=cfg.verbose,
    )
    gs.fit(X, y, groups=groups)
    return _extract_results(gs)


def _run_random(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """RandomizedSearchCV によるランダム探索。"""
    # Optuna形式のdict値がある場合はscipy分布に変換
    param_grid = cfg.param_grid
    if any(isinstance(v, dict) and "type" in v for v in param_grid.values()):
        param_grid = _convert_optuna_grid_for_random(param_grid)
    rs = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=cfg.n_iter,
        scoring=cfg.scoring,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        verbose=cfg.verbose,
        random_state=cfg.random_state,
    )
    rs.fit(X, y, groups=groups)
    return _extract_results(rs)


def _run_halving_grid(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """HalvingGridSearchCV による段階的全探索。"""
    if not _halving_available:
        logger.warning("HalvingGridSearchCV 未対応 → GridSearchCVで代替")
        return _run_grid(model, X, y, cfg, groups)
    hs = HalvingGridSearchCV(
        model,
        cfg.param_grid,
        scoring=cfg.scoring,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        verbose=cfg.verbose,
        random_state=cfg.random_state,
    )
    hs.fit(X, y, groups=groups)
    return _extract_results(hs)


def _run_halving_random(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """HalvingRandomSearchCV による段階的ランダム探索。"""
    if not _halving_available:
        logger.warning("HalvingRandomSearchCV 未対応 → RandomizedSearchCVで代替")
        return _run_random(model, X, y, cfg, groups)
    # Optuna形式のdict値がある場合はscipy分布に変換
    param_grid = cfg.param_grid
    if any(isinstance(v, dict) and "type" in v for v in param_grid.values()):
        param_grid = _convert_optuna_grid_for_random(param_grid)
    hs = HalvingRandomSearchCV(
        model,
        param_grid,
        n_candidates=cfg.n_iter,
        scoring=cfg.scoring,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        verbose=cfg.verbose,
        random_state=cfg.random_state,
    )
    hs.fit(X, y, groups=groups)
    return _extract_results(hs)


def _run_optuna(
    model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any,
    *, step_name: str | None = None,
) -> dict[str, Any]:
    """Optuna によるベイズ最適化。param_grid は optuna の suggest形式で指定。

    Pipeline対応: step_nameが指定された場合、
    Pipelineの最終ステップのestimatorに直接パラメータをセットする。
    """
    if not _optuna:
        logger.warning("optuna 未インストール → RandomizedSearchCVで代替")
        # Optuna用dict形式のparam_gridをscipy.stats分布に変換
        converted_grid = _convert_optuna_grid_for_random(cfg.param_grid)
        fallback_cfg = TunerConfig(
            method="random",
            param_grid=converted_grid,
            n_iter=cfg.n_iter,
            cv=cfg.cv,
            scoring=cfg.scoring,
            n_jobs=cfg.n_jobs,
            refit=cfg.refit,
            verbose=cfg.verbose,
            random_state=cfg.random_state,
        )
        return _run_random(model, X, y, fallback_cfg, groups)

    import optuna  # type: ignore
    from sklearn.model_selection import cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # param_gridからプレフィックスを除去してOptuna trialに渡す
    # （プレフィックス付きのキーはOptunaのsuggest名として不適切）
    raw_param_grid = cfg.param_grid
    if step_name is not None:
        raw_param_grid = _strip_prefix_from_params(cfg.param_grid, step_name)

    def objective(trial: Any) -> float:
        params: dict[str, Any] = {}
        for param_name, spec in raw_param_grid.items():
            if isinstance(spec, dict):
                suggest_type = spec.get("type", "float")
                if suggest_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, spec["low"], spec["high"],
                        step=spec.get("step", 1)
                    )
                elif suggest_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, spec["low"], spec["high"],
                        log=spec.get("log", False)
                    )
                elif suggest_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, spec["choices"]
                    )
            elif isinstance(spec, list):
                params[param_name] = trial.suggest_categorical(param_name, spec)

        # Pipeline対応: set_params で estimator__xxx 形式でセット
        if step_name is not None:
            prefixed = {f"{step_name}__{k}": v for k, v in params.items()}
            m = model.set_params(**prefixed)
            try:
                from sklearn.base import clone
                m = clone(model).set_params(**prefixed)
            except Exception:
                m = model.__class__(**{**model.get_params(), **prefixed})
        else:
            m = model.__class__(**{**model.get_params(), **params})

        scores = cross_val_score(
            m, X, y,
            cv=cfg.cv,
            scoring=cfg.scoring,
            n_jobs=cfg.n_jobs,
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction=cfg.optuna_direction,
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
    )
    study.optimize(objective, n_trials=cfg.n_iter, timeout=cfg.optuna_timeout)

    best_params = study.best_params

    # refit: ベストパラメータで再学習
    if cfg.refit:
        if step_name is not None:
            from sklearn.base import clone
            prefixed = {f"{step_name}__{k}": v for k, v in best_params.items()}
            best_model = clone(model).set_params(**prefixed)
        else:
            best_model = model.__class__(**{**model.get_params(), **best_params})
        best_model.fit(X, y)
    else:
        best_model = model

    return {
        "best_estimator": best_model,
        "best_params": best_params,  # プレフィックスなし（tune()側で統一）
        "best_score": study.best_value,
        "cv_results": study.trials_dataframe(),
    }


def _run_bayes(model: Any, X: Any, y: Any, cfg: TunerConfig, groups: Any) -> dict[str, Any]:
    """BayesSearchCV (scikit-optimize) によるベイズ最適化。"""
    if not _skopt:
        logger.warning("scikit-optimize 未インストール → RandomizedSearchCVで代替")
        return _run_random(model, X, y, cfg, groups)

    from skopt import BayesSearchCV  # type: ignore

    bs = BayesSearchCV(
        model,
        cfg.param_grid,
        n_iter=cfg.n_iter,
        scoring=cfg.scoring,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        refit=cfg.refit,
        verbose=cfg.verbose,
        random_state=cfg.random_state,
    )
    bs.fit(X, y, groups=groups)
    return _extract_results(bs)


def _extract_results(search_obj: Any) -> dict[str, Any]:
    """sklearn Search系オブジェクトから結果を抽出する。"""
    import pandas as pd
    return {
        "best_estimator": search_obj.best_estimator_,
        "best_params": search_obj.best_params_,
        "best_score": float(search_obj.best_score_),
        "cv_results": pd.DataFrame(search_obj.cv_results_),
    }
