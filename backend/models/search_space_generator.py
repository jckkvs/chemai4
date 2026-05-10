"""
backend/models/search_space_generator.py

ParamSpecリストからGridSearchCV / OptunaSearchCV用の探索空間を自動生成する。

設計思想:
    - ParamSpec（型・デフォルト値・範囲情報）から
      GridSearch用の有限離散リスト、Optuna用の連続/離散/カテゴリカル
      探索空間を自動的に推論する。
    - ユーザーが登録した未知のestimatorにも対応できるよう、
      型とデフォルト値だけから合理的な探索空間を生成する。

Implements:
    F-SSG01: generate_grid_space     — ParamSpec → GridSearchCV param_grid
    F-SSG02: generate_optuna_space   — ParamSpec → OptunaSearchCV param_grid
    F-SSG03: generate_search_spaces  — 一括生成（Grid + Optuna）
    F-SSG04: SearchParamSpec          — 個別パラメータの探索空間定義
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from backend.ui.param_schema import ParamSpec

logger = logging.getLogger(__name__)


# ============================================================
# 探索空間パラメータ定義
# ============================================================

@dataclass
class SearchParamSpec:
    """1つのパラメータの探索空間定義。

    Attributes:
        name:           パラメータ名
        param_type:     "int" | "float" | "categorical" | "bool"
        enabled:        探索対象にするか（UIで有効/無効切替）
        description:    パラメータの説明文

        # --- Grid Search 用 ---
        grid_values:    GridSearchCV用の有限値リスト

        # --- Optuna Search 用 ---
        optuna_type:    Optuna suggest型 ("int" | "float" | "categorical")
        optuna_low:     下限
        optuna_high:    上限
        optuna_step:    ステップ幅（int/float）
        optuna_log:     対数スケールか
        optuna_choices: カテゴリカルの選択肢

        default_value:  デフォルト値（参考情報）
    """
    name: str
    param_type: str  # "int", "float", "categorical", "bool"
    enabled: bool = True
    description: str = ""

    # GridSearchCV
    grid_values: list[Any] = field(default_factory=list)

    # OptunaSearchCV
    optuna_type: str = "float"  # "int", "float", "categorical"
    optuna_low: float | None = None
    optuna_high: float | None = None
    optuna_step: float | None = None
    optuna_log: bool = False
    optuna_choices: list[Any] = field(default_factory=list)

    default_value: Any = None

    def to_grid_entry(self) -> list[Any] | None:
        """GridSearchCV param_grid 用のエントリを返す。"""
        if not self.enabled or not self.grid_values:
            return None
        return self.grid_values

    def to_optuna_entry(self) -> dict[str, Any] | None:
        """Optuna param_grid 用のエントリを返す。"""
        if not self.enabled:
            return None
        if self.optuna_type == "categorical":
            return {"type": "categorical", "choices": self.optuna_choices}
        elif self.optuna_type == "int":
            return {
                "type": "int",
                "low": int(self.optuna_low or 1),
                "high": int(self.optuna_high or 100),
                "step": int(self.optuna_step or 1),
            }
        elif self.optuna_type == "float":
            entry: dict[str, Any] = {
                "type": "float",
                "low": float(self.optuna_low or 1e-6),
                "high": float(self.optuna_high or 10.0),
                "log": self.optuna_log,
            }
            if self.optuna_step is not None and not self.optuna_log:
                entry["step"] = float(self.optuna_step)
            return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON直列化可能な辞書に変換。"""
        return {
            "name": self.name,
            "param_type": self.param_type,
            "enabled": self.enabled,
            "description": self.description,
            "grid_values": self.grid_values,
            "optuna_type": self.optuna_type,
            "optuna_low": self.optuna_low,
            "optuna_high": self.optuna_high,
            "optuna_step": self.optuna_step,
            "optuna_log": self.optuna_log,
            "optuna_choices": self.optuna_choices,
            "default_value": self.default_value,
        }


# ============================================================
# 探索空間名前の推奨設定（既知パラメータ用）
# ============================================================

_KNOWN_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "n_estimators": {
        "grid": [50, 100, 200, 500],
        "optuna": {"type": "int", "low": 10, "high": 1000, "step": 10, "log": False},
    },
    "max_depth": {
        "grid": [3, 5, 7, 10, 15, 20, None],
        "optuna": {"type": "int", "low": 1, "high": 50, "step": 1},
    },
    "min_samples_split": {
        "grid": [2, 5, 10, 20],
        "optuna": {"type": "int", "low": 2, "high": 50, "step": 1},
    },
    "min_samples_leaf": {
        "grid": [1, 2, 5, 10],
        "optuna": {"type": "int", "low": 1, "high": 50, "step": 1},
    },
    "learning_rate": {
        "grid": [0.001, 0.01, 0.05, 0.1, 0.3],
        "optuna": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
    },
    "alpha": {
        "grid": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "optuna": {"type": "float", "low": 1e-6, "high": 100.0, "log": True},
    },
    "l1_ratio": {
        "grid": [0.1, 0.3, 0.5, 0.7, 0.9],
        "optuna": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
    },
    "C": {
        "grid": [0.01, 0.1, 1.0, 10.0, 100.0],
        "optuna": {"type": "float", "low": 1e-3, "high": 1000.0, "log": True},
    },
    "epsilon": {
        "grid": [0.01, 0.05, 0.1, 0.3, 0.5],
        "optuna": {"type": "float", "low": 0.001, "high": 1.0, "log": True},
    },
    "subsample": {
        "grid": [0.5, 0.7, 0.8, 0.9, 1.0],
        "optuna": {"type": "float", "low": 0.3, "high": 1.0, "log": False},
    },
    "colsample_bytree": {
        "grid": [0.5, 0.7, 0.8, 0.9, 1.0],
        "optuna": {"type": "float", "low": 0.3, "high": 1.0, "log": False},
    },
    "reg_alpha": {
        "grid": [0.0, 0.01, 0.1, 1.0, 10.0],
        "optuna": {"type": "float", "low": 1e-6, "high": 100.0, "log": True},
    },
    "reg_lambda": {
        "grid": [0.0, 0.01, 0.1, 1.0, 10.0],
        "optuna": {"type": "float", "low": 1e-6, "high": 100.0, "log": True},
    },
    "n_neighbors": {
        "grid": [3, 5, 7, 10, 15, 20],
        "optuna": {"type": "int", "low": 1, "high": 50, "step": 1},
    },
    "n_components": {
        "grid": [2, 3, 5, 7, 10],
        "optuna": {"type": "int", "low": 1, "high": 50, "step": 1},
    },
    "max_features": {
        "grid": ["sqrt", "log2", None],
        "optuna": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    },
    "num_leaves": {
        "grid": [15, 31, 63, 127, 255],
        "optuna": {"type": "int", "low": 2, "high": 512, "step": 1},
    },
    "max_bins": {
        "grid": [32, 64, 128, 255],
        "optuna": {"type": "int", "low": 16, "high": 512, "step": 1},
    },
    "gamma": {
        "grid": ["scale", "auto", 0.01, 0.1, 1.0],
        "optuna": {"type": "categorical", "choices": ["scale", "auto", 0.001, 0.01, 0.1, 1.0]},
    },
    "criterion": {
        "grid": ["squared_error", "absolute_error", "friedman_mse"],
        "optuna": {"type": "categorical", "choices": ["squared_error", "absolute_error", "friedman_mse"]},
    },
    "kernel": {
        "grid": ["linear", "rbf", "poly"],
        "optuna": {"type": "categorical", "choices": ["linear", "rbf", "poly"]},
    },
    "solver": {
        "grid": ["auto", "svd", "lsqr", "saga"],
        "optuna": {"type": "categorical", "choices": ["auto", "svd", "lsqr", "saga"]},
    },
}


# ============================================================
# メインAPI
# ============================================================

def generate_search_space(
    param_spec: ParamSpec,
    *,
    include_advanced: bool = False,
) -> SearchParamSpec | None:
    """ParamSpecから1つのSearchParamSpecを生成する。

    Args:
        param_spec: ParamSpec（param_schema.pyで生成）
        include_advanced: advanced グループも含めるか

    Returns:
        SearchParamSpec or None（探索対象外の場合）
    """
    # スキップ対象
    _SKIP_SEARCH = {"random_state", "n_jobs", "verbose", "warm_start",
                    "copy_X", "copy", "oob_score", "class_weight"}
    if param_spec.name in _SKIP_SEARCH:
        return None
    if param_spec.group == "advanced" and not include_advanced:
        return None

    # 既知パラメータの場合はプリセットを使用
    if param_spec.name in _KNOWN_SEARCH_SPACES:
        known = _KNOWN_SEARCH_SPACES[param_spec.name]
        optuna_def = known.get("optuna", {})
        return SearchParamSpec(
            name=param_spec.name,
            param_type=_map_param_type(param_spec),
            enabled=True,
            description=param_spec.description,
            grid_values=known.get("grid", []),
            optuna_type=optuna_def.get("type", "float"),
            optuna_low=optuna_def.get("low"),
            optuna_high=optuna_def.get("high"),
            optuna_step=optuna_def.get("step"),
            optuna_log=optuna_def.get("log", False),
            optuna_choices=optuna_def.get("choices", []),
            default_value=param_spec.default,
        )

    # 未知パラメータ: デフォルト値と型から自動推論
    return _auto_generate_search_spec(param_spec)


def generate_grid_space(
    param_specs: list[ParamSpec],
    *,
    include_advanced: bool = False,
) -> dict[str, list[Any]]:
    """ParamSpecリストからGridSearchCV用のparam_gridを生成する。

    Implements: F-SSG01

    Args:
        param_specs: ParamSpecリスト
        include_advanced: advancedパラメータも含めるか

    Returns:
        {param_name: [value_list]} — GridSearchCV.param_grid 形式
    """
    grid: dict[str, list[Any]] = {}
    for spec in param_specs:
        ss = generate_search_space(spec, include_advanced=include_advanced)
        if ss is not None and ss.enabled:
            entry = ss.to_grid_entry()
            if entry:
                grid[ss.name] = entry
    return grid


def generate_optuna_space(
    param_specs: list[ParamSpec],
    *,
    include_advanced: bool = False,
) -> dict[str, dict[str, Any]]:
    """ParamSpecリストからOptuna用のparam_gridを生成する。

    Implements: F-SSG02

    Args:
        param_specs: ParamSpecリスト
        include_advanced: advancedパラメータも含めるか

    Returns:
        {param_name: {"type": ..., "low": ..., "high": ..., ...}}
        — tuner.py の Optuna 形式
    """
    space: dict[str, dict[str, Any]] = {}
    for spec in param_specs:
        ss = generate_search_space(spec, include_advanced=include_advanced)
        if ss is not None and ss.enabled:
            entry = ss.to_optuna_entry()
            if entry:
                space[ss.name] = entry
    return space


def generate_search_spaces(
    param_specs: list[ParamSpec],
    *,
    include_advanced: bool = False,
) -> dict[str, SearchParamSpec]:
    """ParamSpecリストから全SearchParamSpecを生成する。

    Implements: F-SSG03

    Args:
        param_specs: ParamSpecリスト
        include_advanced: advancedパラメータも含めるか

    Returns:
        {param_name: SearchParamSpec} の辞書
    """
    spaces: dict[str, SearchParamSpec] = {}
    for spec in param_specs:
        ss = generate_search_space(spec, include_advanced=include_advanced)
        if ss is not None:
            spaces[ss.name] = ss
    return spaces


def generate_search_spaces_from_estimator(
    estimator_cls: type,
    *,
    include_advanced: bool = False,
) -> dict[str, SearchParamSpec]:
    """estimatorクラスからSearchParamSpecを一括生成する便利関数。

    Args:
        estimator_cls: sklearn互換estimatorクラス
        include_advanced: advancedパラメータも含めるか

    Returns:
        {param_name: SearchParamSpec}
    """
    from backend.ui.param_schema import introspect_params
    param_specs = introspect_params(estimator_cls)
    return generate_search_spaces(param_specs, include_advanced=include_advanced)


# ============================================================
# 型ユーティリティ
# ============================================================

def _map_param_type(spec: ParamSpec) -> str:
    """ParamSpecのparam_typeをSearchParamSpecの型にマップ。"""
    if spec.param_type == "bool":
        return "bool"
    if spec.param_type == "int":
        return "int"
    if spec.param_type == "float":
        return "float"
    if spec.param_type in ("select", "multiselect", "str", "text"):
        return "categorical"
    return "categorical"


def _auto_generate_search_spec(spec: ParamSpec) -> SearchParamSpec | None:
    """未知のパラメータについてデフォルト値と型から探索空間を自動生成する。"""
    default = spec.default

    # bool型
    if spec.param_type == "bool":
        return SearchParamSpec(
            name=spec.name,
            param_type="bool",
            enabled=True,
            description=spec.description,
            grid_values=[True, False],
            optuna_type="categorical",
            optuna_choices=[True, False],
            default_value=default,
        )

    # int型
    if spec.param_type == "int" and isinstance(default, (int, float)):
        dv = int(default)
        if dv <= 0:
            dv = 1
        # デフォルト値を中心に探索空間を生成
        grid, low, high, log = _infer_int_range(dv, spec.name)
        return SearchParamSpec(
            name=spec.name,
            param_type="int",
            enabled=True,
            description=spec.description,
            grid_values=grid,
            optuna_type="int",
            optuna_low=low,
            optuna_high=high,
            optuna_step=1,
            optuna_log=log,
            default_value=default,
        )

    # float型
    if spec.param_type == "float" and isinstance(default, (int, float)):
        dv = float(default)
        grid, low, high, log = _infer_float_range(dv, spec.name)
        return SearchParamSpec(
            name=spec.name,
            param_type="float",
            enabled=True,
            description=spec.description,
            grid_values=grid,
            optuna_type="float",
            optuna_low=low,
            optuna_high=high,
            optuna_log=log,
            default_value=default,
        )

    # select型（選択肢あり）
    if spec.param_type == "select" and spec.choices:
        return SearchParamSpec(
            name=spec.name,
            param_type="categorical",
            enabled=True,
            description=spec.description,
            grid_values=spec.choices,
            optuna_type="categorical",
            optuna_choices=spec.choices,
            default_value=default,
        )

    # str型（選択肢なし）— 探索対象外
    if spec.param_type in ("str", "text"):
        return None

    # multiselect — 探索対象外（複合型）
    if spec.param_type == "multiselect":
        return None

    # union / フォールバック
    return None


def _infer_int_range(
    default: int,
    name: str,
) -> tuple[list[int], int, int, bool]:
    """int型パラメータのGridリスト、Optuna範囲を推論する。

    戦略:
    - デフォルト値を中心に、1/4, 1/2, 1x, 2x, 4x の5点
    - 下限は max(1, default // 4)
    - 上限は default * 4（上限10000）
    """
    if default <= 0:
        default = 1

    low = max(1, default // 4)
    high = min(default * 4, 10000)

    if default <= 5:
        grid = sorted(set(range(max(1, default - 2), min(default + 3, high + 1))))
    else:
        candidates = {
            max(1, default // 4),
            max(1, default // 2),
            default,
            default * 2,
            min(default * 4, 10000),
        }
        grid = sorted(candidates)

    # 対数スケール判定: range > 100 かつ default > 10
    use_log = (high / max(low, 1) > 100) and default >= 10

    return grid, low, high, use_log


def _infer_float_range(
    default: float,
    name: str,
) -> tuple[list[float], float, float, bool]:
    """float型パラメータのGridリスト、Optuna範囲を推論する。

    戦略:
    - 0 < default <= 1 の場合: [0.01, 0.1, default, 0.5, 1.0]
    - default > 1 の場合: [default/10, default/2, default, default*2, default*10]
    - default == 0 の場合: [0.0, 0.001, 0.01, 0.1, 1.0]
    """
    if default == 0 or default is None:
        grid = [0.0, 0.001, 0.01, 0.1, 1.0]
        return grid, 0.0, 1.0, False

    if 0 < abs(default) <= 1:
        low = max(1e-6, abs(default) / 100)
        high = min(1.0, abs(default) * 10)
        grid = sorted(set([
            round(low, 6),
            round(abs(default) / 2, 6),
            round(abs(default), 6),
            round(min(abs(default) * 2, high), 6),
            round(high, 6),
        ]))
        use_log = low > 0 and (high / low > 100)
        return grid, low, high, use_log

    # default > 1
    low = max(1e-3, abs(default) / 10)
    high = abs(default) * 10
    grid = sorted(set([
        round(low, 4),
        round(abs(default) / 2, 4),
        round(abs(default), 4),
        round(abs(default) * 2, 4),
        round(high, 4),
    ]))
    use_log = low > 0 and (high / low > 100)
    return grid, low, high, use_log
