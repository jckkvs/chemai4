"""
backend/data/feature_engineer.py

特徴量エンジニアリングモジュール。
多項式・交互作用・集約・時系列特徴量を生成するsklearn互換Transformer群。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


# ============================================================
# 多項式・交互作用特徴量
# ============================================================

class InteractionTransformer(BaseEstimator, TransformerMixin):
    """
    指定列の全ペアの積 (交互作用項) を追加するTransformer。

    Implements: 要件定義書 §3.4 特徴量エンジニアリング

    Args:
        degree: 最大次数 (2 = 二次の交互作用のみ)
        include_bias: バイアス項を含めるか
        interaction_only: 積のみ（自乗項なし）
    """

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = True,
        include_bias: bool = False,
    ) -> None:
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self._poly: PolynomialFeatures | None = None
        self._feature_names_in: list[str] = []

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "InteractionTransformer":
        self._poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
        else:
            self._feature_names_in = [f"x{i}" for i in range(X.shape[1])]
        self._poly.fit(X)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> np.ndarray:
        assert self._poly is not None, "fit() を先に呼んでください。"
        return self._poly.transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if self._poly is None:
            return np.array([])
        names = input_features or self._feature_names_in
        return self._poly.get_feature_names_out(names)


# ============================================================
# 集約特徴量
# ============================================================

class GroupAggTransformer(BaseEstimator, TransformerMixin):
    """
    カテゴリ列でグループ化した集約特徴量（mean/std/min/max/count）を追加する。

    Args:
        group_col: グループ化するカテゴリ列名
        agg_cols: 集約する数値列名のリスト
        agg_funcs: 集約関数のリスト
    """

    def __init__(
        self,
        group_col: str,
        agg_cols: list[str],
        agg_funcs: list[str] | None = None,
    ) -> None:
        self.group_col = group_col
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs or ["mean", "std", "min", "max"]
        self._agg_df: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "GroupAggTransformer":
        if self.group_col not in X.columns:
            raise ValueError(f"グループ列 '{self.group_col}' がデータに存在しません。")
        valid_cols = [c for c in self.agg_cols if c in X.columns]
        self._agg_df = X.groupby(self.group_col)[valid_cols].agg(self.agg_funcs)
        self._agg_df.columns = [f"grp_{self.group_col}_{c}_{fn}"
                                  for c, fn in self._agg_df.columns]
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        assert self._agg_df is not None, "fit() を先に呼んでください。"
        merged = X.join(
            self._agg_df, on=self.group_col, how="left", rsuffix="_agg"
        )
        new_cols = self._agg_df.columns.tolist()
        return merged[new_cols].fillna(0).to_numpy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if self._agg_df is None:
            return np.array([])
        return np.array(self._agg_df.columns.tolist())


# ============================================================
# 時系列特徴量
# ============================================================

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Datetime列から時系列特徴量（年/月/日/時/曜日/週番号等）を抽出するTransformer。

    Args:
        components: 抽出するコンポーネントのリスト
        add_cyclic: 月・時・曜日のsin/cos変換を追加するか
    """

    _COMPONENTS = ["year", "month", "day", "hour", "minute",
                   "dayofweek", "dayofyear", "weekofyear", "quarter",
                   "is_weekend", "is_month_start", "is_month_end"]

    def __init__(
        self,
        components: list[str] | None = None,
        add_cyclic: bool = True,
    ) -> None:
        self.components = components or self._COMPONENTS
        self.add_cyclic = add_cyclic
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "DatetimeFeatureExtractor":
        # feature_names は transform 後に確定するため一度変換して取得
        sample = X.iloc[[0]] if isinstance(X, pd.DataFrame) else X[:1]
        out = self._extract(sample)
        self._feature_names = out.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> np.ndarray:
        df = self._extract(X)
        return df.to_numpy(dtype=float)

    def _extract(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            series = pd.to_datetime(X.ravel())
        elif isinstance(X, pd.DataFrame):
            series = pd.to_datetime(X.iloc[:, 0])
        else:
            series = pd.to_datetime(X)

        rows: dict[str, pd.Series] = {}
        comp_map = {
            "year": lambda s: s.dt.year,
            "month": lambda s: s.dt.month,
            "day": lambda s: s.dt.day,
            "hour": lambda s: s.dt.hour,
            "minute": lambda s: s.dt.minute,
            "dayofweek": lambda s: s.dt.dayofweek,
            "dayofyear": lambda s: s.dt.dayofyear,
            "weekofyear": lambda s: s.dt.isocalendar().week.astype(int),
            "quarter": lambda s: s.dt.quarter,
            "is_weekend": lambda s: (s.dt.dayofweek >= 5).astype(int),
            "is_month_start": lambda s: s.dt.is_month_start.astype(int),
            "is_month_end": lambda s: s.dt.is_month_end.astype(int),
        }
        for comp in self.components:
            if comp in comp_map:
                rows[comp] = comp_map[comp](series).values

        if self.add_cyclic:
            if "month" in rows:
                rows["month_sin"] = np.sin(2 * np.pi * rows["month"] / 12)
                rows["month_cos"] = np.cos(2 * np.pi * rows["month"] / 12)
            if "hour" in rows:
                rows["hour_sin"] = np.sin(2 * np.pi * rows["hour"] / 24)
                rows["hour_cos"] = np.cos(2 * np.pi * rows["hour"] / 24)
            if "dayofweek" in rows:
                rows["dow_sin"] = np.sin(2 * np.pi * rows["dayofweek"] / 7)
                rows["dow_cos"] = np.cos(2 * np.pi * rows["dayofweek"] / 7)

        return pd.DataFrame(rows, index=range(len(series)))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self._feature_names)


# ============================================================
# ラグ・ローリング特徴量
# ============================================================

class LagRollingTransformer(BaseEstimator, TransformerMixin):
    """
    時系列データのラグ特徴量・ローリング統計量を生成するTransformer。

    Args:
        lags: ラグ数のリスト（例: [1, 2, 3]）
        windows: ローリングウィンドウサイズのリスト（例: [3, 7]）
        agg_funcs: ローリング集約関数（例: ["mean", "std"]）
    """

    def __init__(
        self,
        lags: list[int] | None = None,
        windows: list[int] | None = None,
        agg_funcs: list[str] | None = None,
    ) -> None:
        self.lags = lags or [1, 2, 3]
        self.windows = windows or [3, 7]
        self.agg_funcs = agg_funcs or ["mean", "std"]
        self._feature_names: list[str] = []
        self._n_input_cols: int = 0

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "LagRollingTransformer":
        self._n_input_cols = X.shape[1]
        if isinstance(X, pd.DataFrame):
            cols = X.columns.tolist()
        else:
            cols = [f"x{i}" for i in range(X.shape[1])]

        names: list[str] = []
        for col in cols:
            for lag in self.lags:
                names.append(f"{col}_lag{lag}")
            for w in self.windows:
                for fn in self.agg_funcs:
                    names.append(f"{col}_roll{w}_{fn}")
        self._feature_names = names
        return self

    def transform(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

        result_parts: list[np.ndarray] = []
        for col in X.columns:
            series = X[col]
            # ラグ特徴量
            for lag in self.lags:
                result_parts.append(series.shift(lag).fillna(0).values.reshape(-1, 1))
            # ローリング特徴量
            for w in self.windows:
                for fn in self.agg_funcs:
                    result_parts.append(
                        getattr(series.rolling(w, min_periods=1), fn)()
                        .fillna(0).values.reshape(-1, 1)
                    )
        return np.hstack(result_parts)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self._feature_names)


# ============================================================
# 設定クラス
# ============================================================

@dataclass
class FeatureEngineeringConfig:
    """特徴量エンジニアリングの設定。"""
    # 多項式・交互作用
    add_interactions: bool = False
    interaction_degree: int = 2
    interaction_only: bool = True

    # グループ集約
    group_agg_configs: list[dict[str, Any]] = field(default_factory=list)
    # [{group_col: str, agg_cols: list[str], agg_funcs: list[str]}]

    # 時系列特徴量
    datetime_cols: list[str] = field(default_factory=list)
    add_cyclic: bool = True

    # ラグ・ローリング
    lag_rolling_cols: list[str] = field(default_factory=list)
    lags: list[int] = field(default_factory=lambda: [1, 2, 3])
    rolling_windows: list[int] = field(default_factory=lambda: [3, 7])


def build_feature_engineering_pipeline(
    config: FeatureEngineeringConfig,
) -> list[tuple[str, Any]]:
    """
    設定に基づいて特徴量エンジニアリングステップのリストを返す。
    sklearn Pipeline の steps として利用可能。

    Args:
        config: FeatureEngineeringConfig

    Returns:
        [(name, transformer)] のリスト
    """
    steps: list[tuple[str, Any]] = []

    # 1. 交互作用特徴量
    if config.add_interactions:
        steps.append(("interactions", InteractionTransformer(
            degree=config.interaction_degree,
            interaction_only=config.interaction_only,
        )))

    # 2. グループ集約特徴量
    for i, agg_cfg in enumerate(config.group_agg_configs):
        group_col = agg_cfg.get("group_col", "")
        agg_cols = agg_cfg.get("agg_cols", [])
        agg_funcs = agg_cfg.get("agg_funcs", None)
        if group_col and agg_cols:
            steps.append((
                f"group_agg_{i}",
                GroupAggTransformer(
                    group_col=group_col,
                    agg_cols=agg_cols,
                    agg_funcs=agg_funcs,
                ),
            ))

    # 3. 時系列特徴量（Datetime列があれば）
    if config.datetime_cols:
        steps.append(("datetime_features", DatetimeFeatureExtractor(
            add_cyclic=config.add_cyclic,
        )))

    # 4. ラグ・ローリング特徴量
    if config.lag_rolling_cols:
        steps.append(("lag_rolling", LagRollingTransformer(
            lags=config.lags,
            windows=config.rolling_windows,
        )))

    return steps

