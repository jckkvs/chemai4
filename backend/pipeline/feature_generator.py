"""
backend/pipeline/feature_generator.py

特徴量生成モジュール。
PolynomialFeatures 等をラップした多段変換可能な Transformer。

対応メソッド:
  none            : スキップ（パススルー）
  polynomial      : PolynomialFeatures（自乗項含む多項式）
  interaction_only: PolynomialFeatures（交互作用項のみ）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


# ============================================================
# 設定クラス
# ============================================================

@dataclass
class FeatureGenConfig:
    """
    特徴量生成の設定。

    Attributes:
        method: 生成メソッド。
            "none"             : スキップ（パススルー）
            "polynomial"       : PolynomialFeatures（自乗項含む）
            "interaction_only" : 積のみ（自乗項なし）
        degree: 多項式の次数（degree=2 が一般的）
        include_bias: バイアス項（定数列）を含めるか
    """
    method: str = "none"       # "none" | "polynomial" | "interaction_only"
    degree: int = 2
    include_bias: bool = False


# ============================================================
# メインクラス
# ============================================================

class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    特徴量生成 Transformer。

    FeatureGenConfig に従って特徴量を生成する。
    method="none" の場合はパススルー（何も変換しない）。

    多段変換したい場合は sklearn Pipeline で複数の FeatureGenerator を
    直列に繋ぐことで実現できる。

    Args:
        config: FeatureGenConfig（省略時は method="none"）
    """

    def __init__(self, config: FeatureGenConfig | None = None) -> None:
        self.config = config or FeatureGenConfig()
        self._transformer: PolynomialFeatures | None = None
        self._feature_names_in: list[str] = []
        self._n_features_in: int = 0

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Any = None,
    ) -> "FeatureGenerator":
        """
        変換器を fit する。

        Args:
            X: 入力特徴量行列
            y: 未使用

        Returns:
            self
        """
        cfg = self.config

        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
            self._n_features_in = X.shape[1]
        else:
            self._n_features_in = X.shape[1]
            self._feature_names_in = [f"x{i}" for i in range(self._n_features_in)]

        if cfg.method == "none":
            logger.debug("FeatureGenerator: method=none → パススルー")
            return self

        interaction_only = (cfg.method == "interaction_only")
        self._transformer = PolynomialFeatures(
            degree=cfg.degree,
            interaction_only=interaction_only,
            include_bias=cfg.include_bias,
        )
        self._transformer.fit(
            X.values if isinstance(X, pd.DataFrame) else X
        )

        n_out = self._transformer.n_output_features_
        logger.info(
            f"FeatureGenerator.fit(): method={cfg.method}, degree={cfg.degree}, "
            f"{self._n_features_in} → {n_out} 特徴量"
        )
        return self

    def transform(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Any = None,
    ) -> np.ndarray:
        """
        特徴量変換を適用する。

        Args:
            X: 入力特徴量行列

        Returns:
            変換済み ndarray
        """
        cfg = self.config
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        if cfg.method == "none" or self._transformer is None:
            return X_arr

        return self._transformer.transform(X_arr)

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        """変換後の特徴量名を返す。"""
        cfg = self.config

        if cfg.method == "none" or self._transformer is None:
            return np.array(self._feature_names_in)

        names = input_features if input_features is not None else self._feature_names_in
        return self._transformer.get_feature_names_out(names)

    @property
    def is_passthrough(self) -> bool:
        """method=none（パススルー）かどうかを返す。"""
        return self.config.method == "none"

    @property
    def n_output_features(self) -> int:
        """変換後の特徴量数を返す（fit 後のみ有効）。"""
        if self._transformer is not None:
            return self._transformer.n_output_features_
        return self._n_features_in
