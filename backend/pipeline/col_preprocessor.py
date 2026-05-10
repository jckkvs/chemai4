"""
backend/pipeline/col_preprocessor.py

列別前処理モジュール。
TypeDetector による自動判定をベースに、ユーザーが各列・各種別の
前処理を上書き設定できる ColumnTransformer ビルダー。

対応スケーラー:
  standard / minmax / robust / maxabs / power_yj / power_bc /
  quantile_normal / quantile_uniform / log / none

対応エンコーダー:
  onehot / ordinal / target / binary / woe / hashing / leaveoneout

対応 Imputer:
  mean / median / knn / iterative / constant / most_frequent
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    PowerTransformer,
    QuantileTransformer,
    OneHotEncoder,
    OrdinalEncoder,
)

from backend.data.type_detector import ColumnType, TypeDetector, DetectionResult
from backend.data.preprocessor import LogTransformer
from backend.utils.config import RANDOM_STATE
from backend.utils.optional_import import safe_import

logger = logging.getLogger(__name__)

_category_encoders = safe_import("category_encoders", "category_encoders")


# ============================================================
# 設定クラス
# ============================================================

@dataclass
class ColPreprocessConfig:
    """
    列別前処理の設定。

    TypeDetector による自動判定をベースとし、ユーザーが上書き可能。

    Attributes:
        override_types: 列名 → 型文字列の辞書でTypeDetector結果を上書き。
            有効値: "numeric" | "category_low" | "category_high" | "binary" | "passthrough"
        numeric_imputer: 数値列の欠損補間。
            "mean" | "median" | "knn" | "iterative" | "constant"
        numeric_scaler: 数値列のスケーリング・変換。
            "standard" | "minmax" | "robust" | "maxabs" |
            "power_yj" | "power_bc" | "quantile_normal" | "quantile_uniform" |
            "log" | "none"
        cat_low_encoder: 低カーディナリティカテゴリのエンコーダー。
            "onehot" | "ordinal" | "target" | "binary" | "woe"
        cat_high_encoder: 高カーディナリティカテゴリのエンコーダー。
            "target" | "hashing" | "binary" | "leaveoneout" | "ordinal"
        binary_encoder: バイナリ列のエンコーダー。
            "ordinal" | "passthrough"
        categorical_imputer: カテゴリ列の欠損補間。
            "most_frequent" | "constant"
        add_missing_indicator: 欠損インジケータ列を追加するか。
        cardinality_threshold: 低/高カーディナリティの境界ユニーク数。
        onehot_drop: OneHotEncoder の drop 設定 ("first" | "if_binary" | None)。
        onehot_handle_unknown: OneHotEncoder の handle_unknown。
        quantile_n_quantiles: QuantileTransformer の分位数。
        constant_fill_value: 定数補間時の値。
    """
    # TypeDetector 上書き: {"列名": "numeric" | "category_low" | "category_high" | "binary" | "passthrough"}
    override_types: dict[str, str] = field(default_factory=dict)

    # 数値列
    numeric_imputer: str = "mean"
    numeric_scaler: str = "standard"

    # カテゴリ列（低カーディナリティ）
    cat_low_encoder: str = "onehot"

    # カテゴリ列（高カーディナリティ）
    cat_high_encoder: str = "ordinal"

    # バイナリ列
    binary_encoder: str = "ordinal"
    binary_imputer: str = "most_frequent"  # バイナリ列の欠損補間: "most_frequent" | "constant" | "knn"

    # カテゴリ欠損処理
    categorical_imputer: str = "most_frequent"

    # 欠損インジケータ
    add_missing_indicator: bool = False

    # カーディナリティ境界
    cardinality_threshold: int = 20

    # OneHotEncoder 追加設定
    onehot_drop: str | None = "first"
    onehot_handle_unknown: str = "ignore"
    onehot_max_categories: int | None = None

    # QuantileTransformer
    quantile_n_quantiles: int = 1000

    # 定数補間値
    constant_fill_value: float | str = 0


# ============================================================
# メインクラス
# ============================================================

class ColPreprocessor(BaseEstimator, TransformerMixin):
    """
    列別前処理 Transformer。

    TypeDetector で列型を自動判定し、ColPreprocessConfig に従って
    各列に適切な前処理（Imputer + Scaler/Encoder）を適用する。
    pandas DataFrame の入力・出力に対応（set_output互換）。

    Args:
        config: ColPreprocessConfig（省略時はデフォルト設定）
    """

    def __init__(self, config: ColPreprocessConfig | None = None) -> None:
        self.config = config or ColPreprocessConfig()
        self._ct: ColumnTransformer | None = None
        self._detection: DetectionResult | None = None
        self._feature_names_out: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "ColPreprocessor":
        """
        データの列型を判定し ColumnTransformer を構築・fit する。

        Args:
            X: 入力 DataFrame
            y: 目的変数（Target Encoder 使用時に必要）

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        cfg = self.config

        # TypeDetector で型判定
        detector = TypeDetector(
            cardinality_threshold=cfg.cardinality_threshold,
        )
        self._detection = detector.detect(X)

        # 列グルーピング
        groups = self._group_columns(X, self._detection, cfg)

        # ColumnTransformer を構築
        transformers: list[tuple] = []

        if groups["numeric"]:
            transformers.append((
                "numeric",
                self._build_numeric_pipeline(cfg),
                groups["numeric"],
            ))

        if groups["category_low"]:
            transformers.append((
                "cat_low",
                self._build_categorical_pipeline(cfg.cat_low_encoder, cfg),
                groups["category_low"],
            ))

        if groups["category_high"]:
            transformers.append((
                "cat_high",
                self._build_categorical_pipeline(cfg.cat_high_encoder, cfg),
                groups["category_high"],
            ))

        if groups["binary"]:
            transformers.append((
                "binary",
                self._build_binary_pipeline(cfg),
                groups["binary"],
            ))

        if groups["passthrough"]:
            transformers.append(("passthrough", "passthrough", groups["passthrough"]))

        if not transformers:
            raise ValueError("前処理対象列が見つかりません。入力 DataFrame を確認してください。")

        self._ct = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self._ct.fit(X, y)

        # 出力列名を記録
        try:
            self._feature_names_out = self._ct.get_feature_names_out().tolist()
        except Exception:
            self._feature_names_out = []

        logger.info(
            f"ColPreprocessor.fit() 完了: "
            f"数値={len(groups['numeric'])}, "
            f"カテゴリ低={len(groups['category_low'])}, "
            f"カテゴリ高={len(groups['category_high'])}, "
            f"バイナリ={len(groups['binary'])}, "
            f"passthrough={len(groups['passthrough'])}"
        )
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> np.ndarray:
        """
        前処理を適用して変換済み配列を返す。

        Args:
            X: 入力 DataFrame

        Returns:
            変換済み ndarray
        """
        if self._ct is None:
            raise RuntimeError("fit() を先に呼び出してください。")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self._ct.transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """変換後の特徴量名を返す。"""
        return np.array(self._feature_names_out)

    @property
    def column_transformer(self) -> ColumnTransformer:
        """構築済み ColumnTransformer を返す。"""
        if self._ct is None:
            raise RuntimeError("fit() を先に呼び出してください。")
        return self._ct

    # ----------------------------------------------------------
    # 列グルーピング
    # ----------------------------------------------------------

    def _group_columns(
        self,
        X: pd.DataFrame,
        detection: DetectionResult,
        cfg: ColPreprocessConfig,
    ) -> dict[str, list[str]]:
        """列を種別ごとにグルーピングする。override_types を反映。"""
        groups: dict[str, list[str]] = {
            "numeric": [],
            "category_low": [],
            "category_high": [],
            "binary": [],
            "passthrough": [],
        }

        for col in X.columns:
            # ユーザー上書き
            if col in cfg.override_types:
                override = cfg.override_types[col]
                if override in groups:
                    groups[override].append(col)
                else:
                    logger.warning(f"列 '{col}': 無効な override_type '{override}' → 自動判定を使用")
                    self._auto_assign(col, detection, groups)
            else:
                self._auto_assign(col, detection, groups)

        return groups

    def _auto_assign(
        self,
        col: str,
        detection: DetectionResult,
        groups: dict[str, list[str]],
    ) -> None:
        """TypeDetector の結果に基づいて列をグループに割り当てる。"""
        info = detection.column_info.get(col)
        if info is None:
            groups["passthrough"].append(col)
            return

        ct = info.col_type
        if ct == ColumnType.BINARY:
            groups["binary"].append(col)
        elif ct == ColumnType.CATEGORY_LOW:
            groups["category_low"].append(col)
        elif ct == ColumnType.CATEGORY_HIGH:
            groups["category_high"].append(col)
        elif ct in (
            ColumnType.NUMERIC_NORMAL,
            ColumnType.NUMERIC_LOG,
            ColumnType.NUMERIC_POWER,
            ColumnType.NUMERIC_OUTLIER,
        ):
            groups["numeric"].append(col)
        else:
            # CONSTANT / DATETIME / TEXT / SMILES / PERIODIC → passthrough
            groups["passthrough"].append(col)

    # ----------------------------------------------------------
    # Pipeline 構築ヘルパー
    # ----------------------------------------------------------

    def _build_numeric_pipeline(self, cfg: ColPreprocessConfig) -> Pipeline:
        """数値列用 Imputer + Scaler パイプライン。"""
        steps: list[tuple] = [
            ("impute", self._build_numeric_imputer(cfg)),
        ]

        scaler = self._build_scaler(cfg.numeric_scaler, cfg)
        if isinstance(scaler, Pipeline):
            steps.extend(scaler.steps)
        elif scaler != "passthrough":
            steps.append(("scale", scaler))

        return Pipeline(steps)

    def _build_numeric_imputer(self, cfg: ColPreprocessConfig) -> Any:
        """数値列用 Imputer を返す。"""
        strategy = cfg.numeric_imputer
        if strategy == "iterative":
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                return IterativeImputer(random_state=RANDOM_STATE, max_iter=10)
            except ImportError:
                logger.warning("IterativeImputer 未対応 → SimpleImputer(mean) で代替")
                return SimpleImputer(strategy="mean")
        elif strategy == "knn":
            return KNNImputer(n_neighbors=5)
        elif strategy == "constant":
            return SimpleImputer(strategy="constant", fill_value=cfg.constant_fill_value)
        elif strategy in ("mean", "median"):
            return SimpleImputer(strategy=strategy)
        else:
            logger.warning(f"未知の numeric_imputer '{strategy}' → mean を使用")
            return SimpleImputer(strategy="mean")

    def _build_scaler(self, scaler_name: str, cfg: ColPreprocessConfig) -> Any:
        """スケーラー・変換器を返す。Pipelineを返す場合あり（log変換等）。"""
        scaler_map: dict[str, Any] = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler(),
            "power_yj": PowerTransformer(method="yeo-johnson"),
            "power_bc": PowerTransformer(method="box-cox"),
            "quantile_normal": QuantileTransformer(
                n_quantiles=cfg.quantile_n_quantiles,
                output_distribution="normal",
                random_state=RANDOM_STATE,
            ),
            "quantile_uniform": QuantileTransformer(
                n_quantiles=cfg.quantile_n_quantiles,
                output_distribution="uniform",
                random_state=RANDOM_STATE,
            ),
            "log": Pipeline([
                ("log", LogTransformer()),
                ("std", StandardScaler()),
            ]),
            "none": "passthrough",
        }
        result = scaler_map.get(scaler_name)
        if result is None:
            logger.warning(f"未知の numeric_scaler '{scaler_name}' → standard を使用")
            return StandardScaler()
        return result

    def _build_categorical_pipeline(
        self,
        encoder_name: str,
        cfg: ColPreprocessConfig,
    ) -> Pipeline:
        """カテゴリ列用 Imputer + Encoder パイプライン。"""
        steps: list[tuple] = [
            ("impute", SimpleImputer(strategy=cfg.categorical_imputer)),
            ("encode", self._build_encoder(encoder_name, cfg)),
        ]
        return Pipeline(steps)

    def _build_encoder(self, encoder_name: str, cfg: ColPreprocessConfig) -> Any:
        """エンコーダーを返す。category_encoders 未インストール時はフォールバック。"""
        if encoder_name == "onehot":
            return OneHotEncoder(
                drop=cfg.onehot_drop,
                handle_unknown=cfg.onehot_handle_unknown,
                sparse_output=False,
                max_categories=cfg.onehot_max_categories,
            )

        elif encoder_name == "ordinal":
            return OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )

        elif encoder_name == "target":
            try:
                from sklearn.preprocessing import TargetEncoder  # sklearn 1.3+
                return TargetEncoder(smooth="auto")
            except ImportError:
                logger.warning("TargetEncoder 未対応 → OrdinalEncoder で代替")
                return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        elif encoder_name == "binary":
            if _category_encoders:
                import category_encoders as ce  # type: ignore
                return ce.BinaryEncoder()
            logger.warning("category_encoders 未インストール → OneHotEncoder で代替")
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        elif encoder_name == "woe":
            if _category_encoders:
                import category_encoders as ce  # type: ignore
                return ce.WOEEncoder(random_state=RANDOM_STATE)
            logger.warning("category_encoders 未インストール → OrdinalEncoder で代替")
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        elif encoder_name == "hashing":
            if _category_encoders:
                import category_encoders as ce  # type: ignore
                return ce.HashingEncoder(n_components=8)
            logger.warning("category_encoders 未インストール → OrdinalEncoder で代替")
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        elif encoder_name == "leaveoneout":
            if _category_encoders:
                import category_encoders as ce  # type: ignore
                return ce.LeaveOneOutEncoder(random_state=RANDOM_STATE)
            logger.warning("category_encoders 未インストール → OrdinalEncoder で代替")
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        else:
            logger.warning(f"未知のエンコーダー名 '{encoder_name}' → OneHotEncoder を使用")
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def _build_binary_pipeline(self, cfg: ColPreprocessConfig) -> Pipeline:
        """バイナリ列用 Imputer + Encoder パイプライン。"""
        if cfg.binary_encoder == "passthrough":
            encoder: Any = "passthrough"
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )

        # binary_imputer を cfg から取得
        bi_strategy = cfg.binary_imputer
        if bi_strategy == "knn":
            binary_imputer_obj: Any = KNNImputer(n_neighbors=5)
        elif bi_strategy == "constant":
            binary_imputer_obj = SimpleImputer(strategy="constant", fill_value=cfg.constant_fill_value)
        else:
            binary_imputer_obj = SimpleImputer(strategy="most_frequent")

        steps: list[tuple] = [
            ("impute", binary_imputer_obj),
        ]
        if encoder != "passthrough":
            steps.append(("encode", encoder))

        return Pipeline(steps)
