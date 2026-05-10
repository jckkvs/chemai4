"""
backend/data/preprocessor.py

sklearn ColumnTransformer を使った前処理パイプライン構築モジュール。
TypeDetector の DetectionResult を受け取り、変数種別に応じた
Scaler/Encoder/Imputer を自動選択して Pipeline を組み上げる。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
from sklearn.base import BaseEstimator, TransformerMixin

from backend.data.type_detector import ColumnType, DetectionResult
from backend.utils.config import RANDOM_STATE
from backend.utils.optional_import import safe_import

logger = logging.getLogger(__name__)

# オプショナルライブラリ
_target_encoder = safe_import("sklearn.preprocessing", "target_encoder")
_category_encoders = safe_import("category_encoders", "category_encoders")


# ============================================================
# カスタムTransformer
# ============================================================

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    対数変換を行うカスタムTransformer。
    負の値や0への対処として offset を加算してから log1p を適用する。
    """

    def __init__(self, offset: float = 1.0) -> None:
        self.offset = offset

    def fit(self, X: np.ndarray, y: Any = None) -> "LogTransformer":
        return self

    def transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.log1p(X_arr + max(0, self.offset - 1))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.expm1(X_arr) - max(0, self.offset - 1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return input_features


class SinCosTransformer(BaseEstimator, TransformerMixin):
    """
    周期変数（角度, 時刻等）に対して sin/cos 変換を行うTransformer。
    period = 2π（ラジアン）, 360（度数）, 24（時刻）などを設定する。
    """

    def __init__(self, period: float = 2 * np.pi) -> None:
        self.period = period

    def fit(self, X: np.ndarray, y: Any = None) -> "SinCosTransformer":
        return self

    def transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        angle = 2 * np.pi * X_arr / self.period
        return np.column_stack([np.sin(angle), np.cos(angle)])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(["sin", "cos"])


# ============================================================
# Scaler / Encoder の設定クラス
# ============================================================

@dataclass
class PreprocessConfig:
    """
    前処理パイプラインの設定。AutoMLモードでは自動設定、
    エキスパートモードではユーザーが明示的に指定できる。
    """
    # 数値変数
    numeric_scaler: str = "auto"
    # "auto" | "standard" | "minmax" | "robust" | "maxabs" | "power_yj"
    # | "power_bc" | "quantile_normal" | "quantile_uniform" | "log" | "none"

    # カテゴリ変数（低cardinality）
    cat_low_encoder: str = "onehot"
    # "onehot" | "ordinal" | "target" | "binary" | "woe"

    # カテゴリ変数（高cardinality）
    cat_high_encoder: str = "ordinal"
    # "target" | "hashing" | "binary" | "leaveoneout" | "ordinal"

    # 欠損値補完
    numeric_imputer: str = "mean"
    # "mean" | "median" | "knn" | "iterative" | "constant"
    categorical_imputer: str = "most_frequent"
    # "most_frequent" | "constant"

    # 欠損インジケーター追加フラグ
    add_missing_indicator: bool = True

    # OneHotEncoder設定
    onehot_drop: str | None = "first"
    onehot_handle_unknown: str = "ignore"
    onehot_max_categories: int | None = None

    # Power Transformer設定
    power_method: str = "yeo-johnson"  # "yeo-johnson" | "box-cox"

    # QuantileTransformer設定
    quantile_n_quantiles: int = 1000
    quantile_output_distribution: str = "normal"  # "normal" | "uniform"

    # 周期変数の周期設定 {col_name: period}
    periodic_periods: dict[str, float] = field(default_factory=dict)

    # 除外する列（前処理から除外）
    passthrough_cols: list[str] = field(default_factory=list)

    # SMILES列・DateTime列は除外（別モジュールで処理）
    exclude_smiles: bool = True
    exclude_datetime: bool = True
    exclude_constant: bool = True

    # ── Johnson-Lindenstrauss ランダム射影 ──────────────────────────────────
    # True: n_features > jl_min_dim(n_samples, eps) の場合のみ自動適用
    # False: 常にスキップ（デフォルト: 無効 = ユーザーが明示的に有効化）
    random_projection_enable: bool = False
    random_projection_eps: float = 0.1   # JL歪み許容誤差 (0 < eps < 1)
    random_projection_method: str = "auto"  # "auto" | "sparse" | "gaussian"


# ============================================================
# メインクラス
# ============================================================

class Preprocessor:
    """
    TypeDetector の結果を受け取り、sklearn ColumnTransformer パイプラインを
    自動構築するクラス。

    Implements: 要件定義書 §3.3 前処理パイプライン

    Args:
        config: PreprocessConfig インスタンス
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self._transformer: ColumnTransformer | None = None
        self._feature_names_out: list[str] = []

    def build(
        self,
        detection_result: DetectionResult,
        target_col: str | None = None,
    ) -> ColumnTransformer:
        """
        DetectionResult に基づいて ColumnTransformer を構築して返す。
        fit は呼ばれない（fit/transformはユーザーが呼ぶ）。

        Args:
            detection_result: TypeDetector.detect() の結果
            target_col: 目的変数の列名（前処理対象から除外する）

        Returns:
            fitted前のColumnTransformer
        """
        cfg = self.config
        transformers: list[tuple] = []
        col_info = detection_result.column_info

        # 除外列のセット
        exclude: set[str] = set(cfg.passthrough_cols)
        if target_col:
            exclude.add(target_col)
        if cfg.exclude_smiles:
            exclude.update(detection_result.smiles_columns)
        if cfg.exclude_datetime:
            exclude.update(detection_result.datetime_columns)
        if cfg.exclude_constant:
            exclude.update(detection_result.constant_columns)

        # ---- 数値列 ----
        numeric_normal = [
            c for c, i in col_info.items()
            if i.col_type == ColumnType.NUMERIC_NORMAL and c not in exclude
        ]
        numeric_log = [
            c for c, i in col_info.items()
            if i.col_type == ColumnType.NUMERIC_LOG and c not in exclude
        ]
        numeric_power = [
            c for c, i in col_info.items()
            if i.col_type == ColumnType.NUMERIC_POWER and c not in exclude
        ]
        numeric_outlier = [
            c for c, i in col_info.items()
            if i.col_type == ColumnType.NUMERIC_OUTLIER and c not in exclude
        ]

        if numeric_normal:
            transformers.append((
                "num_normal",
                self._build_numeric_pipeline(cfg.numeric_scaler, "normal"),
                numeric_normal,
            ))
        if numeric_log:
            transformers.append((
                "num_log",
                self._build_numeric_pipeline("log", "log"),
                numeric_log,
            ))
        if numeric_power:
            transformers.append((
                "num_power",
                self._build_numeric_pipeline("power_yj", "power"),
                numeric_power,
            ))
        if numeric_outlier:
            transformers.append((
                "num_outlier",
                self._build_numeric_pipeline("robust", "outlier"),
                numeric_outlier,
            ))

        # ---- バイナリ列 ----
        binary_cols = [
            c for c, i in col_info.items()
            if i.col_type == ColumnType.BINARY and c not in exclude
        ]
        if binary_cols:
            transformers.append((
                "binary",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                              unknown_value=-1)),
                ]),
                binary_cols,
            ))

        # ---- カテゴリ(低) ----
        cat_low_cols = [
            c for c, i in col_info.items()
            if i.col_type == ColumnType.CATEGORY_LOW and c not in exclude
        ]
        if cat_low_cols:
            transformers.append((
                "cat_low",
                self._build_categorical_pipeline(cfg.cat_low_encoder, "low"),
                cat_low_cols,
            ))

        # ---- カテゴリ(高) ----
        cat_high_cols = [
            c for c, i in col_info.items()
            if i.col_type == ColumnType.CATEGORY_HIGH and c not in exclude
        ]
        if cat_high_cols:
            transformers.append((
                "cat_high",
                self._build_categorical_pipeline(cfg.cat_high_encoder, "high"),
                cat_high_cols,
            ))

        # ---- 周期変数 ----
        periodic_cols = [
            c for c, i in col_info.items()
            if i.col_type == ColumnType.PERIODIC and c not in exclude
        ]
        for col in periodic_cols:
            period = cfg.periodic_periods.get(col, 2 * np.pi)
            transformers.append((
                f"periodic_{col}",
                Pipeline([
                    ("impute", SimpleImputer(strategy="mean")),
                    ("sincos", SinCosTransformer(period=period)),
                ]),
                [col],
            ))

        # passthrough
        passthrough_active = [c for c in cfg.passthrough_cols if c in col_info]
        if passthrough_active:
            transformers.append(("passthrough", "passthrough", passthrough_active))

        if not transformers:
            logger.warning("前処理対象の列がすべて除外されました（定数列のみなど）。学習を継続するため、安全なフォールバックとして最初の列をパススルーします。")
            first_col = list(col_info.keys())[0] if col_info else None
            if first_col:
                transformers.append(("fallback_passthrough", "passthrough", [first_col]))
            else:
                raise ValueError("入力データに列が1つも存在しないため、前処理パイプラインを構築できません。")

        ct = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self._transformer = ct
        logger.info(
            f"ColumnTransformer構築完了: {len(transformers)}グループ / "
            f"除外列={len(exclude)}"
        )
        return ct

    def _build_numeric_pipeline(self, scaler_name: str, context: str) -> Pipeline:
        """数値列用のImputer + Scalerパイプラインを構築する。"""
        cfg = self.config
        steps: list[tuple] = []

        # Imputer
        steps.append(("impute", self._build_numeric_imputer()))
        if cfg.add_missing_indicator:
            # MissingIndicator は ColumnTransformer 外で追加するのが一般的だが、
            # FunctionTransformerなどで将来的に検討
            logger.debug("MissingIndicator is not yet implemented in this pipeline")

        # Scaler / Transformer
        effective_scaler = scaler_name if scaler_name != "auto" else "standard"

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
        scaler = scaler_map.get(effective_scaler, StandardScaler())
        if isinstance(scaler, Pipeline):
            # ログ変換は複合なのでflatに展開
            for name, step in scaler.steps:
                steps.append((name, step))
        else:
            steps.append(("scale", scaler))

        return Pipeline(steps)

    def _build_numeric_imputer(self) -> Any:
        """数値欠損補完のImputer を設定に基づいて返す。"""
        strategy = self.config.numeric_imputer
        if strategy == "iterative":
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                return IterativeImputer(random_state=RANDOM_STATE, max_iter=10)
            except ImportError:
                logger.warning("IterativeImputer 未対応 → SimpleImputer(mean)で代替")
                return SimpleImputer(strategy="mean")
        elif strategy == "knn":
            return KNNImputer(n_neighbors=5)
        elif strategy in ("mean", "median", "constant"):
            return SimpleImputer(strategy=strategy)
        else:
            return SimpleImputer(strategy="mean")

    def _build_categorical_pipeline(self, encoder_name: str, cardinality: str) -> Pipeline:
        """カテゴリ列用のImputer + Encoderパイプラインを構築する。"""
        cfg = self.config
        steps: list[tuple] = [
            ("impute", SimpleImputer(strategy=cfg.categorical_imputer)),
        ]

        # Encoder
        if encoder_name == "onehot":
            encoder = OneHotEncoder(
                drop=cfg.onehot_drop,
                handle_unknown=cfg.onehot_handle_unknown,
                sparse_output=False,
                max_categories=cfg.onehot_max_categories,
            )
        elif encoder_name == "ordinal":
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        elif encoder_name == "target":
            try:
                from sklearn.preprocessing import TargetEncoder  # sklearn 1.3+
                encoder = TargetEncoder(smooth="auto")
            except ImportError:
                logger.warning("TargetEncoder 未対応バージョン → OrdinalEncoderで代替")
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                         unknown_value=-1)
        elif encoder_name == "binary":
            if _category_encoders:
                import category_encoders as ce  # type: ignore
                encoder = ce.BinaryEncoder()
            else:
                logger.warning("category_encoders 未インストール → OneHotEncoderで代替")
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif encoder_name == "hashing":
            if _category_encoders:
                import category_encoders as ce  # type: ignore
                encoder = ce.HashingEncoder(n_components=8)
            else:
                logger.warning("category_encoders 未インストール → OrdinalEncoderで代替")
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                         unknown_value=-1)
        elif encoder_name == "leaveoneout":
            if _category_encoders:
                import category_encoders as ce  # type: ignore
                encoder = ce.LeaveOneOutEncoder(random_state=RANDOM_STATE)
            else:
                logger.warning("category_encoders 未インストール → OrdinalEncoderで代替")
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                         unknown_value=-1)
        elif encoder_name == "woe":
            if _category_encoders:
                import category_encoders as ce  # type: ignore
                encoder = ce.WOEEncoder(random_state=RANDOM_STATE)
            else:
                logger.warning("category_encoders 未インストール → OrdinalEncoderで代替")
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                         unknown_value=-1)
        else:
            logger.warning(f"未知のエンコーダー名 '{encoder_name}' → OneHotEncoder使用")
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        steps.append(("encode", encoder))
        return Pipeline(steps)

    @property
    def transformer(self) -> ColumnTransformer:
        """構築済みの ColumnTransformer を返す。未構築の場合は例外。"""
        if self._transformer is None:
            raise RuntimeError("build() を先に呼び出してください。")
        return self._transformer


def build_full_pipeline(
    detection_result: DetectionResult,
    model: Any,
    target_col: str | None = None,
    config: PreprocessConfig | None = None,
) -> Pipeline:
    """
    前処理 + (JL-RP) + モデルの sklearn Pipeline を構築して返す。

    特徴量数がJL補題の要求次元を超える場合、自動的に
    JLRandomProjection ステップが挿入される。
    config.random_projection_enable=False（デフォルト）の場合はスキップ。

    Args:
        detection_result: TypeDetector の判定結果
        model: sklearn互換の推定器（fit/predictを持つもの）
        target_col: 目的変数列名（前処理から除外）
        config: PreprocessConfig（省略時はデフォルト）

    Returns:
        Pipeline([("preprocess", ColumnTransformer)
                  (, "jl_rp", JLRandomProjection)  # 条件成立時のみ
                  ("model", model)])
    """
    cfg = config or PreprocessConfig()
    preprocessor = Preprocessor(cfg)
    ct = preprocessor.build(detection_result, target_col=target_col)

    steps: list[tuple] = [("preprocess", ct)]

    # JL Random Projection ステップ
    if cfg.random_projection_enable:
        from backend.data.random_projection import JLRandomProjection
        jl_rp = JLRandomProjection(
            eps=cfg.random_projection_eps,
            method=cfg.random_projection_method,
        )
        steps.append(("jl_rp", jl_rp))
        logger.info(
            f"JLRandomProjection ステップを追加 "
            f"(eps={cfg.random_projection_eps}, method={cfg.random_projection_method}). "
            f"fit()時に自動判定: n_features > jl_min_dim のみ実際に適用."
        )

    steps.append(("model", model))
    return Pipeline(steps)
