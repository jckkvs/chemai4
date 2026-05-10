"""
backend/interpret/shap_explainer.py

SHAPを使ったモデル解釈モジュール。
TreeExplainer, LinearExplainer, KernelExplainer, DeepExplainer に対応。
各種可視化プロット (Summary, Waterfall, Force, Dependence, Heatmap, Decision) を提供。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.utils.optional_import import require, safe_import
from backend.utils.config import SHAP_MAX_DISPLAY, SHAP_KERNEL_NSAMPLES, SHAP_KERNEL_NSAMPLES_MAX

_shap = safe_import("shap", "shap")

logger = logging.getLogger(__name__)


@dataclass
class ShapConfig:
    """SHAP計算の設定を保持するデータクラス。

    Attributes:
        explainer_type: Explainerの種類 ("auto" | "tree" | "linear" | "kernel" | "deep")
        max_display: プロットに表示する特徴量の最大数
        background_size: KernelExplainerのバックグラウンドサンプル数
        plot_types: 生成するプロット種別のリスト
        kernel_nsamples: KernelExplainerのnsamples (backgroundと別)
    """
    explainer_type: str = "auto"
    max_display: int = SHAP_MAX_DISPLAY
    background_size: int = 100
    plot_types: list[str] = field(default_factory=lambda: ["summary", "waterfall", "dependence"])
    kernel_nsamples: int = SHAP_KERNEL_NSAMPLES
    kernel_nsamples_max: int = SHAP_KERNEL_NSAMPLES_MAX


@dataclass
class ShapResult:
    """SHAP計算結果を保持するデータクラス。"""
    shap_values: np.ndarray             # shape: (n_samples, n_features) for single output
    expected_value: float | np.ndarray
    feature_names: list[str]
    X_transformed: np.ndarray          # 変換後の特徴量行列
    explainer_type: str                 # "tree" | "linear" | "kernel" | "deep"
    is_multiclass: bool = False
    shap_interaction_values: np.ndarray | None = None
    base_values: np.ndarray | None = None

    def feature_importance(self) -> pd.DataFrame:
        """SHAP値の絶対値平均から特徴量重要度DataFrameを返す。

        Returns:
            {feature, importance} のDataFrame（降順ソート済み）
        """
        sv = self.shap_values
        if self.is_multiclass and sv.ndim == 3:
            imp = np.abs(sv).mean(axis=(0, 2))
        else:
            imp = np.abs(sv).mean(axis=0)

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def top_features(self, n: int = 10) -> list[str]:
        """重要度上位n件の特徴量名リストを返す。

        Args:
            n: 返す特徴量数

        Returns:
            特徴量名のリスト（降順）
        """
        fi = self.feature_importance()
        return fi["feature"].head(n).tolist()


class ShapExplainer:
    """
    sklearn/XGBoost/LightGBM/CatBoost等のモデルに対してSHAPを計算するクラス。

    Implements: 要件定義書 §3.8 モデル解釈 (SHAP)

    Args:
        config_or_max_display: ShapConfig または max_display (int)
        kernel_nsamples: KernelExplainerのサンプル数 (config未使用時)
    """

    def __init__(
        self,
        config_or_max_display: ShapConfig | int = SHAP_MAX_DISPLAY,
        kernel_nsamples: int = SHAP_KERNEL_NSAMPLES,
        kernel_nsamples_max: int = SHAP_KERNEL_NSAMPLES_MAX,
    ) -> None:
        require("shap", feature="SHAP解釈")
        if isinstance(config_or_max_display, ShapConfig):
            self._config = config_or_max_display
            self.max_display = config_or_max_display.max_display
            self.kernel_nsamples = config_or_max_display.kernel_nsamples
            self.kernel_nsamples_max = config_or_max_display.kernel_nsamples_max
        else:
            self._config = ShapConfig(
                max_display=config_or_max_display,
                kernel_nsamples=kernel_nsamples,
                kernel_nsamples_max=kernel_nsamples_max,
            )
            self.max_display = config_or_max_display
            self.kernel_nsamples = kernel_nsamples
            self.kernel_nsamples_max = kernel_nsamples_max

    def _select_explainer_type(self, model: Any) -> str:
        """モデルの種類に応じてExplainerタイプ文字列を返す。

        Args:
            model: 学習済みモデル

        Returns:
            "tree" | "linear" | "kernel" | "deep"
        """
        if self._config.explainer_type != "auto":
            return self._config.explainer_type

        model_type = type(model).__name__.lower()

        tree_keywords = ["tree", "forest", "boost", "xgb", "lgbm", "lgb",
                         "catboost", "gradient", "ada", "extra", "bagging"]
        if any(kw in model_type for kw in tree_keywords):
            return "tree"

        linear_keywords = ["linear", "logistic", "ridge", "lasso", "elastic", "pls",
                           "bayesianridge", "ard", "huber"]
        if any(kw in model_type for kw in linear_keywords):
            return "linear"

        deep_keywords = ["torch", "keras", "tensorflow", "neural"]
        if any(kw in model_type for kw in deep_keywords):
            return "deep"

        return "kernel"

    def explain(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        feature_names: list[str] | None = None,
        background_data: np.ndarray | pd.DataFrame | None = None,
        compute_interactions: bool = False,
    ) -> ShapResult:
        """
        モデルとデータからSHAP値を計算する。
        モデルの種類を自動判定してExplainerを選択する。

        Args:
            model: 学習済みモデル
            X: SHAP値を計算するデータ
            feature_names: 特徴量名リスト（省略時はDataFrameの列名を使用）
            background_data: KernelExplainerのバックグラウンドデータ
            compute_interactions: SHAP Interaction Valuesを計算するか

        Returns:
            ShapResult インスタンス
        """
        import shap  # type: ignore

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        fnames = (
            list(X.columns) if isinstance(X, pd.DataFrame)
            else (feature_names or [f"f{i}" for i in range(X_arr.shape[1])])
        )

        explainer_type, explainer = self._build_explainer(
            shap, model, X_arr, background_data
        )

        shap_values = explainer(X_arr)

        # shap.Explanation または np.ndarray に対応
        if hasattr(shap_values, "values"):
            sv_arr = shap_values.values
            base_values = shap_values.base_values
        else:
            sv_arr = np.asarray(shap_values)
            base_values = None

        # マルチクラス判定（3次元の場合）
        is_multiclass = sv_arr.ndim == 3

        expected_value = explainer.expected_value
        if hasattr(expected_value, "__len__"):
            expected_value = np.asarray(expected_value)

        # Interaction Values
        interaction_values = None
        if compute_interactions and explainer_type == "tree":
            try:
                interaction_values = explainer.shap_interaction_values(X_arr)
            except Exception as e:
                logger.warning(f"Interaction Values の計算に失敗: {e}")

        return ShapResult(
            shap_values=sv_arr,
            expected_value=expected_value,
            feature_names=fnames,
            X_transformed=X_arr,
            explainer_type=explainer_type,
            is_multiclass=is_multiclass,
            shap_interaction_values=interaction_values,
            base_values=base_values,
        )

    def _build_explainer(
        self,
        shap: Any,
        model: Any,
        X: np.ndarray,
        background_data: np.ndarray | None,
    ) -> tuple[str, Any]:
        """モデルの種類に応じてExplainerを選択して返す。"""
        model_type = type(model).__name__.lower()

        # TreeExplainer: tree系モデル
        tree_keywords = ["tree", "forest", "boost", "xgb", "lgbm", "lgb",
                         "catboost", "gradient", "ada", "extra", "bagging"]
        if any(kw in model_type for kw in tree_keywords):
            logger.info(f"TreeExplainer を使用 (model={model_type})")
            try:
                return "tree", shap.TreeExplainer(model)
            except Exception as e:
                logger.warning(f"TreeExplainer失敗: {e}。KernelExplainerに切り替え")

        # LinearExplainer: 線形モデル
        linear_keywords = ["linear", "logistic", "ridge", "lasso", "elastic", "pls",
                           "bayesianridge", "ard", "huber"]
        if any(kw in model_type for kw in linear_keywords):
            logger.info(f"LinearExplainer を使用 (model={model_type})")
            try:
                bg = background_data if background_data is not None else X
                return "linear", shap.LinearExplainer(model, bg)
            except Exception as e:
                logger.warning(f"LinearExplainer失敗: {e}。KernelExplainerに切り替え")

        # DeepExplainer: ニューラルネット (torch/tf)
        deep_keywords = ["torch", "keras", "tensorflow", "neural"]
        if any(kw in model_type for kw in deep_keywords):
            logger.info("DeepExplainer を使用")
            bg = background_data if background_data is not None else X[:50]
            try:
                return "deep", shap.DeepExplainer(model, bg)
            except Exception as e:
                logger.warning(f"DeepExplainer失敗: {e}。KernelExplainerに切り替え")

        # KernelExplainer: フォールバック（モデル非依存）
        logger.info("KernelExplainer を使用（フォールバック）")
        n_samples = len(X) if background_data is None else len(background_data)
        
        if background_data is None:
            if n_samples > self.kernel_nsamples_max:
                logger.info(f"データ数({n_samples})上限超過。shap.kmeans()で{self.kernel_nsamples}点に縮約します。")
                bg = shap.kmeans(X, self.kernel_nsamples)
            else:
                bg = shap.sample(X, min(self.kernel_nsamples, n_samples))
        else:
            if n_samples > self.kernel_nsamples_max:
                logger.info(f"バックグラウンドデータ上限超過。shap.kmeans()で{self.kernel_nsamples}点に縮約します。")
                bg = shap.kmeans(background_data, self.kernel_nsamples)
            else:
                bg = background_data

        def _predict_fn(data: np.ndarray) -> np.ndarray:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(data)
            return model.predict(data)

        return "kernel", shap.KernelExplainer(_predict_fn, bg)

    # ---- 可視化メソッド ----

    def plot_summary(
        self,
        result: ShapResult,
        plot_type: str = "dot",
        save_path: str | None = None,
    ) -> None:
        """
        SHAPのSummaryプロット（ビープロット / バープロット）を表示・保存する。

        Args:
            result: ShapResult インスタンス
            plot_type: "dot" | "bar" | "violin"
            save_path: 保存パス（省略時は表示のみ）
        """
        import shap  # type: ignore
        import matplotlib.pyplot as plt

        sv = result.shap_values
        if result.is_multiclass:
            sv = sv[:, :, 0]  # 最初のクラス

        shap.summary_plot(
            sv,
            features=result.X_transformed,
            feature_names=result.feature_names,
            plot_type=plot_type,
            max_display=self.max_display,
            show=False,
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()
        plt.close()

    def plot_waterfall(
        self,
        result: ShapResult,
        sample_idx: int = 0,
        save_path: str | None = None,
    ) -> None:
        """
        指定サンプルのWaterfallプロットを表示する。

        Args:
            result: ShapResult インスタンス
            sample_idx: 可視化するサンプルのインデックス
            save_path: 保存パス
        """
        import shap  # type: ignore
        import matplotlib.pyplot as plt

        exp = shap.Explanation(
            values=result.shap_values[sample_idx],
            base_values=(
                result.expected_value
                if np.isscalar(result.expected_value)
                else result.expected_value[sample_idx]
            ),
            data=result.X_transformed[sample_idx],
            feature_names=result.feature_names,
        )
        shap.plots.waterfall(exp, max_display=self.max_display, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()
        plt.close()

    def plot_dependence(
        self,
        result: ShapResult,
        feature: str,
        interaction_feature: str = "auto",
        save_path: str | None = None,
    ) -> None:
        """
        Dependence プロットを表示する。

        Args:
            result: ShapResult インスタンス
            feature: 主特徴量名
            interaction_feature: 交互作用特徴量名（"auto"で自動選択）
            save_path: 保存パス
        """
        import shap  # type: ignore
        import matplotlib.pyplot as plt

        sv = result.shap_values
        if result.is_multiclass:
            sv = sv[:, :, 0]

        shap.dependence_plot(
            feature,
            sv,
            result.X_transformed,
            feature_names=result.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()
        plt.close()

    def get_feature_importance_df(self, result: ShapResult) -> pd.DataFrame:
        """
        SHAP値の絶対値平均から特徴量重要度DataFrameを返す（GUI表示用）。

        Returns:
            {feature, importance} のDataFrame（降順ソート済み）
        """
        sv = result.shap_values
        if result.is_multiclass:
            sv = np.abs(sv).mean(axis=(0, 2))
        else:
            sv = np.abs(sv).mean(axis=0)

        return pd.DataFrame({
            "feature": result.feature_names,
            "importance": sv,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
