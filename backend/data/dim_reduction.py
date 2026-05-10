"""
backend/data/dim_reduction.py

次元削減モジュール。PCA / t-SNE / UMAP をsklearn互換Transformerとして提供する。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from backend.utils.optional_import import safe_import

logger = logging.getLogger(__name__)

# 任意依存 (遅延インポートにより環境エラーを絶縁)
def _get_umap_class() -> Any:
    try:
        mod = safe_import("umap", alias="umap-learn")
        return getattr(mod, "UMAP", None) if mod is not None else None
    except Exception:
        return None


@dataclass
class DimReductionConfig:
    """次元削減の設定。"""
    method: str = "pca"            # "pca" | "tsne" | "umap"
    n_components: int = 2
    # PCA
    whiten: bool = False
    # t-SNE
    perplexity: float = 30.0
    tsne_max_iter: int = 1000
    # UMAP
    n_neighbors: int = 15
    min_dist: float = 0.1
    # 共通
    random_state: int = 42
    scale: bool = True             # 事前にStandardScalerを適用するか
    method_params: dict[str, Any] = field(default_factory=dict) # 追加の任意引数


class DimReducer(BaseEstimator, TransformerMixin):
    """
    次元削減Transformer (PCA / t-SNE / UMAP)。
    """

    def __init__(self, config: DimReductionConfig | None = None) -> None:
        self.config = config or DimReductionConfig()
        self._scaler: StandardScaler | None = None
        self._reducer: Any = None
        self._embedding: np.ndarray | None = None
        self._feature_names_in: list[str] = []
        self._reconstruction_error: np.ndarray | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "DimReducer":
        """次元削減モデルをfitする。"""
        X_arr = self._prepare(X)
        cfg = self.config
        n_samples, n_features = X_arr.shape

        if cfg.method == "pca":
            n_comp = min(cfg.n_components, n_features)
            # 優先引数
            pca_kwargs = {
                "n_components": n_comp,
                "whiten": cfg.whiten,
                "random_state": cfg.random_state,
                **cfg.method_params
            }
            # 大規模データのバッチ処理対応 (Hardening 要件)
            if n_samples > 10000 and n_features > 100 and "IncrementalPCA" not in str(cfg.method_params):
                # IncrementalPCAはrandom_stateを持たないので調整
                pca_kwargs.pop("random_state", None)
                self._reducer = IncrementalPCA(**pca_kwargs)
            else:
                self._reducer = PCA(**pca_kwargs)
            self._reducer.fit(X_arr)

        elif cfg.method == "tsne":
            tsne_kwargs = {
                "n_components": cfg.n_components,
                "perplexity": cfg.perplexity,
                "max_iter": cfg.tsne_max_iter,
                "random_state": cfg.random_state,
                **cfg.method_params
            }
            self._reducer = TSNE(**tsne_kwargs)
            self._embedding = self._reducer.fit_transform(X_arr)

        elif cfg.method == "umap":
            umap_cls = _get_umap_class()
            if umap_cls is None:
                raise ImportError("UMAPを使用するには 'umap-learn' が必要です。")
            umap_kwargs = {
                "n_components": cfg.n_components,
                "n_neighbors": cfg.n_neighbors,
                "min_dist": cfg.min_dist,
                "random_state": cfg.random_state,
                **cfg.method_params
            }
            self._reducer = umap_cls(**umap_kwargs)
            self._embedding = self._reducer.fit_transform(X_arr)
        else:
            raise ValueError(f"未知の次元削減手法: {cfg.method}")

        return self

    def transform(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> np.ndarray:
        """次元削減を適用し、PCAの場合は再構成誤差を計算する。"""
        assert self._reducer is not None, "fit()を先に呼んでください。"
        X_arr = self._prepare(X, fit=False)
        
        if self.config.method == "pca":
            X_reduced = self._reducer.transform(X_arr)
            # 数学的厳密性: 再構成誤差 (Q-residual) の計算
            # 投影後の空間から元の空間に戻し、その差の L2 ノルムを計算する
            X_recon = self._reducer.inverse_transform(X_reduced)
            # 行ごとの Frobenius norm (Euclidean distance)
            self._reconstruction_error = np.linalg.norm(X_arr - X_recon, axis=1)
            return X_reduced
        
        if self.config.method in ("tsne", "umap"):
            if self._embedding is not None:
                return self._embedding
            return self._reducer.fit_transform(X_arr)
        
        raise ValueError(f"未知手法: {self.config.method}")

    def fit_transform(self, X: np.ndarray | pd.DataFrame, y: Any = None, **params: Any) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _prepare(self, X: np.ndarray | pd.DataFrame, fit: bool = True) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            # 汎用対応: 文字列列(SMILESなど)が混入した場合にPCAエラーになるのを防ぐため、数値列のみに絞る
            numeric_df = X.select_dtypes(include="number")
            self._feature_names_in = numeric_df.columns.tolist()
            X_arr = numeric_df.values.astype(float)
        else:
            X_arr = np.asarray(X)
            # numpy arrayの場合は数値型に変換できない要素があればエラーになるのを避けるため
            # もし object 型の場合は数値化を試みる (無理なら元のままにして後続でエラーを出す)
            if X_arr.dtype == object:
                try:
                    X_arr = X_arr.astype(float)
                except ValueError:
                    # そのまま通して後続の適切なエラー表示に任せる
                    pass

        if self.config.scale:
            if fit:
                self._scaler = StandardScaler()
                X_arr = self._scaler.fit_transform(X_arr)
            elif self._scaler is not None:
                X_arr = self._scaler.transform(X_arr)
        return X_arr

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        cfg = self.config
        return np.array([f"{cfg.method}_{i}" for i in range(cfg.n_components)])

    @property
    def explained_variance_ratio_(self) -> np.ndarray | None:
        """PCAの寄与率。"""
        return getattr(self._reducer, "explained_variance_ratio_", None)

    @property
    def loadings_(self) -> np.ndarray | None:
        """PCAの成分重み (Loadings)。"""
        return getattr(self._reducer, "components_", None)

    @property
    def feature_names_in_(self) -> list[str]:
        """学習時の特徴量名。"""
        return self._feature_names_in

    @property
    def reconstruction_error_(self) -> np.ndarray | None:
        """PCAの再構成誤差。"""
        return self._reconstruction_error


# ============================================================
# 便利関数 (旧来のAPIを100%復元)
# ============================================================

def run_pca(
    df: pd.DataFrame,
    n_components: int = 2,
    scale: bool = True,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """DataFrameにPCAを適用して2D埋め込み + 寄与率を返す。"""
    numeric = df.select_dtypes(include="number")
    if target_col and target_col in numeric.columns:
        numeric = numeric.drop(columns=[target_col])

    cfg = DimReductionConfig(method="pca", n_components=n_components, scale=scale)
    reducer = DimReducer(cfg)
    embedding = reducer.fit_transform(numeric)

    col_names = [f"PC{i+1}" for i in range(n_components)]
    emb_df = pd.DataFrame(embedding, columns=col_names, index=df.index)
    evr = reducer.explained_variance_ratio_
    return emb_df, evr if evr is not None else np.array([])


def run_tsne(
    df: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    scale: bool = True,
    target_col: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """DataFrameにt-SNEを適用して2D埋め込みを返す。"""
    numeric = df.select_dtypes(include="number")
    if target_col and target_col in numeric.columns:
        numeric = numeric.drop(columns=[target_col])

    n = len(numeric)
    perplexity = min(perplexity, (n - 1) / 3)

    cfg = DimReductionConfig(
        method="tsne", n_components=n_components, perplexity=perplexity,
        scale=scale, random_state=random_state,
    )
    reducer = DimReducer(cfg)
    embedding = reducer.fit_transform(numeric)

    col_names = [f"tSNE{i+1}" for i in range(n_components)]
    return pd.DataFrame(embedding, columns=col_names, index=df.index)


def run_umap(
    df: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    scale: bool = True,
    target_col: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """DataFrameにUMAPを適用して埋め込みを返す。"""
    numeric = df.select_dtypes(include="number")
    if target_col and target_col in numeric.columns:
        numeric = numeric.drop(columns=[target_col])

    cfg = DimReductionConfig(
        method="umap", n_components=n_components, n_neighbors=n_neighbors,
        min_dist=min_dist, scale=scale, random_state=random_state,
    )
    reducer = DimReducer(cfg)
    embedding = reducer.fit_transform(numeric)

    col_names = [f"UMAP{i+1}" for i in range(n_components)]
    return pd.DataFrame(embedding, columns=col_names, index=df.index)


import logging

def compute_dim_reduction_and_importance(df: pd.DataFrame) -> dict:
    """PCA/t-SNE座標および特徴量重要度を計算"""
    try:
        # 数値列のみ抽出、欠損値80%超の列は除外
        num_df = df.select_dtypes(include=["number"]).dropna(axis=1, thresh=len(df) * 0.2)
        num_df = num_df.dropna(axis=0)  # 行欠損も除去
        
        # 零分散列の除去
        if not num_df.empty:
            variances = num_df.var()
            num_df = num_df.loc[:, variances > 1e-9]
            
        if num_df.shape[1] < 2:
            return {"status": "skip", "message": "有効な数値特徴量が2列未満です"}
        if len(num_df) < 5:
            return {"status": "skip", "message": "サンプル数が不足しています（最低5行必要）"}

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(num_df)

        # PCA計算
        from sklearn.decomposition import PCA
        n_comp = min(3, num_df.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        pca_coords = pca.fit_transform(X_scaled)
        pca_importance = pd.DataFrame(
            np.abs(pca.components_[:2]),
            columns=num_df.columns,
            index=["PC1", "PC2"]
        ).T

        # t-SNE計算（データサイズに応じてperplexityを自動調整）
        from sklearn.manifold import TSNE
        perp = float(max(2.0, min(30.0, (len(num_df) - 1) / 3.0))) # max_iter as n_iter was deprecated
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", max_iter=1000)
        tsne_coords = tsne.fit_transform(X_scaled)

        # t-SNE重要度（各特徴量と埋め込み軸とのSpearman相関）
        tsne_importance = pd.DataFrame({
            "t-SNE1": num_df.apply(lambda c: c.rank(method="average").corr(pd.Series(tsne_coords[:, 0]), method="spearman")),
            "t-SNE2": num_df.apply(lambda c: c.rank(method="average").corr(pd.Series(tsne_coords[:, 1]), method="spearman"))
        }).abs()

        return {
            "status": "success",
            "pca_coords": pd.DataFrame(pca_coords[:, :2], columns=["PC1", "PC2"]),
            "pca_importance": pca_importance,
            "tsne_coords": pd.DataFrame(tsne_coords, columns=["t-SNE1", "t-SNE2"]),
            "tsne_importance": tsne_importance,
            "explained_var": pca.explained_variance_ratio_[:2],
            "n_features": num_df.shape[1],
            "n_samples": num_df.shape[0]
        }

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"次元削減計算失敗: {e}", exc_info=True)
        return {"status": "error", "message": f"計算エラー: {str(e)}"}
