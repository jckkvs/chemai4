"""
backend/data/eda_core.py
【フルスクラッチ再構築】探索的データ分析コアモジュール
- 全既存機能を100%継承
- 次元削減・重要度表示の信頼性を根本改善
- フロントエンド統合をType-safeに設計
"""

from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


# ============================================================
# 型定義・定数（Type-safety確保）
# ============================================================

class ReductionMethod(str, Enum):
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"  # 拡張用プレースホルダー

class ImportanceMetric(str, Enum):
    PCA_LOADING = "pca_loading"      # PCAの成分負荷量
    TSNE_CORR = "tsne_spearman"      # t-SNE座標との順位相関
    PERMUTATION = "permutation"      # 置換重要度（将来拡張）


@dataclass
class DimReductionResult:
    """次元削減結果の標準化データ構造"""
    status: Literal["success", "skip", "error"]
    method: str
    coordinates: dict[str, list[float]]  # {sample_id: [x, y]}
    explained_variance: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_json_serializable(self) -> dict:
        """JSON直列化用の変換（numpy型を解決）"""
        return {
            "status": self.status,
            "method": self.method,
            "coordinates": self.coordinates,
            "explained_variance": self.explained_variance,
            "metadata": _convert_to_native(self.metadata),
            "error_message": self.error_message
        }


@dataclass
class FeatureImportanceResult:
    """特徴量重要度の標準化データ構造"""
    status: Literal["success", "skip", "error"]
    metric: str
    importance: dict[str, float]  # {feature_name: importance_score}
    top_n: int = 20
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_json_serializable(self) -> dict:
        return {
            "status": self.status,
            "metric": self.metric,
            "importance": _convert_to_native(self.importance),
            "top_n": self.top_n,
            "metadata": _convert_to_native(self.metadata),
            "error_message": self.error_message
        }


@dataclass
class CombinedEDAResult:
    """EDA全体結果の統合コンテナ"""
    dim_reduction: Optional[DimReductionResult] = None
    feature_importance: Optional[FeatureImportanceResult] = None
    warnings: list[str] = field(default_factory=list)
    
    def to_api_response(self) -> dict:
        """APIレスポンス用変換"""
        return {
            "dim_reduction": self.dim_reduction.to_json_serializable() if self.dim_reduction else None,
            "feature_importance": self.feature_importance.to_json_serializable() if self.feature_importance else None,
            "warnings": self.warnings
        }


# ============================================================
# ユーティリティ関数（型安全な変換）
# ============================================================

def _convert_to_native(obj: Any) -> Any:
    """numpy/scipy型 → Pythonネイティブ型へ再帰的変換"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj


def _validate_numeric_df(df: pd.DataFrame, min_samples: int = 5, min_features: int = 2) -> tuple[bool, str]:
    """数値DataFrameの事前検証"""
    if df.empty:
        return False, "データが空です"
    if len(df) < min_samples:
        return False, f"サンプル数が不足しています（最低{min_samples}行必要）"
    if df.shape[1] < min_features:
        return False, f"特徴量数が不足しています（最低{min_features}列必要）"
    if df.isnull().any().any():
        return False, "欠損値が含まれています（前処理で除去してください）"
    return True, ""


# ============================================================
# 【再構築】次元削減エンジン
# ============================================================

def compute_dimensionality_reduction(
    df: pd.DataFrame,
    method: ReductionMethod = ReductionMethod.PCA,
    n_components: int = 2,
    scale: bool = True,
    random_state: int = 42,
    **kwargs
) -> DimReductionResult:
    """
    次元削減を計算（フルスクラッチ実装）
    """
    try:
        is_valid, msg = _validate_numeric_df(df)
        if not is_valid:
            return DimReductionResult(
                status="skip", method=method.value, coordinates={}, error_message=msg
            )
        
        X = df.values.astype(np.float64)
        feature_names = df.columns.tolist()
        
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        if method == ReductionMethod.PCA:
            pca = PCA(n_components=n_components, random_state=random_state)
            coords = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_.tolist()
            metadata = {
                "feature_names": feature_names,
                "components": _convert_to_native(pca.components_[:n_components].tolist()),
                "n_features": len(feature_names),
                "n_samples": len(df)
            }
            
        elif method == ReductionMethod.TSNE:
            perplexity = kwargs.get("perplexity", 30.0)
            n_samples = len(df)
            perplexity = max(5.0, min(perplexity, (n_samples - 1) / 3.0))
            
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                init="pca",
                max_iter=kwargs.get("max_iter", 1000),
                learning_rate="auto"
            )
            coords = tsne.fit_transform(X)
            explained_var = None
            metadata = {
                "feature_names": feature_names,
                "perplexity": perplexity,
                "n_features": len(feature_names),
                "n_samples": n_samples
            }
        else:
            return DimReductionResult(
                status="error", method=method.value, coordinates={},
                error_message=f"未サポートの手法: {method.value}"
            )
        
        coord_dict = {
            str(idx): _convert_to_native(coords[i].tolist())
            for i, idx in enumerate(df.index)
        }
        
        return DimReductionResult(
            status="success",
            method=method.value,
            coordinates=coord_dict,
            explained_variance=explained_var,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"[DimReduction] 計算エラー: {e}", exc_info=True)
        return DimReductionResult(
            status="error", method=method.value, coordinates={},
            error_message=f"内部エラー: {str(e)}"
        )


# ============================================================
# 【再構築】特徴量重要度エンジン
# ============================================================

def compute_feature_importance(
    df: pd.DataFrame,
    reduction_result: DimReductionResult,
    metric: ImportanceMetric = ImportanceMetric.PCA_LOADING,
    top_n: int = 20
) -> FeatureImportanceResult:
    """
    特徴量重要度を計算（次元削減結果と連動）
    """
    try:
        if reduction_result.status != "success":
            return FeatureImportanceResult(
                status="skip", metric=metric.value, importance={},
                error_message=f"次元削減が未成功: {reduction_result.status}"
            )
        
        feature_names = reduction_result.metadata.get("feature_names", df.columns.tolist())
        
        if metric == ImportanceMetric.PCA_LOADING:
            components = reduction_result.metadata.get("components")
            if not components:
                return FeatureImportanceResult(
                    status="error", metric=metric.value, importance={},
                    error_message="PCA成分情報が不足しています"
                )
            
            importance = {}
            for i, fname in enumerate(feature_names):
                loadings = [abs(components[0][i]), abs(components[1][i])]
                importance[fname] = float(np.mean(loadings))
                
        elif metric == ImportanceMetric.TSNE_CORR:
            coords = np.array([
                reduction_result.coordinates[str(idx)] 
                for idx in df.index
            ])
            
            importance = {}
            for col in df.columns:
                # 定数列チェック：相関を計算せず0を返す
                if df[col].nunique() <= 1:
                    importance[col] = 0.0
                    continue
                # t-SNE座標も定数チェック
                tsne1 = pd.Series(coords[:, 0])
                tsne2 = pd.Series(coords[:, 1])
                if tsne1.nunique() <= 1 or tsne2.nunique() <= 1:
                    importance[col] = 0.0
                    continue
                # 警告を一時的に抑制して相関計算
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    corr1 = df[col].rank().corr(tsne1, method="spearman")
                    corr2 = df[col].rank().corr(tsne2, method="spearman")
                importance[col] = float(np.sqrt(corr1**2 + corr2**2)) if not (np.isnan(corr1) or np.isnan(corr2)) else 0.0
                
        else:
            return FeatureImportanceResult(
                status="error", metric=metric.value, importance={},
                error_message=f"未サポートの重要度指標: {metric.value}"
            )
        
        if importance:
            max_val = max(importance.values())
            if max_val > 0:
                importance = {k: v / max_val for k, v in importance.items()}
        
        return FeatureImportanceResult(
            status="success",
            metric=metric.value,
            importance=importance,
            top_n=top_n,
            metadata={"n_features": len(importance)}
        )
        
    except Exception as e:
        logger.error(f"[FeatureImportance] 計算エラー: {e}", exc_info=True)
        return FeatureImportanceResult(
            status="error", metric=metric.value, importance={},
            error_message=f"内部エラー: {str(e)}"
        )


# ============================================================
# 【統合API】ワンコールで全結果取得
# ============================================================

def run_dim_reduction_with_importance(
    df: pd.DataFrame,
    method: str = "pca",
    scale: bool = True,
    top_n_importance: int = 20,
    **kwargs
) -> CombinedEDAResult:
    warnings = []
    
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        warnings.append("数値列が見つかりませんでした")
        return CombinedEDAResult(warnings=warnings)
    
    reduction_method = ReductionMethod(method.lower())
    dim_result = compute_dimensionality_reduction(
        numeric_df, method=reduction_method, scale=scale, **kwargs
    )
    
    if dim_result.status != "success":
        warnings.append(f"次元削減: {dim_result.error_message}")
        return CombinedEDAResult(dim_reduction=dim_result, warnings=warnings)
    
    importance_metric = (
        ImportanceMetric.PCA_LOADING if reduction_method == ReductionMethod.PCA
        else ImportanceMetric.TSNE_CORR
    )
    importance_result = compute_feature_importance(
        numeric_df, dim_result, metric=importance_metric, top_n=top_n_importance
    )
    
    if importance_result.status != "success":
        warnings.append(f"重要度計算: {importance_result.error_message}")
    
    return CombinedEDAResult(
        dim_reduction=dim_result,
        feature_importance=importance_result,
        warnings=warnings
    )


def compute_dim_reduction_and_importance_legacy(df: pd.DataFrame) -> dict:
    result = run_dim_reduction_with_importance(df, method="pca")
    
    if result.dim_reduction and result.dim_reduction.status == "success":
        coords_df = pd.DataFrame({
            k: v for k, v in result.dim_reduction.coordinates.items()
        }).T
        coords_df.columns = ["PC1", "PC2"] if result.dim_reduction.method == "pca" else ["t-SNE1", "t-SNE2"]
        
        importance_df = pd.DataFrame(
            list(result.feature_importance.importance.items()) 
            if result.feature_importance and result.feature_importance.status == "success"
            else [],
            columns=["feature", "importance"]
        )
        
        return {
            "status": "success",
            "pca_coords": coords_df if result.dim_reduction.method == "pca" else None,
            "tsne_coords": coords_df if result.dim_reduction.method == "tsne" else None,
            "pca_importance": importance_df if result.dim_reduction.method == "pca" else None,
            "tsne_importance": importance_df if result.dim_reduction.method == "tsne" else None,
            "explained_var": result.dim_reduction.explained_variance,
            "n_features": result.dim_reduction.metadata.get("n_features", 0),
            "n_samples": result.dim_reduction.metadata.get("n_samples", 0),
            "warnings": result.warnings
        }
    else:
        return {
            "status": result.dim_reduction.status if result.dim_reduction else "error",
            "message": result.dim_reduction.error_message if result.dim_reduction else "不明なエラー",
            "warnings": result.warnings
        }
