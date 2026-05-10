"""
階層別メトリック評価エンジン（事前一括計算版）

設計原則:
- 軽量指標（R²/MAE/RMSE/Count）: 解析完了時に全レベル事前計算
- 重量処理（クラスタリング）: オプションで事前計算（デフォルト有効）
- UI 応答: 計算待ちゼロ、描画のみで即時表示
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricRecord:
    """単一グループのメトリックを保持する軽量データクラス"""
    r2: float
    mae: float
    rmse: float
    n: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricRecord":
        return cls(**data)


@dataclass
class StratifiedMetrics:
    """
    全階層のメトリックを事前計算済みで保持。
    JSON シリアライズ対応によりフロントエンドへ直接送信可能。
    """
    global_metrics: MetricRecord
    by_category: Dict[str, Dict[str, MetricRecord]]
    by_cluster: Dict[str, MetricRecord]
    cluster_method: Optional[str] = None
    n_clusters: int = 5
    min_group_size: int = 10
    available_categories: List[str] = field(default_factory=list)
    
    def get_category_columns(self) -> List[str]:
        return self.available_categories
    
    def to_dict(self) -> Dict[str, Any]:
        """フロントエンド送信用シリアライズ"""
        def serialize(obj):
            if isinstance(obj, MetricRecord):
                return obj.to_dict()
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [serialize(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj
        return serialize(asdict(self))
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StratifiedMetrics":
        """JSON からの復元"""
        def deserialize(obj):
            if isinstance(obj, dict) and "r2" in obj and "mae" in obj:
                return MetricRecord.from_dict(obj)
            if isinstance(obj, dict):
                return {k: deserialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [deserialize(v) for v in obj]
            return obj
        deserialized = deserialize(data)
        return cls(**deserialized)


class StratifiedMetricCalculator:
    """
    軽量指標を事前一括計算するエンジン。
    
    計算コスト見積もり（典型ケース n=1,000）:
    - Global: <0.1ms
    - Category (5列×10グループ): ~2ms
    - Cluster (KMeans, k=5): ~50ms
    - 合計: <100ms（解析全体に比して無視可能）
    """
    
    def __init__(self, min_group_size: int = 10):
        self.min_group_size = min_group_size
    
    def compute_all(
        self,
        y_true: Union[np.ndarray, List[float]],
        y_pred: Union[np.ndarray, List[float]],
        metadata: Optional[pd.DataFrame] = None,
        auto_cluster: bool = True,
        n_clusters: int = 5,
        cluster_method: str = "kmeans"
    ) -> StratifiedMetrics:
        """
        全階層のメトリックを一括計算。
        
        Args:
            y_true: 実測値配列
            y_pred: 予測値配列
            metadata: 補助データ（カテゴリ列・特徴量を含むDataFrame）
            auto_cluster: 自動クラスタリングを実行するか
            n_clusters: クラスタ数
            cluster_method: "kmeans" | "hierarchical"
        
        Returns:
            StratifiedMetrics: 全階層の計算結果を保持するオブジェクト
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true と y_pred の長さが一致しません: {len(y_true)} vs {len(y_pred)}")
        
        # ── 1. Global 計算（必須） ──
        global_metrics = self._compute_record(y_true, y_pred)
        
        # ── 2. Category 計算（利用可能な列を自動検出） ──
        by_category = {}
        available_categories = []
        
        if metadata is not None and not metadata.empty and len(metadata) == len(y_true):
            # 低基数（2〜19値）の列をカテゴリ候補として検出
            candidate_cols = [
                col for col in metadata.columns
                if 2 <= metadata[col].nunique() < 20
            ]
            
            for col in candidate_cols:
                groups = self._compute_by_group(y_true, y_pred, metadata[col].astype(str))
                if groups:
                    by_category[col] = groups
                    available_categories.append(col)
        
        # ── 3. Cluster 計算（オプション） ──
        by_cluster = {}
        cluster_method_used = None
        
        if auto_cluster and metadata is not None and not metadata.empty:
            cluster_labels = self._auto_cluster(
                metadata, n_clusters, cluster_method, 
                min_samples=max(self.min_group_size, len(y_true) // (n_clusters * 2))
            )
            
            if cluster_labels is not None:
                groups = self._compute_by_group(y_true, y_pred, pd.Series(cluster_labels, index=metadata.index).astype(str))
                if groups:
                    by_cluster = groups
                    cluster_method_used = cluster_method
        
        return StratifiedMetrics(
            global_metrics=global_metrics,
            by_category=by_category,
            by_cluster=by_cluster,
            cluster_method=cluster_method_used,
            n_clusters=n_clusters,
            min_group_size=self.min_group_size,
            available_categories=available_categories
        )
    
    def _compute_record(self, y_true: np.ndarray, y_pred: np.ndarray) -> MetricRecord:
        """単一グループの指標を計算（O(n)・極めて軽量）"""
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = float('nan')
        
        return MetricRecord(
            r2=float(r2) if np.isfinite(r2) else float('nan'),
            mae=float(mean_absolute_error(y_true, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            n=int(len(y_true))
        )
    
    def _compute_by_group(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        groups: pd.Series
    ) -> Dict[str, MetricRecord]:
        """グループ別メトリックを一括計算"""
        result = {}
        
        for name, idx in groups.groupby(groups).groups.items():
            idx_arr = np.asarray(idx)
            if len(idx_arr) < self.min_group_size:
                continue
            
            yt = y_true[idx_arr]
            yp = y_pred[idx_arr]
            result[str(name)] = self._compute_record(yt, yp)
        
        return result
    
    def _auto_cluster(
        self,
        metadata: pd.DataFrame,
        n_clusters: int,
        method: str,
        min_samples: int = 20
    ) -> Optional[np.ndarray]:
        """
        自動クラスタリング実行（O(n×k×iter)・中量計算）
        
        Returns:
            cluster labels array or None if clustering is not possible
        """
        if len(metadata) < n_clusters * min_samples:
            logger.debug(f"Clustering skipped: insufficient samples ({len(metadata)} < {n_clusters * min_samples})")
            return None
        
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("sklearn clustering not available; skipping auto-cluster")
            return None
        
        try:
            # 数値特徴量を抽出
            numeric_cols = [
                c for c in metadata.columns 
                if pd.api.types.is_numeric_dtype(metadata[c])
            ]
            
            if len(numeric_cols) < 2:
                logger.debug("Clustering skipped: insufficient numeric features")
                return None
            
            X = metadata[numeric_cols].values
            
            # 欠損値がある行を除外
            mask = ~np.any(np.isnan(X), axis=1)
            if mask.sum() < n_clusters * min_samples:
                logger.debug("Clustering skipped: too many missing values")
                return None
            
            X_clean = X[mask]
            original_idx = np.where(mask)[0]
            
            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # クラスタリング実行
            if method == "kmeans":
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            else:  # hierarchical
                model = AgglomerativeClustering(n_clusters=n_clusters)
            
            labels_clean = model.fit_predict(X_scaled)
            
            # 元のインデックスにマッピング（未使用行は -1）
            labels = np.full(len(metadata), -1, dtype=int)
            labels[original_idx] = labels_clean
            
            return labels
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return None
