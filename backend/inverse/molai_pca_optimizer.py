"""
MolAI + PCA 逆解析エンジン (Retrieval-based)

設計思想:
1. 最適化は低次元のPCA空間（潜在空間）で行う。
2. 最適化後のベクトルは記述子空間へ復元。
3. 復元された記述子に最も近い「実在する分子」を検索（Retrieval）。
   → これにより、化学的に無効な構造が生成されるリスクを完全に排除する。
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MolAIPCAInverseOptimizer:
    """MolAI記述子のPCA空間を用いた逆解析オプティマイザー"""
    
    def __init__(
        self, 
        predictor_model: Any,          # 学習済み予測モデル（Pipeline推奨）
        pca_model: Any,                # 学習済みPCAモデル
        reference_descriptors: np.ndarray, # 参照データセットの記述子
        reference_smiles: List[str]    # 参照データセットのSMILESリスト
    ):
        self.predictor = predictor_model
        self.pca = pca_model
        self.ref_desc = np.asarray(reference_descriptors)
        self.ref_smiles = reference_smiles
        
        # 最近傍探索インデックスの事前構築（KD-Tree等）
        self.nn_search = NearestNeighbors(n_neighbors=1, algorithm='auto')
        self.nn_search.fit(self.ref_desc)
        
        logger.info(f"MolAIPCAInverseOptimizer: {len(reference_smiles)} molecules indexed.")

    def search(
        self, 
        target_value: float, 
        top_k: int = 5,
        max_iter: int = 100
    ) -> List[Dict[str, Any]]:
        """
        目標値に最も近い分子を探索。
        
        Args:
            target_value: 目標物性値
            top_k: 提案候補数
            max_iter: 最適化反復上限
            
        Returns:
            List[Dict]: 提案分子情報
        """
        # ── Step 1: PCA空間での最適化 ──
        # 目的関数: |Predict(InverseTransform(z)) - Target|^2
        def objective(z: np.ndarray) -> float:
            try:
                # 潜在ベクトル z -> 記述子空間 d_rec
                d_rec = self.pca.inverse_transform(z.reshape(1, -1))
                # 予測値 pred
                pred = self.predictor.predict(d_rec)[0]
                return (pred - target_value) ** 2
            except Exception as e:
                logger.warning(f"Optimization step error: {e}")
                return 1e6 # ペナルティ
        
        # 初期値: PCAスコアの平均（通常0）
        z0 = np.zeros(self.pca.n_components_)
        
        # 最適化実行 (L-BFGS-B)
        res = minimize(objective, z0, method='L-BFGS-B', options={'maxiter': max_iter})
        
        # ── Step 2: 最近傍探索 (Retrieval) ──
        # 最適化された潜在ベクトルから記述子を復元
        d_ideal = self.pca.inverse_transform(res.x.reshape(1, -1))
        
        # 実在する分子の中から最も近いものを検索
        distances, indices = self.nn_search.kneighbors(d_ideal, n_neighbors=top_k)
        
        # ── Step 3: 結果整形 ──
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            dist = distances[0][i]
            smiles = self.ref_smiles[idx]
            
            # 念のため、取得した分子の予測値も再計算
            try:
                actual_pred = self.predictor.predict(self.ref_desc[idx].reshape(1, -1))[0]
            except Exception:
                actual_pred = float('nan')
            
            results.append({
                "smiles": smiles,
                "index": int(idx),
                "distance": float(dist),       # 理想記述子との距離（越小越好）
                "predicted_value": float(actual_pred), # モデル予測値
                "optimization_loss": float(res.fun)
            })
            
        return results
