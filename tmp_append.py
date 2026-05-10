import pandas as pd
import numpy as np
import logging

def compute_dim_reduction_and_importance(df: pd.DataFrame) -> dict:
    """PCA/t-SNE座標および特徴量重要度を計算"""
    try:
        # 数値列のみ抽出、欠損値80%超の列は除外
        num_df = df.select_dtypes(include=["number"]).dropna(axis=1, thresh=len(df) * 0.2)
        num_df = num_df.dropna(axis=0)  # 行欠損も除去
        
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
        perp = max(5.0, min(30.0, len(num_df) - 1))
        # max_iter as n_iter was deprecated
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
