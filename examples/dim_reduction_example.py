"""
examples/dim_reduction_example.py

次元削減モジュール (backend.data.dim_reduction) の使用例。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backend.data.dim_reduction import DimReducer, DimReductionConfig

def run_example():
    # 1. テストデータの生成 (3つのクラスタを持つダミーデータ)
    print("--- テストデータの生成 ---")
    np.random.seed(42)
    data1 = np.random.randn(50, 10) + [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    data2 = np.random.randn(50, 10) + [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
    data3 = np.random.randn(50, 10) + [5, -5, 5, -5, 5, -5, 5, -5, 5, -5]
    
    X = np.vstack([data1, data2, data3])
    y = np.array(['Cluster A'] * 50 + ['Cluster B'] * 50 + ['Cluster C'] * 50)
    
    # 2. PCA の実行
    print("\n--- PCA を実行中 ---")
    pca_config = DimReductionConfig(method='pca', n_components=2)
    pca_reducer = DimReducer(pca_config)
    X_pca = pca_reducer.fit_transform(X)
    print(f"PCA 寄与率: {pca_reducer.explained_variance_ratio_}")
    
    # 3. UMAP の実行 (依存ライブラリがあれば)
    print("\n--- UMAP を実行中 ---")
    try:
        umap_config = DimReductionConfig(method='umap', n_components=2, n_neighbors=15)
        umap_reducer = DimReducer(umap_config)
        X_umap = umap_reducer.fit_transform(X)
    except ImportError:
        print("UMAP がインストールされていないためスキップします。")
        X_umap = None

    # 4. 可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PCA Plot
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Target'] = y
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Target', ax=axes[0])
    axes[0].set_title("PCA Results")
    
    # UMAP Plot
    if X_umap is not None:
        umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
        umap_df['Target'] = y
        sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Target', ax=axes[1])
        axes[1].set_title("UMAP Results")
    else:
        axes[1].text(0.5, 0.5, "UMAP not available", ha='center')

    plt.tight_layout()
    plt.show()
    print("\n可視化を完了しました。")

if __name__ == "__main__":
    run_example()
