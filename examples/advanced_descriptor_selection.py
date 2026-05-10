"""
examples/advanced_descriptor_selection.py

高度な記述子選択機能（メタデータ分類、相関分析、推奨カテゴリ）を
Pythonスクリプトから直接利用する例。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.chem import RDKitAdapter, MordredAdapter
from backend.chem.recommender import get_target_recommendation_by_name

def main():
    # 1. ダミーデータの準備
    smiles_list = ["CCO", "c1ccccc1", "CCC", "CC(=O)O", "CCN", "c1ccccc1O"]
    # 溶解度(logS)を想定した目的変数データ
    y = np.array([-1.2, -3.5, -2.1, -0.5, -1.0, -2.8])
    
    print(f"--- 1. アダプタからのメタデータ取得 ---")
    rdkit = RDKitAdapter()
    mdata = rdkit.get_descriptors_metadata()
    
    # 「数え上げ系(is_count)」のみを抽出
    count_vars = [m for m in mdata if m.is_count]
    print(f"RDKit 数え上げ系変数の例: {[m.name for m in count_vars[:5]]} (全{len(count_vars)}件)")
    for m in count_vars[:3]:
        print(f"  - {m.name}: {m.meaning}")

    print(f"\n--- 2. 推奨記述子の取得 (ターゲット属性に基づく) ---")
    # 「溶解度(solubility)」に適した推奨セットを取得
    rec = get_target_recommendation_by_name("solubility")
    if rec:
        print(f"ターゲットカテゴリ: {rec.category}")
        print(f"要件サマリー: {rec.summary}")
        print(f"推奨記述子: {[d.name for d in rec.descriptors]}")

    print(f"\n--- 3. 相関係数に基づく動的な選択 ---")
    # 記述子を一括計算
    print("記述子を計算中...")
    result = rdkit.compute(smiles_list)
    df_desc = result.descriptors
    
    # 目的変数 y との相関(Pearson絶対値)を算出
    corrs = df_desc.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
    
    print("相関トップ10記述子 (y = solubility想定):")
    print(corrs.head(10))

    # 上位30件を選択する例
    selected_top_30 = corrs.head(30).index.tolist()
    print(f"\n上位30件が選択されました。 (例: {selected_top_30[:5]}...)")

    print(f"\n--- 4. 統合ヒートマップ用データの構築例 ---")
    # 元データにある他の数値列（温度、圧力など）と結合
    df_meta = pd.DataFrame({"temperature": [25, 25, 30, 30, 40, 40]})
    df_combined = pd.concat([df_meta, df_desc[selected_top_30[:5]]], axis=1)
    
    print("統合データフレームの形状:", df_combined.shape)
    print("相関行列の算出完了。")

if __name__ == "__main__":
    main()
