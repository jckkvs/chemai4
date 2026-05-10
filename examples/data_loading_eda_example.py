"""
examples/data_loading_eda_example.py

データの読み込み (loader.py) と探索的データ解析 (eda.py) の使用例。
"""
import os
import pandas as pd
import numpy as np
from backend.data.loader import load_file, save_dataframe
from backend.data.eda import compute_column_stats, compute_correlation, summarize_dataframe
from backend.data.type_detector import TypeDetector

def run_example():
    # 1. サンプルCSVファイルの作成
    print("--- サンプルデータの作成 ---")
    data = {
        'ID': range(1, 11),
        'Target': [10.5, 20.1, 15.3, 25.4, 30.6, 22.1, 18.3, 28.7, 35.0, 12.4],
        'Feature1': np.random.rand(10),
        'Feature2': np.random.rand(10) * 100,
        'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
        'SMILES': ['CCO', 'c1ccccc1', 'CCC', 'CC(=O)O', 'C1CCCCC1', 'CC', 'CN', 'CC=O', 'C#C', 'CO']
    }
    df_sample = pd.DataFrame(data)
    csv_path = "sample_data.csv"
    save_dataframe(df_sample, csv_path)
    print(f"サンプルデータを保存しました: {csv_path}")

    # 2. データのロード
    print("\n--- データのロード ---")
    df = load_file(csv_path)
    print(f"ロードされたデータの形状: {df.shape}")

    # 3. カラム型の自動判定
    print("\n--- カラム型の自動判定 ---")
    detector = TypeDetector()
    result = detector.detect(df)
    for col, info in result.column_info.items():
        print(f"  {col}: {info.col_type.name}")

    # 4. 基本統計量の取得
    print("\n--- 基本統計量 ---")
    stats = compute_column_stats(df)
    for s in stats[:3]: # 最初の3つを表示
        print(f"  Column: {s.name}, Null Rate: {s.null_rate:.2f}, Mean: {s.mean}")

    # 5. データフレームのサマリー
    print("\n--- データフレームサマリー ---")
    summary = summarize_dataframe(df)
    print(summary)

    # 6. 相関行列の計算
    print("\n--- 相関行列 ---")
    # 数値カラムのみ抽出して相関を計算
    numeric_cols = result.get_numeric_columns()
    if numeric_cols:
        corr = compute_correlation(df[numeric_cols])
        print(corr)

    # 7. 後片付け
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"\n一時ファイルを削除しました: {csv_path}")

if __name__ == "__main__":
    run_example()
