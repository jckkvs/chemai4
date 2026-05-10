"""
examples/descriptor_calculation_example.py

RDKitやMordredを使って、入力SMILESから化学記述子を独立して計算し、
CSVファイルとして保存するサンプルスクリプト。
"""

import sys
import pandas as pd
from pathlib import Path

# backendパッケージのパスを通すための設定
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from backend.chem.rdkit_adapter import RDKitAdapter
from backend.chem.mordred_adapter import MordredAdapter

def main():
    print("--- 記述子計算サンプル ---")
    
    # 1. 処理する分子のSMILESリスト
    smiles_list = [
        "c1ccccc1",         # ベンゼン
        "CCO",              # エタノール
        "CC(=O)O",          # 酢酸
        "invalid_smiles",   # パース失敗する無効な文字列
        "c1cc(O)ccc1C(=O)O" # サリチル酸
    ]
    
    print(f"\n入力SMILES ({len(smiles_list)}件):")
    for s in smiles_list:
        print(f"  - {s}")
        
    # 2. RDKitを用いた標準的な物理化学・FP記述子の計算
    print("\n[1] RDKitAdapter を使用して計算します...")
    rdkit_adapter = RDKitAdapter(compute_fp=True)
    rdkit_result = rdkit_adapter.compute(smiles_list)
    
    print(f"RDKit記述子計算完了: {rdkit_result.n_descriptors} 種の記述子を計算しました")
    print(f"成功率: {rdkit_result.success_rate * 100:.1f}%")
    if rdkit_result.failed_indices:
        print(f"失敗したSMILESのインデックス: {rdkit_result.failed_indices}")
    
    # 3. Mordredを用いた高度な2Dトポロジカル記述子の計算
    print("\n[2] MordredAdapter を使用して計算します...")
    mordred_adapter = MordredAdapter(selected_only=True)
    if mordred_adapter.is_available():
        mordred_result = mordred_adapter.compute(smiles_list)
        print(f"Mordred記述子計算完了: {mordred_result.n_descriptors} 種の記述子を計算しました")
        print(f"成功率: {mordred_result.success_rate * 100:.1f}%")
        
        # 4. 結果の結合と保存
        # 失敗した分子は NaN または 0.0 で適切に詰められているのでそのまま結合可能
        df_combined = pd.concat([rdkit_result.descriptors, mordred_result.descriptors], axis=1)
        # SIRTのカラム名重複を避けるための処理など（必要に応じて）
        df_combined = df_combined.loc[:,~df_combined.columns.duplicated()]
        
        df_combined.insert(0, "SMILES", smiles_list)
        
        output_path = project_root / "examples" / "calculated_descriptors.csv"
        df_combined.to_csv(output_path, index=False)
        print(f"\n結果を保存しました: {output_path}")
        print(f"出力Shape: {df_combined.shape}")
        
    else:
        print("\n(!注意) mordred ライブラリがインストールされていません。")
        print("Mordred記述子を計算するには: pip install mordred")
        
        # RDKitの結果のみ保存
        df_out = rdkit_result.descriptors.copy()
        df_out.insert(0, "SMILES", smiles_list)
        output_path = project_root / "examples" / "calculated_descriptors.csv"
        df_out.to_csv(output_path, index=False)
        print(f"\nRDKit結果のみ保存しました: {output_path}")

if __name__ == "__main__":
    main()
