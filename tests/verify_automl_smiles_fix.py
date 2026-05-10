"""
tests/verify_automl_smiles_fix.py

SMILES列のみを含むデータセットでAutoMLが正常に動作することを検証するスクリプト。
"""
import pandas as pd
import numpy as np
from backend.models.automl import AutoMLEngine

def test_automl_smiles_only():
    print("--- SMILES列のみのデータセットでAutoMLを実行 ---")
    
    # 1. テストデータの生成 (SMILES + Target)
    data = {
        'SMILES': [
            'CCO', 'c1ccccc1', 'CCC', 'CC(=O)O', 'C1CCCCC1', 
            'CC', 'CN', 'CC=O', 'C#C', 'CO',
            'CCCC', 'c1ccccc1C', 'CC(C)O', 'NCC(=O)O', 'OC=O'
        ],
        'Target': [
            46.07, 78.11, 44.10, 60.05, 84.16,
            30.07, 31.06, 44.05, 26.04, 32.04,
            58.12, 92.14, 60.10, 75.07, 46.03
        ]
    }
    df = pd.DataFrame(data)
    
    # 2. AutoMLエンジンの初期化
    engine = AutoMLEngine(model_keys=["rf", "lr", "dt"]) # 検証のためモデルを絞る
    
    # 3. 実行
    try:
        print("AutoML.run を開始します...")
        result = engine.run(df, target_col='Target', smiles_col='SMILES')
        
        print("\n--- 実行結果 ---")
        print(f"Task: {result.task}")
        print(f"Best Model: {result.best_model_key}")
        print(f"Best Score (CV): {result.best_score:.4f}")
        print(f"Elapsed Time: {result.elapsed_seconds:.2f}s")
        
        if len(result.warnings) > 0:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")
                
        print("\n[SUCCESS] AutoML successfully processed SMILES and generated a model.")
        
    except Exception as e:
        print(f"\n[FAILURE] AutoML failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_automl_smiles_only()
