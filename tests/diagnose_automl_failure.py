"""
tests/diagnose_automl_failure.py

AutoMLの各モデル失敗の原因を詳しく調査するスクリプト。
"""
import sys
import os
import traceback
import logging

# ルートパスを通す
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

# ログを詳しく表示
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s [%(name)s] %(message)s"
)

print("=" * 60)
print("AutoML 診断スクリプト - Step 1: サンプルデータ作成")
print("=" * 60)

# sample_smiles.csv を読む（実際のデータ）
sample_csv = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_smiles.csv")
if os.path.exists(sample_csv):
    df = pd.read_csv(sample_csv)
    print(f"CSVロード完了: {df.shape}, 列={df.columns.tolist()}")
else:
    print(f"CSVが見つかりません: {sample_csv} → ダミーデータを使用")
    df = pd.DataFrame({
        "smiles": ["CCO", "CCC", "CCCC", "c1ccccc1", "CCN", "CO", "CC=O", "CC(=O)O", "CNC", "CCCO"],
        "solubility": [1.2, 1.3, 1.5, 0.5, 1.1, 2.0, 1.8, 1.9, 1.2, 1.4],
    })

print(f"\n--- データ概要 ---\n{df.dtypes}")
print(df.head(3))

print("\n" + "=" * 60)
print("Step 2: SMILES → 記述子変換テスト")
print("=" * 60)
try:
    from backend.chem.rdkit_adapter import RDKitAdapter
    adapter = RDKitAdapter(compute_fp=True)
    print(f"RDKit利用可能: {adapter.is_available()}")
    smiles_list = df.iloc[:, 0].tolist()[:3]
    res = adapter.compute(smiles_list)
    print(f"記述子shape: {res.descriptors.shape}")
    print(res.descriptors.head(2))
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Step 3: build_full_pipeline テスト（1モデル）")
print("=" * 60)
try:
    from backend.chem.rdkit_adapter import RDKitAdapter
    from backend.data.type_detector import TypeDetector
    from backend.data.preprocessor import PreprocessConfig, build_full_pipeline
    from backend.models.factory import get_model
    from backend.models.cv_manager import CVConfig, run_cross_validation

    target_col = df.columns[-1]
    smiles_col = df.columns[0]

    # SMILES → 記述子
    adapter = RDKitAdapter(compute_fp=True)
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    if adapter.is_available():
        desc_res = adapter.compute(X[smiles_col].tolist())
        X_chem = desc_res.descriptors
        X = X.drop(columns=[smiles_col])
        X = pd.concat([X.reset_index(drop=True), X_chem.reset_index(drop=True)], axis=1)
        print(f"X shape (after SMILES): {X.shape}")
        print(f"X columns (first 5): {X.columns.tolist()[:5]}")
    
    # 変数型判定
    detector = TypeDetector()
    detection_result = detector.detect(X)
    print(f"detection_result: {detection_result}")

    # パイプライン構築
    model_inst = get_model("lasso", task="regression")
    preprocess_cfg = PreprocessConfig()
    
    print("build_full_pipeline 実行中...")
    pipeline = build_full_pipeline(
        detection_result, model_inst,
        target_col=target_col,
        config=preprocess_cfg,
    )
    print(f"Pipeline 構築OK: {pipeline.steps}")

    # CV実行
    print("\nCross-validation 実行中...")
    cv_config = CVConfig(cv_key="kfold", n_splits=3)
    result = run_cross_validation(pipeline, X, y, cv_config, scoring="neg_root_mean_squared_error", n_jobs=1)
    print(f"CV OK: mean_test_score = {result.get('mean_test_score')}")

except Exception as e:
    print(f"\n--- ERROR ---")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Step 4: AutoMLEngine.run フルテスト")
print("=" * 60)
try:
    from backend.models.automl import AutoMLEngine

    def cb(s, t, m): print(f"  [{s}/{t}] {m}")
    engine = AutoMLEngine(cv_folds=3, model_keys=["ridge", "rf"], progress_callback=cb)
    
    target_col = df.columns[-1]
    smiles_col = df.columns[0]
    
    automl_res = engine.run(df, target_col=target_col, smiles_col=smiles_col)
    print(f"\n【SUCCESS】 Best: {automl_res.best_model_key} / score: {automl_res.best_score:.4f}")
    print(f"Warnings: {automl_res.warnings}")

except Exception as e:
    print(f"\n--- FATAL ERROR ---")
    traceback.print_exc()
