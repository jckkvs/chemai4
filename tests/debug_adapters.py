"""
各アダプタを個別にテストして、エラーを特定するデバッグスクリプト。
使用: python tests/debug_adapters.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
import traceback
import time

# テスト用SMILES（小さめ）
TEST_SMILES = [
    "C", "CC", "CCC", "CCO", "CCN",
    "c1ccccc1", "c1ccccc1O", "CC(=O)O",
]

# テスト対象アダプタ (モジュール, クラス名, kwargs)
ADAPTERS_TO_TEST = [
    # --- 現在 auto-calc に入っている8エンジン ---
    ("backend.chem.rdkit_adapter",           "RDKitAdapter",           {"compute_fp": False}, "✅ 既追加"),
    ("backend.chem.group_contrib_adapter",   "GroupContribAdapter",     {},                   "✅ 既追加"),
    ("backend.chem.mordred_adapter",         "MordredAdapter",          {"selected_only": True}, "✅ 既追加"),
    ("backend.chem.skfp_adapter",            "SkfpAdapter",             {},                   "✅ 既追加"),
    ("backend.chem.descriptastorus_adapter", "DescriptaStorusAdapter",  {},                   "✅ 既追加"),
    ("backend.chem.molfeat_adapter",         "MolfeatAdapter",          {},                   "✅ 既追加"),
    ("backend.chem.mol2vec_adapter",         "Mol2VecAdapter",          {},                   "✅ 既追加"),
    ("backend.chem.padel_adapter",           "PaDELAdapter",            {},                   "✅ 既追加"),
    # --- 未追加の残り6エンジン ---
    ("backend.chem.molai_adapter",           "MolAIAdapter",            {"n_components": 6},  "🔲 未追加"),
    ("backend.chem.uma_adapter",             "UMAAdapter",              {},                   "🔲 未追加"),
    ("backend.chem.xtb_adapter",             "XTBAdapter",              {},                   "🔲 未追加"),
    ("backend.chem.unipka_adapter",          "UniPkaAdapter",           {},                   "🔲 未追加"),
    ("backend.chem.cosmo_adapter",           "CosmoAdapter",            {},                   "🔲 未追加"),
    ("backend.chem.chemprop_adapter",        "ChempropAdapter",         {},                   "🔲 未追加"),
]


def test_adapter(module_path, class_name, kwargs, status_tag):
    """1アダプタをテストして結果を返す。"""
    print(f"\n{'='*60}")
    print(f"  {status_tag} {class_name}")
    print(f"{'='*60}")
    
    # Step 1: インポート
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        print(f"  [OK] インポート成功")
    except Exception as e:
        print(f"  [FAIL] インポートエラー: {e}")
        traceback.print_exc()
        return "IMPORT_ERROR", str(e)
    
    # Step 2: インスタンス化
    try:
        adapter = cls(**kwargs)
        print(f"  [OK] インスタンス化成功")
    except Exception as e:
        print(f"  [FAIL] インスタンス化エラー: {e}")
        traceback.print_exc()
        return "INIT_ERROR", str(e)
    
    # Step 3: is_available チェック
    try:
        available = adapter.is_available()
        print(f"  [--] is_available() = {available}")
        if not available:
            return "NOT_AVAILABLE", "is_available() = False"
    except Exception as e:
        print(f"  [FAIL] is_available エラー: {e}")
        return "AVAILABILITY_ERROR", str(e)
    
    # Step 4: compute 実行
    try:
        t0 = time.time()
        result = adapter.compute(TEST_SMILES)
        elapsed = time.time() - t0
        df = result.descriptors
        if df is not None:
            print(f"  [OK] compute() 成功: {df.shape} 列, {elapsed:.1f}秒")
            nan_rate = df.isna().mean().mean()
            print(f"       NaN率: {nan_rate:.1%}, dtypes: {df.dtypes.value_counts().to_dict()}")
            return "SUCCESS", f"{df.shape[1]}列, {elapsed:.1f}s"
        else:
            print(f"  [WARN] compute() は None を返しました")
            return "NONE_RESULT", "descriptors is None"
    except Exception as e:
        print(f"  [FAIL] compute() エラー:")
        traceback.print_exc()
        return "COMPUTE_ERROR", str(e)


def main():
    print("\n" + "="*60)
    print("  ChemAI アダプタ診断スクリプト")
    print(f"  テスト分子数: {len(TEST_SMILES)}")
    print("="*60)
    
    results = []
    for args in ADAPTERS_TO_TEST:
        module_path, class_name, kwargs, status_tag = args
        status, detail = test_adapter(module_path, class_name, kwargs, status_tag)
        results.append((status_tag, class_name, status, detail))
    
    # サマリー
    print("\n\n" + "="*60)
    print("  診断結果サマリー")
    print("="*60)
    print(f"  {'状態':4} {'クラス名':30} {'結果':20} 詳細")
    print("  " + "-"*80)
    for tag, cls, status, detail in results:
        icon = "✅" if status == "SUCCESS" else ("⏭️" if status == "NOT_AVAILABLE" else "❌")
        print(f"  {icon} {cls:30} {status:20} {detail[:50]}")


if __name__ == "__main__":
    main()
