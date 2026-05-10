"""
backend/utils/dependency_checker.py

外部バイナリや依存ライブラリが適切にインストールされているかを検証・診断するモジュール。
クロスプラットフォームな環境差吸収や、フロントエンドへの健全性シグナル提供に用いる。
"""
import sys
import os
import shutil
import importlib

def _try_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def _check_cli(command: str) -> bool:
    return shutil.which(command) is not None

def run_doctor() -> dict:
    """
    主要な依存環境の診断レポートを生成して返す。
    
    Returns:
        dict: 各依存要素の生存状況 ("✅" または "❌" など)
    """
    
    # ユーザー設定されたパスを環境変数から取得してチェック
    xtb_available = _check_cli("xtb") or (os.getenv("XTB_PATH") and Path(os.getenv("XTB_PATH")).exists() if os.getenv("XTB_PATH") else False)
    
    # OpenCOSMO もしくは COSMOtherm のチェック
    cosmo_path = os.getenv("COSMO_ENGINE_PATH", "opencosmo")
    cosmo_available = _check_cli(cosmo_path) or (os.path.exists(cosmo_path) if os.path.isabs(cosmo_path) else False)
    
    checks = {
        "python_version_compatible": sys.version_info >= (3, 10),
        "pandas": _try_import("pandas"),
        "scikit_learn": _try_import("sklearn"),
        "rdkit": _try_import("rdkit"),
        "optuna": _try_import("optuna"),
        "xtb_binary": xtb_available,
        "cosmo_binary": cosmo_available
    }
    
    # 表現をフォーマット
    return {k: "✅" if v else "❌" for k, v in checks.items()}

if __name__ == "__main__":
    from pathlib import Path
    print("ChemAI Dependency Doctor:")
    report = run_doctor()
    for k, v in report.items():
        print(f"  {k.ljust(25)} {v}")
