"""
scripts/test_tracker.py

テスト実行ステータス追跡スクリプト。
各バックエンドモジュールに対して:
  1. 対応するテストファイルが存在するか
  2. モジュール更新後にテストが実行済みか
  3. 未実行のテストのみを効率的に再実行

Usage:
    python scripts/test_tracker.py          # ステータス表示
    python scripts/test_tracker.py --run    # 未実行テストのみ実行
    python scripts/test_tracker.py --all    # 全テスト実行
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
BACKEND_DIR = PROJECT_ROOT / "backend"
TRACKER_FILE = PROJECT_ROOT / ".test_tracker.json"

# ──────────────────────────────────────────────
# モジュール → テストファイルのマッピング定義
# ──────────────────────────────────────────────
MODULE_TEST_MAP: dict[str, list[str]] = {
    # === optim ===
    "backend/optim/bayesian_optimizer.py": [
        "tests/test_bayesian_optimizer_comprehensive.py",
        "tests/test_bayesian_optimizer_extra.py",
        "tests/test_bayesian_opt.py",
    ],
    "backend/optim/bo_visualizer.py": [
        "tests/test_bo_visualizer_comprehensive.py",
        "tests/test_bo_visualizer.py",
    ],
    "backend/optim/constraints.py": [
        "tests/test_optim_comprehensive.py",
        "tests/test_constraints_extra.py",
    ],
    "backend/optim/search_space.py": [
        "tests/test_optim_comprehensive.py",
        "tests/test_search_space_extra.py",
    ],
    "backend/optim/inverse_optimizer.py": [
        "tests/test_inverse_optimizer.py",
    ],
    # === models ===
    "backend/models/factory.py": [
        "tests/test_factory_comprehensive.py",
        "tests/test_factory.py",
        "tests/test_factory_extra.py",
    ],
    "backend/models/rgf.py": [
        "tests/test_rgf_comprehensive.py",
        "tests/test_rgf_extra.py",
    ],
    "backend/models/tuner.py": [
        "tests/test_tuner_comprehensive.py",
        "tests/test_tuner.py",
        "tests/test_tuner_extra.py",
    ],
    "backend/models/monotonic_kernel.py": [
        "tests/test_monotonic_kernel_comprehensive.py",
        "tests/test_monotonic_kernel.py",
        "tests/test_monotonic_kernel_extra.py",
    ],
    "backend/models/linear_tree.py": [
        "tests/test_linear_tree_extra.py",
        "tests/test_linear_tree_rgf_monotonic.py",
    ],
    "backend/models/cv_manager.py": [
        "tests/test_cv_manager_comprehensive.py",
        "tests/test_cv_manager_extra.py",
        "tests/test_cv_manager_new.py",
        "tests/test_cv_walkforward.py",
    ],
    "backend/models/cv_bias_evaluator.py": [
        "tests/test_cv_bias_evaluator.py",
        "tests/test_cv_bias_evaluator_extra.py",
    ],
    "backend/models/automl.py": [
        "tests/test_automl_integration.py",
        "tests/test_automl_extra.py",
    ],
    # === pipeline ===
    "backend/pipeline/col_preprocessor.py": [
        "tests/test_col_preprocessor_comprehensive.py",
        "tests/test_col_preprocessor_extra.py",
    ],
    "backend/pipeline/column_selector.py": [
        "tests/test_pipeline_comprehensive.py",
        "tests/test_column_selector_extra.py",
    ],
    "backend/pipeline/feature_generator.py": [
        "tests/test_base_and_feature_gen.py",
        "tests/test_feature_generator_extra.py",
    ],
    "backend/pipeline/feature_selector.py": [
        "tests/test_feature_selector_comprehensive.py",
        "tests/test_feature_selector_extra.py",
    ],
    "backend/pipeline/pipeline_builder.py": [
        "tests/test_pipeline_comprehensive.py",
        "tests/test_pipeline_builder_extra.py",
    ],
    "backend/pipeline/pipeline_grid.py": [
        "tests/test_pipeline_grid_comprehensive.py",
        "tests/test_pipeline_grid.py",
        "tests/test_pipeline_grid_extra.py",
    ],
    # === data ===
    "backend/data/preprocessor.py": [
        "tests/test_preprocessor_comprehensive.py",
        "tests/test_preprocessor_extra.py",
    ],
    "backend/data/type_detector.py": [
        "tests/test_type_detector_comprehensive.py",
        "tests/test_type_detector_extra.py",
    ],
    "backend/data/data_cleaner.py": [
        "tests/test_data_cleaner.py",
        "tests/test_data_cleaner_extra.py",
    ],
    "backend/data/leakage_detector.py": [
        "tests/test_leakage_detector.py",
        "tests/test_leakage_detector_extra.py",
    ],
    "backend/data/dim_reduction.py": [
        "tests/test_dim_reduction.py",
        "tests/test_dim_reduction_extra.py",
    ],
    "backend/data/eda.py": [
        "tests/test_eda.py",
        "tests/test_eda_extra.py",
    ],
    "backend/data/loader.py": [
        "tests/test_data.py",
        "tests/test_loader_extra.py",
    ],
    "backend/data/feature_engineer.py": [
        "tests/test_feature_engineer.py",
        "tests/test_feature_engineer_ext.py",
        "tests/test_feature_engineer_extra.py",
    ],
    "backend/data/benchmark.py": [
        "tests/test_benchmark.py",
        "tests/test_benchmark_extra.py",
    ],
    "backend/data/benchmark_datasets.py": [
        "tests/test_benchmark_datasets_extra.py",
    ],
    # === chem ===
    "backend/chem/base.py": [
        "tests/test_base_and_feature_gen.py",
        "tests/test_chem_base_extra.py",
    ],
    "backend/chem/recommender.py": [
        "tests/test_recommender_comprehensive.py",
        "tests/test_recommender.py",
    ],
    "backend/chem/charge_config.py": [
        "tests/test_charge_config.py",
        "tests/test_charge_config_extra.py",
        "tests/test_charge_extended.py",
    ],
    "backend/chem/protonation.py": [
        "tests/test_protonation.py",
    ],
    "backend/chem/cosmo_adapter.py": [
        "tests/test_cosmo_adapter.py",
    ],
    "backend/chem/rdkit_adapter.py": [
        "tests/test_chem.py",
    ],
    "backend/chem/descriptor_sets.py": [
        "tests/test_chem.py",
    ],
    # === ui ===
    "backend/ui/param_schema.py": [
        "tests/test_param_schema_comprehensive.py",
        "tests/test_param_schema.py",
        "tests/test_param_schema_extra.py",
    ],
    # === utils ===
    "backend/utils/config.py": [
        "tests/test_config_extra.py",
    ],
    "backend/utils/optional_import.py": [
        "tests/test_optional_import_extra.py",
    ],
    # === interpret ===
    "backend/interpret/shap_explainer.py": [
        "tests/test_interpret.py",
        "tests/test_shap_explainer_extra.py",
    ],
    "backend/interpret/sri.py": [
        "tests/test_sri.py",
        "tests/test_sri_extra.py",
    ],
    # === mlops ===
    "backend/mlops/mlflow_manager.py": [
        "tests/test_mlops.py",
        "tests/test_mlops_extended.py",
    ],
}


def _mtime(path: Path) -> float:
    """ファイルの最終更新時刻を返す。存在しなければ0。"""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _load_tracker() -> dict:
    """前回のテスト実行記録を読み込む。"""
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_tracker(data: dict) -> None:
    """テスト実行記録を保存する。"""
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_status_matrix() -> list[dict]:
    """
    全モジュールのステータスマトリクスを返す。

    Returns:
        [{
            "module": str,
            "tests": [str],
            "module_mtime": str,
            "latest_test_mtime": str,
            "status": "済" | "未実行" | "テストなし",
            "reason": str,
        }]
    """
    tracker = _load_tracker()
    results = []

    for module_rel, test_files in sorted(MODULE_TEST_MAP.items()):
        module_path = PROJECT_ROOT / module_rel
        module_mt = _mtime(module_path)
        module_time = datetime.fromtimestamp(module_mt).strftime("%m/%d %H:%M") if module_mt else "N/A"

        existing_tests = [t for t in test_files if (PROJECT_ROOT / t).exists()]

        if not existing_tests:
            results.append({
                "module": module_rel,
                "tests": test_files,
                "module_mtime": module_time,
                "latest_test_mtime": "N/A",
                "status": "NO_TEST",
                "reason": "No test files found",
            })
            continue

        # テストファイルの最新更新日時
        latest_test_mt = max(_mtime(PROJECT_ROOT / t) for t in existing_tests)
        test_time = datetime.fromtimestamp(latest_test_mt).strftime("%m/%d %H:%M")

        # 前回テスト実行記録の確認
        last_run = tracker.get(module_rel, {}).get("last_run", 0)

        if last_run >= module_mt and last_run >= latest_test_mt:
            status = "DONE"
            reason = f"Last run: {datetime.fromtimestamp(last_run).strftime('%m/%d %H:%M')}"
        elif module_mt > latest_test_mt:
            status = "PENDING"
            reason = "Module modified after tests"
        elif last_run == 0:
            status = "PENDING"
            reason = "Never run"
        else:
            status = "PENDING"
            reason = "Test file updated"

        results.append({
            "module": module_rel,
            "tests": existing_tests,
            "module_mtime": module_time,
            "latest_test_mtime": test_time,
            "status": status,
            "reason": reason,
        })

    return results


def print_status_table(matrix: list[dict]) -> None:
    """ステータスマトリクスをテーブルで表示する。"""
    done = sum(1 for m in matrix if m["status"] == "DONE")
    pending = sum(1 for m in matrix if m["status"] == "PENDING")
    no_test = sum(1 for m in matrix if m["status"] == "NO_TEST")

    print("\n" + "=" * 100)
    print("  TEST EXECUTION STATUS MATRIX")
    print("=" * 100)
    print(f"  {'STATUS':<10} {'MODULE':<42} {'MOD_TIME':<14} {'TEST_TIME':<14} {'REASON'}")
    print("-" * 100)

    for m in matrix:
        mod_short = m["module"].replace("backend/", "")
        flag = "[OK]  " if m["status"] == "DONE" else ("[RUN] " if m["status"] == "PENDING" else "[---] ")
        print(f"  {flag} {mod_short:<42} {m['module_mtime']:<14} {m['latest_test_mtime']:<14} {m['reason']}")

    print("-" * 100)
    print(f"  DONE: {done} | PENDING: {pending} | NO_TEST: {no_test} | TOTAL: {len(matrix)}")
    print("=" * 100)


def run_pending_tests(matrix: list[dict], verbose: bool = True) -> int:
    """未実行テストのみ実行して記録を更新する。
    
    失敗テストがあっても、パスしたモジュールの記録は更新する。
    """
    pending = [m for m in matrix if m["status"] == "PENDING"]

    if not pending:
        print("\n[OK] All tests are up-to-date. Nothing to run.")
        return 0

    # テストファイルを重複なく収集
    test_files: set[str] = set()
    for m in pending:
        for t in m["tests"]:
            if (PROJECT_ROOT / t).exists():
                test_files.add(t)

    print(f"\n[RUN] Running {len(pending)} modules / {len(test_files)} test files...")

    # junit XMLで結果を取得
    junit_path = PROJECT_ROOT / ".test_results.xml"
    test_paths = [str(PROJECT_ROOT / t) for t in sorted(test_files)]
    cmd = [sys.executable, "-m", "pytest"] + test_paths
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "--no-header", f"--junitxml={junit_path}"])

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    # 結果を解析して、失敗が含まれるテストファイルを特定
    failed_files: set[str] = set()
    if junit_path.exists():
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(junit_path)
            for tc in tree.iter("testcase"):
                if tc.find("failure") is not None or tc.find("error") is not None:
                    classname = tc.get("classname", "")
                    # classname から相対パスを推定
                    parts = classname.split(".")
                    if parts:
                        failed_files.add(f"tests/{parts[0]}.py")
        except Exception:
            pass

    # パスしたモジュールの記録を更新
    now = datetime.now().timestamp()
    tracker = _load_tracker()
    updated = 0
    for m in pending:
        module_tests = set(m["tests"])
        if module_tests & failed_files:
            continue  # 失敗テストを含むモジュールはスキップ
        tracker[m["module"]] = {"last_run": now, "result": "pass"}
        updated += 1

    _save_tracker(tracker)

    if result.returncode == 0:
        print(f"\n[OK] All tests passed. Updated {updated} module records.")
    else:
        failed_count = len(pending) - updated
        print(f"\n[PARTIAL] {updated} modules PASSED, {failed_count} modules had failures.")
        if failed_files:
            print("  Failed test files:")
            for f in sorted(failed_files):
                print(f"    - {f}")

    return result.returncode


def mark_all_done() -> None:
    """全モジュールを'済'にマークする（手動テスト実行後に使用）。"""
    now = datetime.now().timestamp()
    tracker = _load_tracker()
    for module_rel in MODULE_TEST_MAP:
        tracker[module_rel] = {"last_run": now, "result": "pass"}
    _save_tracker(tracker)
    print(f"[OK] Marked {len(MODULE_TEST_MAP)} modules as DONE.")


def main():
    parser = argparse.ArgumentParser(description="テスト実行ステータス追跡")
    parser.add_argument("--run", action="store_true", help="未実行テストのみ実行")
    parser.add_argument("--all", action="store_true", help="全テスト実行")
    parser.add_argument("--mark-done", action="store_true", help="全モジュールを'済'にマーク")
    parser.add_argument("--quiet", "-q", action="store_true", help="簡潔な出力")
    args = parser.parse_args()

    if args.mark_done:
        mark_all_done()
        return

    matrix = get_status_matrix()

    if not args.quiet:
        print_status_table(matrix)

    if args.all:
        # 全モジュールを未実行にリセットして実行
        for m in matrix:
            if m["status"] != "NO_TEST":
                m["status"] = "PENDING"
        return run_pending_tests(matrix)
    elif args.run:
        return run_pending_tests(matrix)


if __name__ == "__main__":
    sys.exit(main() or 0)
