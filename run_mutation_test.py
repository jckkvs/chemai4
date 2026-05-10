"""
DomainML Mutation Test Script (v5 - write_cache 方式)
mutant.write_cache() で __pycache__ にバイトコードを書き込み、pytest でテストを実行する。
mutatest の正規の使用方法。
"""
import sys
import subprocess
import shutil
from pathlib import Path
from mutatest.api import Genome
from mutatest.transformers import CATEGORIES
from mutatest.filters import CategoryCodeFilter
import os

TARGETS = [
    ("domainml/constraints/engine.py",      "tests/test_domainml_engine.py"),
    ("domainml/constraints/kernel_opt.py",  "tests/test_domainml_kernel_opt.py"),
    ("domainml/constraints/laplacian.py",   "tests/test_domainml_laplacian.py"),
    ("domainml/analysis/metrics.py",        "tests/test_domainml_analysis.py"),
    ("domainml/analysis/constrained_cv.py", "tests/test_domainml_analysis.py"),
    ("domainml/analysis/uncertainty.py",    "tests/test_domainml_analysis.py"),
]

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def run_test(test_file: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", test_file,
         "-x", "-q", "--tb=no", "--no-header", "--timeout=30"],
        capture_output=True, encoding="utf-8", errors="replace",
        timeout=60, env=ENV
    )
    return result.returncode == 0


def restore_cache(src_path: str) -> None:
    """__pycache__ からミューテーション済みキャッシュを削除して元のバイトコードを再生成"""
    src = Path(src_path)
    cache_dir = src.parent / "__pycache__"
    # 該当ファイルのキャッシュを削除
    for pyc in cache_dir.glob(f"{src.stem}.*.pyc"):
        try:
            pyc.unlink()
        except Exception:
            pass
    # 元のソースからバイトコードを再コンパイル
    subprocess.run(
        [sys.executable, "-c", f"import py_compile; py_compile.compile(r'{src}')"],
        capture_output=True, timeout=10, env=ENV
    )


def run_mutation_for_file(src_path: str, test_path: str) -> dict:
    genome = Genome(source_file=Path(src_path))
    targets = list(genome.targets)
    total_mutants = 0
    killed = 0
    survived = 0
    errors = 0

    print(f"\n{'='*60}", flush=True)
    print(f"Target: {src_path}", flush=True)
    print(f"Test:   {test_path}", flush=True)
    print(f"Mutation targets: {len(targets)}", flush=True)
    print(f"{'='*60}", flush=True)

    for i, target in enumerate(targets):
        try:
            op_code = CATEGORIES.get(target.ast_class)
            if op_code is None:
                continue
            valid_mutations = list(CategoryCodeFilter(codes=(op_code,)).valid_mutations)
            if not valid_mutations:
                continue
        except Exception as e:
            errors += 1
            continue

        for mut_op in valid_mutations:
            total_mutants += 1
            try:
                # ミューテーションを生成して __pycache__ にキャッシュとして書き込む
                mutant = genome.mutate(target, mut_op, write_cache=True)

                # テスト実行（キャッシュのミューテーション済みバイトコードが使われる）
                passed = run_test(test_path)
                if passed:
                    survived += 1
                    op_name = mut_op.__name__ if hasattr(mut_op, '__name__') else str(mut_op)
                    print(f"  SURVIVED: {target.ast_class} L{target.lineno} {op_name}", flush=True)
                else:
                    killed += 1
                    op_name = mut_op.__name__ if hasattr(mut_op, '__name__') else str(mut_op)
                    print(f"  KILLED  : {target.ast_class} L{target.lineno} {op_name}", flush=True)
            except Exception as e:
                errors += 1
                msg = str(e)[:80]
                print(f"  ERROR   : {target.ast_class} L{target.lineno}: {msg}", flush=True)
            finally:
                # キャッシュを元に戻す
                restore_cache(src_path)

    score = killed / total_mutants if total_mutants > 0 else 0.0
    print(f"\n  => {killed}/{total_mutants} killed ({score:.1%}) | survived={survived} | errors={errors}", flush=True)
    return {"file": src_path, "total": total_mutants, "killed": killed,
            "survived": survived, "errors": errors, "score": score}


if __name__ == "__main__":
    all_results = []
    total_killed = 0
    total_mutations = 0

    for src, test in TARGETS:
        if not Path(src).exists():
            print(f"SKIP: {src} not found")
            continue
        if not Path(test).exists():
            print(f"SKIP: test {test} not found")
            continue
        res = run_mutation_for_file(src, test)
        all_results.append(res)
        total_killed += res["killed"]
        total_mutations += res["total"]

    if not all_results:
        print("No results.")
        sys.exit(1)

    print("\n" + "="*60)
    print("OVERALL MUTATION SCORE SUMMARY")
    print("="*60)
    for r in all_results:
        bar = "X" * int(r["score"] * 20)
        print(f"  {r['file']:<45} {r['score']:.1%} [{bar:<20}] ({r['killed']}/{r['total']})")

    overall = total_killed / total_mutations if total_mutations > 0 else 0.0
    print(f"\n  TOTAL: {total_killed}/{total_mutations} killed = {overall:.1%}")
    status = "PASS" if overall >= 0.60 else "FAIL"
    print(f"  {status}: Mutation score {'>=60%' if overall >= 0.60 else '<60%'}")

    with open("mutatest_report.txt", "w", encoding="utf-8") as f:
        f.write("DomainML Mutation Test Report (mutatest write_cache)\n")
        f.write(f"Overall: {total_killed}/{total_mutations} = {overall:.1%}\n\n")
        for r in all_results:
            f.write(f"{r['file']}: {r['killed']}/{r['total']} ({r['score']:.1%})\n")
        f.write(f"\nStatus: {status}\n")
    print("\n  Report written to mutatest_report.txt")
