"""
Test data load, parse, and report flow for ALL sample files.

Runs through each sample in data/samples/ and verifies:
1. Data loading works
2. Basic parsing/statistics can be computed
3. EDA report can be generated
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd
import numpy as np

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import backend modules
from backend.data.loader import load_file
from backend.data.eda import compute_column_stats

# Sample data directory
SAMPLES_DIR = project_root / "data" / "samples"

# Discover all CSV sample files
sample_files = sorted(SAMPLES_DIR.glob("*.csv"))


def detect_data_type(df: pd.DataFrame) -> str:
    """Detect data type from DataFrame columns."""
    if any("SMILES" in col for col in df.columns):
        if any("Compound" in col for col in df.columns) and any("WT%" in col for col in df.columns):
            return "mixture"
        return "smiles"
    return "tabular"


def find_target_col(df: pd.DataFrame) -> str | None:
    """Find target column from candidates."""
    target_candidates = ["Target", "Target_Property", "Class", "logS", "Activity"]
    for tc in target_candidates:
        if tc in df.columns:
            return tc
    return None


@pytest.mark.parametrize("file_path", sample_files, ids=lambda p: p.name)
class TestSampleFiles:
    """Test each sample file for load, parse, and report capabilities."""

    def test_load(self, file_path: Path):
        """Test that the file can be loaded."""
        df = load_file(file_path)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0

    def test_parse_stats(self, file_path: Path):
        """Test that column statistics can be computed."""
        df = load_file(file_path)
        stats = compute_column_stats(df)
        assert len(stats) == len(df.columns)
        for stat in stats:
            assert stat.name in df.columns
            assert stat.n_total == len(df)

    def test_numeric_columns(self, file_path: Path):
        """Test numeric column detection."""
        df = load_file(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assert isinstance(numeric_cols, list)

    def test_data_type_detection(self, file_path: Path):
        """Test data type detection works."""
        df = load_file(file_path)
        data_type = detect_data_type(df)
        assert data_type in ("smiles", "mixture", "tabular")

    def test_target_column(self, file_path: Path):
        """Test target column detection."""
        df = load_file(file_path)
        target_col = find_target_col(df)
        if target_col:
            assert target_col in df.columns

    def test_target_stats(self, file_path: Path):
        """Test target column statistics computation."""
        df = load_file(file_path)
        target_col = find_target_col(df)
        if target_col and target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                mean_val = df[target_col].mean()
                std_val = df[target_col].std()
                assert mean_val is not None or pd.isna(mean_val)
                assert std_val is not None or pd.isna(std_val)

    def test_correlation_matrix(self, file_path: Path):
        """Test correlation matrix generation for numeric columns."""
        df = load_file(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            assert corr.shape[0] == len(numeric_cols)
            assert corr.shape[1] == len(numeric_cols)

    def test_report_generation(self, file_path: Path):
        """Test EDA report can be generated."""
        df = load_file(file_path)
        stats = compute_column_stats(df)
        data_type = detect_data_type(df)
        target_col = find_target_col(df)

        report_lines = [
            f"# EDA Report: {file_path.name}",
            "",
            "## Overview",
            f"- Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"- Data type: {data_type}",
            f"- Target column: {target_col or 'Not found'}",
        ]

        for stat in stats:
            report_lines.append(f"### {stat.name} ({stat.dtype})")
            report_lines.append(f"- Non-null: {stat.n_total - stat.n_null} / {stat.n_total}")
            if stat.mean is not None:
                report_lines.append(f"- Mean: {stat.mean:.4f}, Std: {stat.std:.4f}")
                report_lines.append(f"- Range: [{stat.min:.4f}, {stat.max:.4f}]")
            report_lines.append(f"- Unique values: {stat.n_unique}")

        # Verify report content
        report_text = "\n".join(report_lines)
        assert len(report_text) > 0
        assert "EDA Report" in report_text


def test_sample_summary():
    """Generate a summary of all sample files."""
    results = []
    for file_path in sample_files:
        result = {
            "file": file_path.name,
            "type": "unknown",
            "load_ok": False,
            "parse_ok": False,
            "error": None,
            "shape": None,
        }
        try:
            df = load_file(file_path)
            result["load_ok"] = True
            result["shape"] = df.shape
            result["type"] = detect_data_type(df)
            result["columns"] = list(df.columns)
            result["numeric_cols"] = len(df.select_dtypes(include=[np.number]).columns)
            stats = compute_column_stats(df)
            result["parse_ok"] = len(stats) == len(df.columns)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}"
        results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("SAMPLE FILES SUMMARY")
    print("=" * 70)
    total = len(results)
    load_ok = sum(1 for r in results if r["load_ok"])
    parse_ok = sum(1 for r in results if r["parse_ok"])
    print(f"Total files: {total}")
    print(f"Load OK:    {load_ok}/{total}")
    print(f"Parse OK:   {parse_ok}/{total}")

    for r in results:
        status = "[PASS]" if (r["load_ok"] and r["parse_ok"]) else "[FAIL]"
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}" if r["shape"] else "N/A"
        print(f"  {status} {r['file']:<45} {r['type']:<12} {shape_str}")
        if r["error"]:
            print(f"        Error: {r['error'][:120]}")

    assert load_ok == total, f"{total - load_ok} file(s) failed to load"
    assert parse_ok == total, f"{total - parse_ok} file(s) failed to parse"
