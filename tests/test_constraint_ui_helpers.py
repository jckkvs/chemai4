"""
tests/test_constraint_ui_helpers.py

constraint_ui_helpers.py のユニットテスト。
- UI制約 → バックエンド Constraint 変換
- 制約検証（競合検出・充足率）
- テンプレート IO（保存・読込・削除）
- 自然言語プレビュー
- エラーメッセージ変換
- データ統計ヘルパー
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from frontend_nicegui.components.constraint_ui_helpers import (
    CONSTRAINT_TYPES,
    SUM_PRESETS,
    RATIO_OPERATORS,
    ERROR_MESSAGES,
    ui_constraints_to_backend,
    validate_constraints,
    describe_constraint,
    save_template,
    load_template,
    list_templates,
    delete_template,
    friendly_error,
    get_column_stats,
)
from backend.optim.constraints import (
    RangeConstraint,
    SumConstraint,
    InequalityConstraint,
    AtLeastNConstraint,
    CustomConstraint,
)


# ═══════════════════════════════════════════════════════════
# 定数の整合性テスト
# ═══════════════════════════════════════════════════════════
class TestConstants:
    def test_constraint_types_has_six_entries(self):
        assert len(CONSTRAINT_TYPES) == 6

    def test_constraint_types_keys(self):
        keys = {ct["key"] for ct in CONSTRAINT_TYPES}
        assert keys == {"range", "sum", "ratio", "exclusion", "conditional", "formula"}

    def test_simple_types_are_range_and_sum(self):
        simple = [ct["key"] for ct in CONSTRAINT_TYPES if ct.get("simple")]
        assert set(simple) == {"range", "sum"}

    def test_sum_presets_structure(self):
        assert len(SUM_PRESETS) >= 3
        for p in SUM_PRESETS:
            assert "label" in p
            assert "target" in p
            assert "tolerance" in p

    def test_ratio_operators_structure(self):
        assert "ge" in RATIO_OPERATORS
        assert "le" in RATIO_OPERATORS
        for key, val in RATIO_OPERATORS.items():
            assert "label" in val
            assert "symbol" in val


# ═══════════════════════════════════════════════════════════
# UI → バックエンド変換テスト
# ═══════════════════════════════════════════════════════════
class TestUIToBackendConversion:
    def test_convert_range_constraint(self):
        items = [{"type": "range", "column": "A", "lo": 10.0, "hi": 50.0}]
        result = ui_constraints_to_backend(items)
        assert len(result) == 1
        assert isinstance(result[0], RangeConstraint)
        assert result[0].column == "A"
        assert result[0].lo == 10.0
        assert result[0].hi == 50.0

    def test_convert_sum_constraint(self):
        items = [{"type": "sum", "columns": ["A", "B", "C"], "target": 100.0, "tolerance": 0.1}]
        result = ui_constraints_to_backend(items)
        assert len(result) == 1
        assert isinstance(result[0], SumConstraint)
        assert result[0].columns == ["A", "B", "C"]
        assert result[0].target == 100.0

    def test_convert_ratio_constraint(self):
        items = [{
            "type": "ratio",
            "target_var": "A",
            "base_var": "B",
            "ratio": 2.0,
            "operator": "ge",
        }]
        result = ui_constraints_to_backend(items)
        assert len(result) == 1
        assert isinstance(result[0], InequalityConstraint)

    def test_convert_exclusion_constraint(self):
        items = [{
            "type": "exclusion",
            "columns": ["A", "B", "C"],
            "mode": "exactly_one",
            "threshold": 0.01,
            "group_name": "溶媒",
        }]
        result = ui_constraints_to_backend(items)
        assert len(result) == 1
        assert isinstance(result[0], AtLeastNConstraint)

    def test_convert_conditional_constraint(self):
        items = [{
            "type": "conditional",
            "if_var": "A",
            "if_op": ">",
            "if_val": 10,
            "then_var": "B",
            "then_op": ">=",
            "then_val": 50,
        }]
        result = ui_constraints_to_backend(items)
        assert len(result) == 1
        assert isinstance(result[0], CustomConstraint)

    def test_convert_formula_constraint(self):
        items = [{
            "type": "formula",
            "expression": "A**2 + B*C",
            "operator": "<=",
            "rhs": 1000,
        }]
        result = ui_constraints_to_backend(items)
        assert len(result) == 1
        assert isinstance(result[0], CustomConstraint)

    def test_convert_multiple_constraints(self):
        items = [
            {"type": "range", "column": "A", "lo": 0, "hi": 100},
            {"type": "sum", "columns": ["B", "C"], "target": 1.0},
            {"type": "formula", "expression": "True", "operator": "==", "rhs": True},
        ]
        result = ui_constraints_to_backend(items)
        assert len(result) == 3

    def test_convert_empty_list(self):
        result = ui_constraints_to_backend([])
        assert result == []

    def test_convert_unknown_type_skipped(self):
        items = [{"type": "unknown_xyz"}]
        result = ui_constraints_to_backend(items)
        assert result == []

    def test_convert_invalid_item_skipped(self):
        items = [{"type": "range"}]  # columnなし→エラーだが例外はキャッチ
        result = ui_constraints_to_backend(items)
        assert result == []  # エラーでスキップ


# ═══════════════════════════════════════════════════════════
# 自然言語プレビューテスト
# ═══════════════════════════════════════════════════════════
class TestDescribeConstraint:
    def test_range_both_bounds(self):
        desc = describe_constraint({"type": "range", "column": "温度", "lo": 10, "hi": 50})
        assert "温度" in desc
        assert "10" in desc
        assert "50" in desc

    def test_range_lower_only(self):
        desc = describe_constraint({"type": "range", "column": "X", "lo": 5, "hi": None})
        assert "5" in desc
        assert "以上" in desc

    def test_range_upper_only(self):
        desc = describe_constraint({"type": "range", "column": "X", "lo": None, "hi": 100})
        assert "100" in desc
        assert "以下" in desc

    def test_sum_constraint(self):
        desc = describe_constraint({
            "type": "sum", "columns": ["A", "B", "C"], "target": 100,
        })
        assert "A" in desc
        assert "B" in desc
        assert "100" in desc

    def test_ratio_constraint(self):
        desc = describe_constraint({
            "type": "ratio", "target_var": "X", "base_var": "Y", "ratio": 2.0, "operator": "ge",
        })
        assert "X" in desc
        assert "Y" in desc
        assert "2.0" in desc
        assert "≥" in desc

    def test_conditional_constraint(self):
        desc = describe_constraint({
            "type": "conditional",
            "if_var": "A", "if_op": ">", "if_val": 10,
            "then_var": "B", "then_op": ">=", "then_val": 50,
        })
        assert "IF" in desc
        assert "THEN" in desc
        assert "A" in desc
        assert "B" in desc


# ═══════════════════════════════════════════════════════════
# 制約検証テスト
# ═══════════════════════════════════════════════════════════
class TestValidateConstraints:
    def test_no_conflicts_when_no_overlap(self):
        items = [
            {"type": "range", "column": "A", "lo": 0, "hi": 50},
            {"type": "range", "column": "B", "lo": 10, "hi": 100},
        ]
        result = validate_constraints(items)
        assert result["conflicts"] == []

    def test_detect_conflict_same_column(self):
        items = [
            {"type": "range", "column": "A", "lo": 0, "hi": 30},
            {"type": "range", "column": "A", "lo": 50, "hi": 100},
        ]
        result = validate_constraints(items)
        assert len(result["conflicts"]) > 0

    def test_no_conflict_overlapping_ranges(self):
        items = [
            {"type": "range", "column": "A", "lo": 0, "hi": 60},
            {"type": "range", "column": "A", "lo": 30, "hi": 100},
        ]
        result = validate_constraints(items)
        assert result["conflicts"] == []

    def test_satisfaction_rate_with_data(self):
        df = pd.DataFrame({"A": [10, 20, 30, 40, 50], "B": [1, 2, 3, 4, 5]})
        items = [{"type": "range", "column": "A", "lo": 15, "hi": 45}]
        result = validate_constraints(items, df)
        # A=10とA=50は制約外なので3/5
        assert result["n_total"] == 5
        assert result["n_satisfied"] == 3
        assert 0.5 < result["overall_rate"] < 0.7

    def test_empty_items(self):
        result = validate_constraints([])
        assert result["conflicts"] == []
        assert result["overall_rate"] == 1.0


# ═══════════════════════════════════════════════════════════
# テンプレート IO テスト
# ═══════════════════════════════════════════════════════════
class TestTemplateIO:
    def test_save_and_load(self, tmp_path):
        items = [
            {"type": "range", "column": "A", "lo": 10, "hi": 50},
            {"type": "sum", "columns": ["B", "C"], "target": 100},
        ]
        with patch("frontend_nicegui.components.constraint_ui_helpers.TEMPLATE_DIR", tmp_path):
            path = save_template("test_template", items, tags=["配合設計"])
            assert path.exists()

            data = load_template(path)
            assert data["name"] == "test_template"
            assert data["tags"] == ["配合設計"]
            assert len(data["constraints"]) == 2

    def test_list_templates(self, tmp_path):
        with patch("frontend_nicegui.components.constraint_ui_helpers.TEMPLATE_DIR", tmp_path):
            save_template("tmpl1", [{"type": "range", "column": "X"}])
            save_template("tmpl2", [{"type": "sum", "columns": ["A"]}])

            templates = list_templates()
            assert len(templates) == 2

    def test_delete_template(self, tmp_path):
        with patch("frontend_nicegui.components.constraint_ui_helpers.TEMPLATE_DIR", tmp_path):
            path = save_template("to_delete", [])
            assert path.exists()
            delete_template(path)
            assert not path.exists()


# ═══════════════════════════════════════════════════════════
# エラーメッセージ変換テスト
# ═══════════════════════════════════════════════════════════
class TestFriendlyError:
    def test_known_error_type(self):
        err = type("ConstraintConflictError", (Exception,), {})("test")
        result = friendly_error(err)
        assert "矛盾" in result["message"]
        assert "緩和" in result["action"]

    def test_unknown_error_type(self):
        err = RuntimeError("something went wrong")
        result = friendly_error(err)
        assert "予期しない" in result["message"]
        assert "確認" in result["action"]


# ═══════════════════════════════════════════════════════════
# データ統計ヘルパーテスト
# ═══════════════════════════════════════════════════════════
class TestGetColumnStats:
    def test_basic_stats(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        stats = get_column_stats(df, "A")
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0
        assert "q1" in stats
        assert "q3" in stats
        assert "mean" in stats
        assert "std" in stats

    def test_with_nan(self):
        df = pd.DataFrame({"A": [1, np.nan, 3, np.nan, 5]})
        stats = get_column_stats(df, "A")
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_empty_column(self):
        df = pd.DataFrame({"A": pd.Series([], dtype=float)})
        stats = get_column_stats(df, "A")
        assert stats["min"] == 0
        assert stats["max"] == 1
"""Complexity: 6, Description: constraint_ui_helpersのユニットテスト。6種制約変換、検証、テンプレートIO、プレビュー、エラー変換、統計ヘルパーを網羅。"""
