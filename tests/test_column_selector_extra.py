"""
tests/test_column_selector_extra.py

column_selector.py のカバレッジ改善テスト。
ColumnMeta, ColumnSelectorWrapper を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.pipeline.column_selector import (
    ColumnMeta,
    ColumnSelectorWrapper,
)


# ============================================================
# ColumnMeta
# ============================================================

class TestColumnMeta:
    def test_defaults(self):
        m = ColumnMeta()
        assert m.monotonic == 0
        assert m.linearity == "unknown"
        assert m.group is None

    def test_custom(self):
        m = ColumnMeta(monotonic=1, linearity="linear", group="g1")
        assert m.monotonic == 1
        assert m.linearity == "linear"
        assert m.group == "g1"


# ============================================================
# ColumnSelectorWrapper — all モード
# ============================================================

class TestColumnSelectorAll:
    def test_fit_transform_all(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        sel = ColumnSelectorWrapper(mode="all")
        result = sel.fit_transform(df)
        assert list(result.columns) == ["a", "b", "c"]

    def test_selected_columns(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        sel = ColumnSelectorWrapper(mode="all")
        sel.fit(df)
        assert sel.selected_columns == ["x", "y"]

    def test_get_feature_names_out(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        sel = ColumnSelectorWrapper(mode="all")
        sel.fit(df)
        names = sel.get_feature_names_out()
        assert list(names) == ["a", "b"]


# ============================================================
# ColumnSelectorWrapper — include モード
# ============================================================

class TestColumnSelectorInclude:
    def test_include_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        sel = ColumnSelectorWrapper(mode="include", columns=["a", "c"])
        result = sel.fit_transform(df)
        assert list(result.columns) == ["a", "c"]

    def test_include_range(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        sel = ColumnSelectorWrapper(mode="include", col_range=(1, 3))
        result = sel.fit_transform(df)
        assert list(result.columns) == ["b", "c"]

    def test_include_missing_col_warning(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        sel = ColumnSelectorWrapper(mode="include", columns=["a", "nonexistent"])
        result = sel.fit_transform(df)
        assert list(result.columns) == ["a"]

    def test_include_no_columns_no_range(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        sel = ColumnSelectorWrapper(mode="include")
        result = sel.fit_transform(df)
        # no columns or range → all columns
        assert len(result.columns) == 2


# ============================================================
# ColumnSelectorWrapper — exclude モード
# ============================================================

class TestColumnSelectorExclude:
    def test_exclude(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        sel = ColumnSelectorWrapper(mode="exclude", columns=["b"])
        result = sel.fit_transform(df)
        assert "b" not in result.columns
        assert len(result.columns) == 2


# ============================================================
# エラー系
# ============================================================

class TestColumnSelectorErrors:
    def test_non_dataframe_fit(self):
        sel = ColumnSelectorWrapper(mode="all")
        with pytest.raises(TypeError, match="DataFrame"):
            sel.fit(np.array([[1, 2]]))

    def test_non_dataframe_transform(self):
        sel = ColumnSelectorWrapper(mode="all")
        df = pd.DataFrame({"a": [1]})
        sel.fit(df)
        with pytest.raises(TypeError, match="DataFrame"):
            sel.transform(np.array([[1]]))

    def test_unknown_mode(self):
        df = pd.DataFrame({"a": [1]})
        sel = ColumnSelectorWrapper(mode="bad_mode")
        with pytest.raises(ValueError, match="未知"):
            sel.fit(df)


# ============================================================
# メタ情報メソッド
# ============================================================

class TestColumnSelectorMeta:
    def test_get_column_meta(self):
        meta = {"a": ColumnMeta(monotonic=1), "b": ColumnMeta(monotonic=-1)}
        sel = ColumnSelectorWrapper(mode="all", column_meta=meta)
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        sel.fit(df)
        assert sel.get_column_meta("a").monotonic == 1
        assert sel.get_column_meta("c").monotonic == 0  # default

    def test_get_monotonic_constraints(self):
        meta = {"a": ColumnMeta(monotonic=1), "b": ColumnMeta(monotonic=-1)}
        sel = ColumnSelectorWrapper(mode="all", column_meta=meta)
        df = pd.DataFrame({"a": [1], "b": [2]})
        sel.fit(df)
        constraints = sel.get_monotonic_constraints()
        assert constraints == (1, -1)

    def test_get_groups_array(self):
        meta = {"a": ColumnMeta(group="g1"), "b": ColumnMeta(group="g2")}
        sel = ColumnSelectorWrapper(mode="all", column_meta=meta)
        df = pd.DataFrame({"a": [1], "b": [2]})
        sel.fit(df)
        groups = sel.get_groups_array()
        assert groups == ["g1", "g2"]
