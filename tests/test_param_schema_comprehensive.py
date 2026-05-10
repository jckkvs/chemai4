"""
tests/test_param_schema_comprehensive.py

param_schema.py の包括テスト。
introspect_params, ParamSpec, apply_params, _convert_value,
get_basic_specs, get_advanced_specs を網羅。
"""
from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR

from backend.ui.param_schema import (
    ParamSpec,
    introspect_params,
    introspect_adapter,
    apply_params,
    get_basic_specs,
    get_advanced_specs,
    _convert_value,
    _extract_docstring_params,
)


# ============================================================
# ParamSpec テスト
# ============================================================

class TestParamSpec:
    def test_defaults(self):
        spec = ParamSpec(name="x", param_type="int")
        assert spec.name == "x"
        assert spec.default is None
        assert spec.group == "basic"

    def test_to_dict(self):
        spec = ParamSpec(name="x", param_type="int", default=10)
        d = spec.to_dict()
        assert d["name"] == "x"
        assert d["default"] == 10


# ============================================================
# introspect_params テスト
# ============================================================

class TestIntrospectParams:
    def test_random_forest(self):
        specs = introspect_params(RandomForestRegressor)
        assert len(specs) > 0
        names = [s.name for s in specs]
        assert "n_estimators" in names

    def test_ridge(self):
        specs = introspect_params(Ridge)
        names = [s.name for s in specs]
        assert "alpha" in names

    def test_svr(self):
        specs = introspect_params(SVR)
        names = [s.name for s in specs]
        assert "C" in names

    def test_logistic(self):
        specs = introspect_params(LogisticRegression)
        names = [s.name for s in specs]
        assert "C" in names

    def test_advanced_group(self):
        specs = introspect_params(RandomForestRegressor)
        advanced = [s for s in specs if s.group == "advanced"]
        names = [s.name for s in advanced]
        assert "random_state" in names
        assert "n_jobs" in names

    def test_instance(self):
        est = RandomForestRegressor(n_estimators=200)
        specs = introspect_params(type(est), instance=est)
        spec_ne = [s for s in specs if s.name == "n_estimators"][0]
        assert spec_ne.default == 200

    def test_skip_params(self):
        specs = introspect_params(Ridge, skip_params={"alpha"})
        names = [s.name for s in specs]
        assert "alpha" not in names

    def test_extra_descriptions(self):
        specs = introspect_params(Ridge, extra_descriptions={"alpha": "正則化強度"})
        spec_a = [s for s in specs if s.name == "alpha"][0]
        assert "正則化" in spec_a.description


# ============================================================
# apply_params テスト
# ============================================================

class TestApplyParams:
    def test_basic(self):
        specs = [
            ParamSpec(name="n", param_type="int", default=100),
            ParamSpec(name="lr", param_type="float", default=0.1),
        ]
        result = apply_params(specs, {"n": 200, "lr": 0.01})
        assert result["n"] == 200
        assert result["lr"] == 0.01

    def test_skip_default(self):
        specs = [ParamSpec(name="n", param_type="int", default=100)]
        result = apply_params(specs, {"n": 100})
        assert "n" not in result  # same as default

    def test_missing_key(self):
        specs = [ParamSpec(name="n", param_type="int", default=100)]
        result = apply_params(specs, {})
        assert len(result) == 0


# ============================================================
# _convert_value テスト
# ============================================================

class TestConvertValue:
    def test_bool_true(self):
        spec = ParamSpec(name="x", param_type="bool")
        assert _convert_value("true", spec) is True
        assert _convert_value(True, spec) is True

    def test_bool_false(self):
        spec = ParamSpec(name="x", param_type="bool")
        assert _convert_value("false", spec) is False
        assert _convert_value("0", spec) is False

    def test_int(self):
        spec = ParamSpec(name="x", param_type="int")
        assert _convert_value("42", spec) == 42
        assert _convert_value(3.7, spec) == 3

    def test_float(self):
        spec = ParamSpec(name="x", param_type="float")
        assert _convert_value("3.14", spec) == pytest.approx(3.14)

    def test_str(self):
        spec = ParamSpec(name="x", param_type="str")
        assert _convert_value(42, spec) == "42"

    def test_select(self):
        spec = ParamSpec(name="x", param_type="select")
        assert _convert_value("rbf", spec) == "rbf"

    def test_multiselect(self):
        spec = ParamSpec(name="x", param_type="multiselect")
        assert _convert_value("a,b,c", spec) == ["a", "b", "c"]
        assert _convert_value(["a", "b"], spec) == ["a", "b"]

    def test_text_number(self):
        spec = ParamSpec(name="x", param_type="text")
        assert _convert_value("42", spec) == 42
        assert _convert_value("3.14", spec) == pytest.approx(3.14)

    def test_text_none(self):
        spec = ParamSpec(name="x", param_type="text")
        assert _convert_value("none", spec) is None

    def test_text_bool(self):
        spec = ParamSpec(name="x", param_type="text")
        assert _convert_value("true", spec) is True

    def test_nullable(self):
        spec = ParamSpec(name="x", param_type="int", nullable=True)
        assert _convert_value(None, spec) is None


# ============================================================
# get_basic / get_advanced テスト
# ============================================================

class TestGroupFilters:
    def test_basic(self):
        specs = [
            ParamSpec(name="a", param_type="int", group="basic"),
            ParamSpec(name="b", param_type="int", group="advanced"),
        ]
        assert len(get_basic_specs(specs)) == 1
        assert len(get_advanced_specs(specs)) == 1


# ============================================================
# _extract_docstring_params テスト
# ============================================================

class TestDocstringExtraction:
    def test_sklearn_class(self):
        result = _extract_docstring_params(RandomForestRegressor)
        assert isinstance(result, dict)
        # sklearn docstrings should yield at least some params
        assert len(result) > 0
