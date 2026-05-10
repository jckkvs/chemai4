"""
tests/test_param_schema_extra.py

param_schema.py の低カバレッジ部分を補うテスト。
introspect_params, _infer_from_hint, _infer_from_default,
_extract_docstring_params, apply_params, _convert_value 等を網羅。
"""
from __future__ import annotations

import enum
import inspect
from typing import Optional, Literal, Union

import pytest

from backend.ui.param_schema import (
    ParamSpec,
    introspect_params,
    introspect_adapter,
    introspect_adapter_class,
    apply_params,
    get_basic_specs,
    get_advanced_specs,
    _infer_param_spec,
    _extract_docstring_params,
    _convert_value,
)


# ============================================================
# テスト用クラス群
# ============================================================

class SimpleModel:
    """A simple model.

    Args:
        n_estimators: Number of trees
        learning_rate: Step size for boosting
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        verbose: bool = False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.verbose = verbose

    def get_params(self, deep=False):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
        }


class ModelWithOptionalHints:
    """Model with Optional-type params.

    Parameters
    ----------
    alpha : float | None
        Regularization parameter
    solver : str
        Algorithm to use
    max_features : int | str | None
        Max features selection
    """
    def __init__(
        self,
        alpha: float | None = None,
        solver: str = "auto",
        max_features: int | str | None = None,
    ):
        self.alpha = alpha
        self.solver = solver
        self.max_features = max_features


class EnumModel:
    class Mode(enum.Enum):
        FAST = "fast"
        SLOW = "slow"
        AUTO = "auto"

    def __init__(self, mode: "EnumModel.Mode" = None):
        self.mode = mode or self.Mode.AUTO


class LiteralModel:
    def __init__(self, method: Literal["lbfgs", "sgd", "adam"] = "adam"):
        self.method = method


class ListParamModel:
    def __init__(self, layers: list[int] = None):
        self.layers = layers or [64, 32]


class DictParamModel:
    def __init__(self, options: dict = None):
        self.options = options or {"key": "val"}


class NumpydocModel:
    """
    A model with numpydoc style docstring.

    Parameters
    ----------
    n_estimators : int
        Number of trees in forest.
    max_depth : int
        Maximum depth of trees.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 5):
        pass


class RstDocModel:
    """
    Model with rST-style docstring.

    :param alpha: Regularization strength
    :param beta: Another param
    """
    def __init__(self, alpha: float = 0.1, beta: float = 0.5):
        pass


class NoDocModel:
    def __init__(self, x: int = 5):
        pass


class VarArgModel:
    def __init__(self, *args, n: int = 3, **kwargs):
        pass


class PrivateParamModel:
    def __init__(self, _internal: int = 0, public: int = 1):
        pass


# ============================================================
# introspect_params テスト
# ============================================================

class TestIntrospectParams:
    def test_basic_model(self):
        specs = introspect_params(SimpleModel)
        names = [s.name for s in specs]
        assert "n_estimators" in names
        assert "learning_rate" in names
        assert "verbose" in names

    def test_basic_types(self):
        specs = introspect_params(SimpleModel)
        spec_map = {s.name: s for s in specs}
        assert spec_map["n_estimators"].param_type == "int"
        assert spec_map["learning_rate"].param_type == "float"
        assert spec_map["verbose"].param_type == "bool"

    def test_advanced_classification(self):
        specs = introspect_params(SimpleModel)
        spec_map = {s.name: s for s in specs}
        assert spec_map["verbose"].group == "advanced"
        assert spec_map["n_estimators"].group == "basic"

    def test_optional_types(self):
        specs = introspect_params(ModelWithOptionalHints)
        spec_map = {s.name: s for s in specs}
        assert spec_map["alpha"].nullable is True
        assert spec_map["solver"].param_type in ("select", "str")

    def test_union_type(self):
        specs = introspect_params(ModelWithOptionalHints)
        spec_map = {s.name: s for s in specs}
        # max_features: int | str | None → text
        assert spec_map["max_features"].param_type == "text"

    def test_literal_type(self):
        specs = introspect_params(LiteralModel)
        spec_map = {s.name: s for s in specs}
        assert spec_map["method"].param_type == "select"
        assert set(spec_map["method"].choices) == {"lbfgs", "sgd", "adam"}

    def test_list_type(self):
        specs = introspect_params(ListParamModel)
        spec_map = {s.name: s for s in specs}
        assert spec_map["layers"].param_type == "multiselect"

    def test_dict_default(self):
        specs = introspect_params(DictParamModel)
        spec_map = {s.name: s for s in specs}
        assert spec_map["options"].param_type == "text"

    def test_skip_params(self):
        specs = introspect_params(SimpleModel, skip_params={"verbose"})
        names = [s.name for s in specs]
        assert "verbose" not in names

    def test_extra_descriptions(self):
        specs = introspect_params(
            SimpleModel,
            extra_descriptions={"n_estimators": "Custom desc"},
        )
        spec_map = {s.name: s for s in specs}
        assert spec_map["n_estimators"].description == "Custom desc"

    def test_instance_defaults(self):
        instance = SimpleModel(n_estimators=200)
        specs = introspect_params(SimpleModel, instance=instance)
        spec_map = {s.name: s for s in specs}
        assert spec_map["n_estimators"].default == 200

    def test_vararg_model(self):
        specs = introspect_params(VarArgModel)
        names = [s.name for s in specs]
        assert "n" in names
        assert "args" not in names
        assert "kwargs" not in names

    def test_private_params_excluded(self):
        specs = introspect_params(PrivateParamModel)
        names = [s.name for s in specs]
        assert "_internal" not in names
        assert "public" in names


# ============================================================
# introspect_adapter / introspect_adapter_class テスト
# ============================================================

class TestIntrospectAdapter:
    def test_adapter_instance(self):
        instance = SimpleModel(n_estimators=50)
        specs = introspect_adapter(instance)
        assert len(specs) > 0
        spec_map = {s.name: s for s in specs}
        assert spec_map["n_estimators"].default == 50

    def test_adapter_class(self):
        specs = introspect_adapter_class(SimpleModel)
        assert len(specs) > 0


# ============================================================
# _extract_docstring_params テスト
# ============================================================

class TestDocstringParsing:
    def test_google_style(self):
        result = _extract_docstring_params(SimpleModel)
        assert "n_estimators" in result or "learning_rate" in result

    def test_numpydoc_style(self):
        result = _extract_docstring_params(NumpydocModel)
        assert "n_estimators" in result

    def test_rst_style(self):
        result = _extract_docstring_params(RstDocModel)
        assert "alpha" in result

    def test_no_doc(self):
        result = _extract_docstring_params(NoDocModel)
        assert isinstance(result, dict)


# ============================================================
# ParamSpec.to_dict テスト
# ============================================================

class TestParamSpecToDict:
    def test_to_dict(self):
        spec = ParamSpec(
            name="n_estimators",
            param_type="int",
            default=100,
            min_val=1,
            max_val=10000,
            step=10,
            description="Number of trees",
            group="basic",
            nullable=False,
        )
        d = spec.to_dict()
        assert d["name"] == "n_estimators"
        assert d["param_type"] == "int"
        assert d["default"] == 100
        assert d["min_val"] == 1
        assert d["max_val"] == 10000


# ============================================================
# apply_params テスト
# ============================================================

class TestApplyParams:
    def test_basic_apply(self):
        specs = introspect_params(SimpleModel)
        result = apply_params(specs, {"n_estimators": 200})
        assert result["n_estimators"] == 200

    def test_no_change(self):
        specs = introspect_params(SimpleModel)
        result = apply_params(specs, {"n_estimators": 100})
        assert "n_estimators" not in result  # Same as default

    def test_type_conversion(self):
        specs = introspect_params(SimpleModel)
        result = apply_params(specs, {"n_estimators": "500"})
        assert result["n_estimators"] == 500
        assert isinstance(result["n_estimators"], int)

    def test_float_conversion(self):
        specs = introspect_params(SimpleModel)
        result = apply_params(specs, {"learning_rate": "0.05"})
        assert abs(result["learning_rate"] - 0.05) < 1e-9

    def test_bool_conversion(self):
        specs = introspect_params(SimpleModel)
        result = apply_params(specs, {"verbose": "true"})
        assert result["verbose"] is True

    def test_missing_param(self):
        specs = introspect_params(SimpleModel)
        result = apply_params(specs, {"nonexistent": "value"})
        assert "nonexistent" not in result


# ============================================================
# _convert_value テスト
# ============================================================

class TestConvertValue:
    def test_none_nullable(self):
        spec = ParamSpec(name="x", param_type="int", nullable=True)
        assert _convert_value(None, spec) is None

    def test_none_not_nullable(self):
        spec = ParamSpec(name="x", param_type="int", default=5, nullable=False)
        assert _convert_value(None, spec) == 5

    def test_empty_string_nullable(self):
        spec = ParamSpec(name="x", param_type="str", nullable=True)
        assert _convert_value("", spec) is None

    def test_bool_string_conversion(self):
        spec = ParamSpec(name="x", param_type="bool")
        assert _convert_value("yes", spec) is True
        assert _convert_value("no", spec) is False
        assert _convert_value("on", spec) is True

    def test_multiselect_string(self):
        spec = ParamSpec(name="x", param_type="multiselect")
        result = _convert_value("a, b, c", spec)
        assert result == ["a", "b", "c"]

    def test_multiselect_list(self):
        spec = ParamSpec(name="x", param_type="multiselect")
        result = _convert_value(["a", "b"], spec)
        assert result == ["a", "b"]

    def test_text_numeric_float(self):
        spec = ParamSpec(name="x", param_type="text")
        assert _convert_value("3.14", spec) == 3.14

    def test_text_numeric_int(self):
        spec = ParamSpec(name="x", param_type="text")
        assert _convert_value("42", spec) == 42

    def test_text_none_string(self):
        spec = ParamSpec(name="x", param_type="text")
        assert _convert_value("None", spec) is None

    def test_text_bool_string(self):
        spec = ParamSpec(name="x", param_type="text")
        assert _convert_value("true", spec) is True
        assert _convert_value("false", spec) is False

    def test_text_plain_string(self):
        spec = ParamSpec(name="x", param_type="text")
        assert _convert_value("hello", spec) == "hello"

    def test_select(self):
        spec = ParamSpec(name="x", param_type="select")
        assert _convert_value("option_a", spec) == "option_a"


# ============================================================
# get_basic_specs / get_advanced_specs テスト
# ============================================================

class TestGroupFiltering:
    def test_basic_specs(self):
        specs = introspect_params(SimpleModel)
        basic = get_basic_specs(specs)
        assert all(s.group == "basic" for s in basic)

    def test_advanced_specs(self):
        specs = introspect_params(SimpleModel)
        advanced = get_advanced_specs(specs)
        assert all(s.group == "advanced" for s in advanced)
        assert any(s.name == "verbose" for s in advanced)
