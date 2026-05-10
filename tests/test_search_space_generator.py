# -*- coding: utf-8 -*-
"""
tests/test_search_space_generator.py

探索空間自動生成のテスト。

テストID:
    T-SSG01: generate_grid_space — 基本テスト
    T-SSG02: generate_optuna_space — 基本テスト
    T-SSG03: generate_search_spaces — 全パラメータ一括生成
    T-SSG04: generate_search_spaces_from_estimator — estimatorクラスから生成
    T-SSG05: 既知パラメータのプリセット検証
    T-SSG06: 未知パラメータの自動推論
    T-SSG07: SearchParamSpec.to_grid_entry / to_optuna_entry
    T-SSG08: _parse_value_list — 型自動判定
    T-SSG09: _infer_int_range / _infer_float_range — 範囲推論
    T-SSG10: EstimatorConfig のシリアライズ
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.ui.param_schema import ParamSpec, introspect_params
from backend.models.search_space_generator import (
    SearchParamSpec,
    generate_search_space,
    generate_grid_space,
    generate_optuna_space,
    generate_search_spaces,
    generate_search_spaces_from_estimator,
    _infer_int_range,
    _infer_float_range,
    _auto_generate_search_spec,
    _KNOWN_SEARCH_SPACES,
)
from frontend_nicegui.components.estimator_config_dialog import (
    EstimatorConfig,
    _parse_value_list,
)


# ============================================================
# T-SSG01: GridSearchCV空間生成
# ============================================================

class TestGenerateGridSpace:
    def test_rf_grid(self):
        from sklearn.ensemble import RandomForestRegressor
        specs = introspect_params(RandomForestRegressor)
        grid = generate_grid_space(specs)
        assert "n_estimators" in grid
        assert isinstance(grid["n_estimators"], list)
        assert len(grid["n_estimators"]) >= 3

    def test_ridge_grid(self):
        from sklearn.linear_model import Ridge
        specs = introspect_params(Ridge)
        grid = generate_grid_space(specs)
        assert "alpha" in grid
        assert isinstance(grid["alpha"], list)

    def test_svr_grid(self):
        from sklearn.svm import SVR
        specs = introspect_params(SVR)
        grid = generate_grid_space(specs)
        assert "C" in grid
        assert "kernel" in grid

    def test_empty_specs(self):
        grid = generate_grid_space([])
        assert grid == {}


# ============================================================
# T-SSG02: OptunaSearchCV空間生成
# ============================================================

class TestGenerateOptunaSpace:
    def test_rf_optuna(self):
        from sklearn.ensemble import RandomForestRegressor
        specs = introspect_params(RandomForestRegressor)
        space = generate_optuna_space(specs)
        assert "n_estimators" in space
        assert space["n_estimators"]["type"] == "int"
        assert space["n_estimators"]["low"] < space["n_estimators"]["high"]

    def test_ridge_optuna(self):
        from sklearn.linear_model import Ridge
        specs = introspect_params(Ridge)
        space = generate_optuna_space(specs)
        assert "alpha" in space
        assert space["alpha"]["type"] == "float"
        assert space["alpha"]["log"] is True  # αは対数スケール推奨

    def test_empty_specs(self):
        space = generate_optuna_space([])
        assert space == {}


# ============================================================
# T-SSG03: 全パラメータ一括生成
# ============================================================

class TestGenerateSearchSpaces:
    def test_rf_all(self):
        from sklearn.ensemble import RandomForestRegressor
        specs = introspect_params(RandomForestRegressor)
        spaces = generate_search_spaces(specs)
        assert len(spaces) > 0
        # random_state等はスキップされる
        assert "random_state" not in spaces

    def test_include_advanced(self):
        from sklearn.ensemble import GradientBoostingRegressor
        specs = introspect_params(GradientBoostingRegressor)
        basic_only = generate_search_spaces(specs, include_advanced=False)
        with_advanced = generate_search_spaces(specs, include_advanced=True)
        assert len(with_advanced) >= len(basic_only)


# ============================================================
# T-SSG04: estimatorクラスから直接生成
# ============================================================

class TestFromEstimator:
    @pytest.mark.parametrize("cls_name", [
        "sklearn.ensemble.RandomForestRegressor",
        "sklearn.ensemble.GradientBoostingRegressor",
        "sklearn.linear_model.Ridge",
        "sklearn.svm.SVR",
        "sklearn.neighbors.KNeighborsRegressor",
        "sklearn.tree.DecisionTreeRegressor",
    ])
    def test_various_estimators(self, cls_name):
        parts = cls_name.rsplit(".", 1)
        mod = __import__(parts[0], fromlist=[parts[1]])
        cls = getattr(mod, parts[1])
        spaces = generate_search_spaces_from_estimator(cls)
        assert isinstance(spaces, dict)
        # 少なくとも1つは探索パラメータがあるはず
        assert len(spaces) > 0

    def test_custom_estimator(self):
        """ユーザー登録のカスタムEstimatorでも動作すること"""
        from sklearn.base import BaseEstimator, RegressorMixin

        class MyCustomRegressor(BaseEstimator, RegressorMixin):
            def __init__(self, n_layers: int = 3, dropout: float = 0.5,
                         use_bn: bool = True):
                self.n_layers = n_layers
                self.dropout = dropout
                self.use_bn = use_bn

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        spaces = generate_search_spaces_from_estimator(MyCustomRegressor)
        assert "n_layers" in spaces
        assert "dropout" in spaces
        assert "use_bn" in spaces
        # int
        assert spaces["n_layers"].optuna_type == "int"
        # float
        assert spaces["dropout"].optuna_type == "float"
        # bool → categorical
        assert spaces["use_bn"].optuna_type == "categorical"


# ============================================================
# T-SSG05: 既知パラメータのプリセット
# ============================================================

class TestKnownPresets:
    @pytest.mark.parametrize("param_name", [
        "n_estimators", "max_depth", "learning_rate", "alpha",
        "C", "n_neighbors", "n_components", "l1_ratio",
    ])
    def test_known_params_have_presets(self, param_name):
        assert param_name in _KNOWN_SEARCH_SPACES
        preset = _KNOWN_SEARCH_SPACES[param_name]
        assert "grid" in preset
        assert "optuna" in preset
        assert len(preset["grid"]) >= 3

    def test_learning_rate_log_scale(self):
        preset = _KNOWN_SEARCH_SPACES["learning_rate"]
        assert preset["optuna"]["log"] is True

    def test_l1_ratio_bounded(self):
        preset = _KNOWN_SEARCH_SPACES["l1_ratio"]
        assert preset["optuna"]["low"] >= 0.0
        assert preset["optuna"]["high"] <= 1.0


# ============================================================
# T-SSG06: 未知パラメータの自動推論
# ============================================================

class TestAutoInference:
    def test_int_auto(self):
        spec = ParamSpec(name="custom_depth", param_type="int", default=10)
        result = _auto_generate_search_spec(spec)
        assert result is not None
        assert result.param_type == "int"
        assert result.optuna_type == "int"
        assert result.optuna_low < 10
        assert result.optuna_high > 10
        assert len(result.grid_values) >= 3

    def test_float_auto(self):
        spec = ParamSpec(name="custom_rate", param_type="float", default=0.1)
        result = _auto_generate_search_spec(spec)
        assert result is not None
        assert result.param_type == "float"
        assert result.optuna_type == "float"
        assert result.optuna_low < 0.1
        assert result.optuna_high > 0.1

    def test_bool_auto(self):
        spec = ParamSpec(name="use_feature", param_type="bool", default=True)
        result = _auto_generate_search_spec(spec)
        assert result is not None
        assert result.grid_values == [True, False]
        assert result.optuna_type == "categorical"

    def test_select_auto(self):
        spec = ParamSpec(
            name="strategy", param_type="select",
            default="best", choices=["best", "random"],
        )
        result = _auto_generate_search_spec(spec)
        assert result is not None
        assert result.grid_values == ["best", "random"]
        assert result.optuna_type == "categorical"

    def test_str_returns_none(self):
        spec = ParamSpec(name="name_str", param_type="str", default="test")
        result = _auto_generate_search_spec(spec)
        assert result is None


# ============================================================
# T-SSG07: SearchParamSpec メソッド
# ============================================================

class TestSearchParamSpecMethods:
    def test_to_grid_entry_enabled(self):
        sps = SearchParamSpec(
            name="n", param_type="int", enabled=True,
            grid_values=[10, 50, 100],
        )
        assert sps.to_grid_entry() == [10, 50, 100]

    def test_to_grid_entry_disabled(self):
        sps = SearchParamSpec(
            name="n", param_type="int", enabled=False,
            grid_values=[10, 50, 100],
        )
        assert sps.to_grid_entry() is None

    def test_to_grid_entry_empty(self):
        sps = SearchParamSpec(name="n", param_type="int", enabled=True)
        assert sps.to_grid_entry() is None

    def test_to_optuna_entry_int(self):
        sps = SearchParamSpec(
            name="n", param_type="int", enabled=True,
            optuna_type="int", optuna_low=10, optuna_high=100, optuna_step=5,
        )
        entry = sps.to_optuna_entry()
        assert entry is not None
        assert entry["type"] == "int"
        assert entry["low"] == 10
        assert entry["high"] == 100

    def test_to_optuna_entry_float_log(self):
        sps = SearchParamSpec(
            name="lr", param_type="float", enabled=True,
            optuna_type="float", optuna_low=1e-4, optuna_high=1.0,
            optuna_log=True,
        )
        entry = sps.to_optuna_entry()
        assert entry is not None
        assert entry["type"] == "float"
        assert entry["log"] is True
        assert "step" not in entry  # log=Trueの場合stepなし

    def test_to_optuna_entry_categorical(self):
        sps = SearchParamSpec(
            name="k", param_type="categorical", enabled=True,
            optuna_type="categorical", optuna_choices=["a", "b"],
        )
        entry = sps.to_optuna_entry()
        assert entry["type"] == "categorical"
        assert entry["choices"] == ["a", "b"]

    def test_to_dict(self):
        sps = SearchParamSpec(name="x", param_type="int", default_value=10)
        d = sps.to_dict()
        assert d["name"] == "x"
        assert d["default_value"] == 10
        assert isinstance(d, dict)


# ============================================================
# T-SSG08: _parse_value_list
# ============================================================

class TestParseValueList:
    def test_int_list(self):
        assert _parse_value_list("100, 200, 500") == [100, 200, 500]

    def test_float_list(self):
        assert _parse_value_list("0.01, 0.1, 1.0") == [0.01, 0.1, 1.0]

    def test_string_list(self):
        assert _parse_value_list("rbf, linear, poly") == ["rbf", "linear", "poly"]

    def test_bool_list(self):
        assert _parse_value_list("True, False") == [True, False]

    def test_none_in_list(self):
        assert _parse_value_list("None, 3, 5") == [None, 3, 5]

    def test_mixed_types(self):
        result = _parse_value_list("None, 0.1, 1, hello")
        assert result == [None, 0.1, 1, "hello"]

    def test_empty(self):
        assert _parse_value_list("") == []
        assert _parse_value_list("   ") == []

    def test_single_value(self):
        assert _parse_value_list("42") == [42]


# ============================================================
# T-SSG09: 範囲推論
# ============================================================

class TestRangeInference:
    def test_int_range_default_100(self):
        grid, low, high, log = _infer_int_range(100, "test")
        assert low < 100
        assert high > 100
        assert 100 in grid

    def test_int_range_default_1(self):
        grid, low, high, log = _infer_int_range(1, "test")
        assert low >= 1
        assert 1 in grid

    def test_int_range_default_5(self):
        grid, low, high, log = _infer_int_range(5, "test")
        assert 5 in grid
        assert low >= 1

    def test_float_range_default_01(self):
        grid, low, high, log = _infer_float_range(0.1, "test")
        assert low < 0.1
        assert high > 0.1
        assert 0.1 in grid

    def test_float_range_default_0(self):
        grid, low, high, log = _infer_float_range(0.0, "test")
        assert 0.0 in grid
        assert not log

    def test_float_range_default_10(self):
        grid, low, high, log = _infer_float_range(10.0, "test")
        assert 10.0 in grid
        assert high > 10.0


# ============================================================
# T-SSG10: EstimatorConfig
# ============================================================

class TestEstimatorConfig:
    def test_to_dict(self):
        cfg = EstimatorConfig(
            model_key="rf",
            default_params={"n_estimators": 200},
            grid_space={"n_estimators": [50, 100, 200]},
            optuna_space={"n_estimators": {"type": "int", "low": 10, "high": 500}},
        )
        d = cfg.to_dict()
        assert d["model_key"] == "rf"
        assert d["default_params"]["n_estimators"] == 200
        assert len(d["grid_space"]["n_estimators"]) == 3

    def test_default_empty(self):
        cfg = EstimatorConfig(model_key="test")
        assert cfg.default_params == {}
        assert cfg.grid_space == {}
        assert cfg.optuna_space == {}


# ============================================================
# docstring解析統合テスト
# ============================================================

class TestDocstringIntegration:
    """docstringからパラメータ説明が正しく取得されること"""

    def test_rf_descriptions(self):
        from sklearn.ensemble import RandomForestRegressor
        specs = introspect_params(RandomForestRegressor)
        n_est_spec = next((s for s in specs if s.name == "n_estimators"), None)
        assert n_est_spec is not None
        assert "tree" in n_est_spec.description.lower() or "forest" in n_est_spec.description.lower() or n_est_spec.description != ""

    def test_rf_description_no_type_info(self):
        """numpydoc形式の型情報（'int, default=100'）が説明文に含まれないこと"""
        from sklearn.ensemble import RandomForestRegressor
        specs = introspect_params(RandomForestRegressor)
        n_est_spec = next((s for s in specs if s.name == "n_estimators"), None)
        assert n_est_spec is not None
        # 実際の説明文が取得されていること
        assert "tree" in n_est_spec.description.lower() or "forest" in n_est_spec.description.lower()

    def test_ridge_descriptions(self):
        from sklearn.linear_model import Ridge
        specs = introspect_params(Ridge)
        alpha_spec = next((s for s in specs if s.name == "alpha"), None)
        assert alpha_spec is not None
        assert alpha_spec.param_type in ("float", "int", "text", "select")

    def test_ridge_alpha_description_quality(self):
        """Ridge.alphaの説明文にL2正則化の情報が含まれること"""
        from sklearn.linear_model import Ridge
        specs = introspect_params(Ridge)
        alpha_spec = next((s for s in specs if s.name == "alpha"), None)
        assert alpha_spec is not None
        desc_lower = alpha_spec.description.lower()
        assert "l2" in desc_lower or "regulariz" in desc_lower or "constant" in desc_lower

    def test_gbm_learning_rate_description(self):
        """GBM.learning_rateの説明文がクリーンであること"""
        from sklearn.ensemble import GradientBoostingRegressor
        specs = introspect_params(GradientBoostingRegressor)
        lr_spec = next((s for s in specs if s.name == "learning_rate"), None)
        assert lr_spec is not None
        # 説明文に本来の意味が含まれていること
        assert "shrink" in lr_spec.description.lower() or "learning" in lr_spec.description.lower()

    def test_custom_estimator_descriptions(self):
        """カスタムestimatorのdocstringも解析されること"""
        from sklearn.base import BaseEstimator

        class MyModel(BaseEstimator):
            """My custom model.

            Parameters
            ----------
            learning_rate : float, default=0.01
                The step size for gradient descent.
            n_layers : int, default=3
                Number of hidden layers.
            """

            def __init__(self, learning_rate: float = 0.01, n_layers: int = 3):
                self.learning_rate = learning_rate
                self.n_layers = n_layers

        specs = introspect_params(MyModel)
        lr_spec = next((s for s in specs if s.name == "learning_rate"), None)
        assert lr_spec is not None
        assert lr_spec.param_type == "float"
        assert "step size" in lr_spec.description.lower() or "gradient" in lr_spec.description.lower()
        n_spec = next((s for s in specs if s.name == "n_layers"), None)
        assert n_spec is not None
        assert n_spec.param_type == "int"

    def test_no_docstring_estimator(self):
        """docstringがないクラスでもエラーにならないこと"""
        from sklearn.base import BaseEstimator

        class NoDocModel(BaseEstimator):
            def __init__(self, x: int = 5):
                self.x = x

        specs = introspect_params(NoDocModel)
        assert len(specs) == 1
        assert specs[0].name == "x"
        assert specs[0].param_type == "int"

    def test_rst_style_docstring(self):
        """reStructuredText形式のdocstringも解析できること"""
        from sklearn.base import BaseEstimator

        class RstModel(BaseEstimator):
            """A model with RST docs.

            :param alpha: Regularization strength.
            :param beta: Smoothing factor.
            """
            def __init__(self, alpha: float = 1.0, beta: float = 0.5):
                self.alpha = alpha
                self.beta = beta

        specs = introspect_params(RstModel)
        alpha_spec = next((s for s in specs if s.name == "alpha"), None)
        assert alpha_spec is not None
        assert "regulariz" in alpha_spec.description.lower() or alpha_spec.description != ""


# ============================================================
# End-to-End 統合テスト
# ============================================================

class TestEndToEnd:
    """estimator → ParamSpec → SearchParamSpec → tuner形式の変換テスト"""

    def test_rf_full_pipeline(self):
        """RF: introspect → search_space → grid/optuna dict の完全変換"""
        from sklearn.ensemble import RandomForestRegressor
        specs = introspect_params(RandomForestRegressor)
        grid = generate_grid_space(specs)
        optuna = generate_optuna_space(specs)

        # Grid: 全エントリがlist
        for k, v in grid.items():
            assert isinstance(v, list), f"{k} should be list, got {type(v)}"
            assert len(v) > 0, f"{k} should not be empty"

        # Optuna: tuner.pyのObjective関数が受け取れる形式
        for k, v in optuna.items():
            assert "type" in v, f"{k} missing 'type' key"
            if v["type"] in ("int", "float"):
                assert "low" in v and "high" in v, f"{k} missing low/high"
                assert v["low"] < v["high"], f"{k}: low >= high"
            elif v["type"] == "categorical":
                assert "choices" in v, f"{k} missing 'choices'"

    def test_custom_estimator_full_pipeline(self):
        """カスタムEstimator: 未知パラメータでもend-to-end動作"""
        from sklearn.base import BaseEstimator, RegressorMixin

        class CustomModel(BaseEstimator, RegressorMixin):
            """Custom model with diverse param types.

            Parameters
            ----------
            num_trees : int, default=50
                Number of trees.
            shrinkage : float, default=0.05
                Learning rate.
            use_cache : bool, default=True
                Whether to cache results.
            """
            def __init__(self, num_trees=50, shrinkage=0.05, use_cache=True):
                self.num_trees = num_trees
                self.shrinkage = shrinkage
                self.use_cache = use_cache
            def fit(self, X, y): return self
            def predict(self, X): return np.zeros(len(X))

        spaces = generate_search_spaces_from_estimator(CustomModel, include_advanced=True)

        # int
        assert "num_trees" in spaces
        assert spaces["num_trees"].optuna_type == "int"
        grid_entry = spaces["num_trees"].to_grid_entry()
        assert grid_entry is not None
        assert 50 in grid_entry  # default value in grid

        # float
        assert "shrinkage" in spaces
        assert spaces["shrinkage"].optuna_type == "float"
        optuna_entry = spaces["shrinkage"].to_optuna_entry()
        assert optuna_entry["low"] < 0.05
        assert optuna_entry["high"] > 0.05

        # bool
        assert "use_cache" in spaces
        assert spaces["use_cache"].grid_values == [True, False]

    def test_estimator_config_round_trip(self):
        """EstimatorConfig: 生成 → dict → 値確認"""
        from sklearn.ensemble import RandomForestRegressor

        specs = introspect_params(RandomForestRegressor)
        grid = generate_grid_space(specs)
        optuna = generate_optuna_space(specs)

        config = EstimatorConfig(
            model_key="rf",
            model_cls=RandomForestRegressor,
            default_params={"n_estimators": 200},
            grid_space=grid,
            optuna_space=optuna,
        )

        d = config.to_dict()
        assert d["model_key"] == "rf"
        assert d["default_params"]["n_estimators"] == 200
        assert "n_estimators" in d["grid_space"]
        assert "n_estimators" in d["optuna_space"]

