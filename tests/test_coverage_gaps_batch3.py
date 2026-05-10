"""
tests/test_coverage_gaps_batch3.py

カバレッジ100%達成のための補充テスト。
各モジュールの未カバー行/分岐を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


# ============================================================
# backend/data/eda.py — 未カバー: L200, 217, 227, 317
# ============================================================

class TestEdaCoverageGaps:
    def test_detect_outliers_modified_zscore_zero_mad(self):
        from backend.data.eda import detect_outliers
        df = pd.DataFrame({"x": [5, 5, 5, 5, 5]})
        results = detect_outliers(df, method="modified_zscore")
        assert len(results) == 0

    def test_detect_outliers_zscore_zero_sigma(self):
        from backend.data.eda import detect_outliers
        df = pd.DataFrame({"x": [3, 3, 3, 3, 3]})
        results = detect_outliers(df, method="zscore")
        assert len(results) == 0

    def test_detect_outliers_nonexistent_col(self):
        from backend.data.eda import detect_outliers
        df = pd.DataFrame({"x": [1, 2, 3]})
        results = detect_outliers(df, cols=["nonexist"])
        assert len(results) == 0

    def test_analyze_target_auto_int_many_unique(self):
        from backend.data.eda import analyze_target
        df = pd.DataFrame({"t": np.arange(100)})
        result = analyze_target(df, "t", task="auto")
        assert result["task"] == "regression"


# ============================================================
# backend/data/data_cleaner.py — 未カバー: L254, 343, 348, 371, 379
# ============================================================

class TestDataCleanerCoverageGaps:
    def test_clip_outliers_non_numeric_col(self):
        from backend.data.data_cleaner import clip_outliers
        df = pd.DataFrame({"num": [1.0, 2.0, 100.0, 3.0], "cat": ["a", "b", "c", "d"]})
        result, action = clip_outliers(df, columns=["num", "cat"])
        assert result.shape == df.shape

    def test_preview_missing_impact_subset(self):
        from backend.data.data_cleaner import preview_missing_impact
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 3]})
        n = preview_missing_impact(df, threshold=0.5, subset=["a", "b"])
        assert isinstance(n, int)

    def test_preview_missing_impact_no_check_cols(self):
        from backend.data.data_cleaner import preview_missing_impact
        df = pd.DataFrame({"a": [1, 2, 3]})
        n = preview_missing_impact(df, threshold=0.5, subset=["nonexist"])
        assert n == 0

    def test_preview_outlier_impact_columns_param(self):
        from backend.data.data_cleaner import preview_outlier_impact
        df = pd.DataFrame({"x": [1, 2, 3, 100], "y": [1, 1, 1, 1]})
        result = preview_outlier_impact(df, columns=["x"])
        assert isinstance(result, dict)

    def test_preview_outlier_non_numeric_skip(self):
        from backend.data.data_cleaner import preview_outlier_impact
        df = pd.DataFrame({"x": [1, 2, 3], "cat": ["a", "b", "c"]})
        result = preview_outlier_impact(df, columns=["x", "cat"])
        assert isinstance(result, dict)


# ============================================================
# backend/data/feature_engineer.py
# ============================================================

class TestFeatureEngineerCoverageGaps:
    def test_datetime_extractor_series_input(self):
        from backend.data.feature_engineer import DatetimeFeatureExtractor
        dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-15"]))
        t = DatetimeFeatureExtractor(components=["year"], add_cyclic=False)
        result = t.fit_transform(dates)
        assert result.shape[0] == 2

    def test_datetime_extractor_unknown_component(self):
        from backend.data.feature_engineer import DatetimeFeatureExtractor
        dates = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=3)})
        t = DatetimeFeatureExtractor(components=["year", "unknown_comp"], add_cyclic=False)
        result = t.fit_transform(dates)
        assert result.shape == (3, 1)

    def test_datetime_no_cyclic_keys(self):
        from backend.data.feature_engineer import DatetimeFeatureExtractor
        dates = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=3)})
        t = DatetimeFeatureExtractor(components=["year"], add_cyclic=True)
        result = t.fit_transform(dates)
        assert result.shape[0] == 3


# ============================================================
# backend/models/rgf.py
# ============================================================

class TestRgfCoverageGaps:
    def test_rgf_empty_trees(self):
        from backend.models.rgf import RGFRegressor
        rgf = RGFRegressor(n_estimators=0)
        X = np.random.randn(10, 3)
        rgf.trees_ = []
        rgf.leaf_id_maps_ = []
        rgf.leaf_counts_ = []
        rgf.weights_ = np.array([])
        rgf.y_mean_ = 0.0
        Phi = rgf._get_leaf_indicators(X)
        assert Phi.shape == (10, 0)
        preds = rgf._predict_from_weights(X)
        assert np.allclose(preds, 0.0)

    def test_rgf_linalg_error_fallback(self):
        from backend.models.rgf import RGFRegressor
        rgf = RGFRegressor()
        rgf.weights_ = np.zeros(5)
        Phi = np.zeros((10, 5))
        residuals = np.ones(10)
        rgf._update_weights(Phi, residuals, lambda_l2=0.0)


# ============================================================
# backend/optim/constraints.py — InequalityConstraint
# ============================================================

class TestConstraintsCoverageGaps:
    def test_range_constraint_describe_lo_only(self):
        from backend.optim.constraints import RangeConstraint
        c = RangeConstraint(column="x", lo=0.5, hi=None)
        desc = c.describe()
        assert "≥" in desc

    def test_inequality_constraint_le(self):
        from backend.optim.constraints import InequalityConstraint
        c = InequalityConstraint(coefficients={"x": 2.0, "y": 1.0}, rhs=10.0, operator="le")
        # DataFrame入力テスト
        df = pd.DataFrame({"x": [3.0, 5.0], "y": [2.0, 6.0]})
        mask = c.mask(df)
        assert mask.iloc[0] is True or mask.iloc[0] == True  # 2*3+1*2=8 <= 10
        # is_satisfied with Series
        row = pd.Series({"x": 3.0, "y": 2.0})
        assert c.is_satisfied(row)

    def test_inequality_constraint_ge(self):
        from backend.optim.constraints import InequalityConstraint
        c = InequalityConstraint(coefficients={"x": 1.0}, rhs=5.0, operator="ge")
        df = pd.DataFrame({"x": [10.0, 3.0]})
        mask = c.mask(df)
        assert mask.iloc[0]  # 10 >= 5

    def test_inequality_constraint_lt_gt(self):
        from backend.optim.constraints import InequalityConstraint
        c_lt = InequalityConstraint(coefficients={"x": 1.0}, rhs=5.0, operator="lt")
        c_gt = InequalityConstraint(coefficients={"x": 1.0}, rhs=5.0, operator="gt")
        row = pd.Series({"x": 3.0})
        assert c_lt.is_satisfied(row)  # 3 < 5
        assert not c_gt.is_satisfied(row)  # 3 > 5 → False


# ============================================================
# backend/optim/search_space.py — Variable(lo,hi,categories)
# ============================================================

class TestSearchSpaceCoverageGaps:
    def test_variable_continuous(self):
        from backend.optim.search_space import Variable, VarType, SearchSpace
        v = Variable(name="lr", var_type=VarType.CONTINUOUS, lo=1e-5, hi=1.0)
        space = SearchSpace([v])
        cands = space.generate_candidates(method="auto", n_max=10, seed=42)
        assert isinstance(cands, pd.DataFrame)
        assert len(cands) >= 1

    def test_variable_discrete(self):
        from backend.optim.search_space import Variable, VarType
        v = Variable(name="n", var_type=VarType.DISCRETE, lo=1, hi=10, step=1)
        assert v.name == "n"

    def test_variable_categorical(self):
        from backend.optim.search_space import Variable, VarType
        v = Variable(name="act", var_type=VarType.CATEGORICAL, categories=["relu", "tanh"])
        assert "relu" in v.categories


# ============================================================
# backend/optim/bayesian_optimizer.py
# ============================================================

class TestBayesianOptimizerCoverageGaps:
    def test_optimizer_import(self):
        """BayesianOptimizer の BOConfig でインスタンス化"""
        from backend.optim.bayesian_optimizer import BayesianOptimizer
        # BOConfig なしでデフォルト設定
        opt = BayesianOptimizer()
        assert opt is not None


# ============================================================
# backend/optim/bo_visualizer.py — plot_convergence(y_history, objective)
# ============================================================

class TestBOVisualizerCoverageGaps:
    def test_plot_convergence_function(self):
        from backend.optim.bo_visualizer import plot_convergence
        y_history = [0.5, 0.7, 0.9, 0.85, 0.92]
        fig = plot_convergence(y_history, objective="maximize")
        assert fig is not None

    def test_plot_convergence_minimize(self):
        from backend.optim.bo_visualizer import plot_convergence
        y_history = [5.0, 3.0, 2.0, 1.5, 1.0]
        fig = plot_convergence(y_history, objective="minimize")
        assert fig is not None


# ============================================================
# backend/data/benchmark.py — evaluate_regression → ModelScore
# ============================================================

class TestBenchmarkCoverageGaps:
    def test_evaluate_regression(self):
        from backend.data.benchmark import evaluate_regression
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        result = evaluate_regression(y_true, y_pred, model_key="test_model")
        assert result is not None

    def test_evaluate_classification(self):
        from backend.data.benchmark import evaluate_classification
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        result = evaluate_classification(y_true, y_pred, model_key="test_cls")
        assert result is not None


# ============================================================
# backend/models/cv_bias_evaluator.py
# estimate_bbc_cv_bias(oos_predictions, y_true, scoring_func, ...)
# ============================================================

class TestCVBiasEvaluatorCoverageGaps:
    def test_estimate_bbc_cv_bias(self):
        from backend.models.cv_bias_evaluator import estimate_bbc_cv_bias
        from sklearn.metrics import r2_score
        rng = np.random.RandomState(42)
        y_true = rng.randn(50)
        oos = {"model_a": y_true + rng.randn(50) * 0.1}
        result = estimate_bbc_cv_bias(oos, y_true, scoring_func=r2_score)
        assert result is not None

    def test_format_bias_report(self):
        from backend.models.cv_bias_evaluator import format_bias_report, CVBiasResult
        result = CVBiasResult(
            method="bbc",
            raw_score=0.9,
            bias_estimate=0.05,
            corrected_score=0.85,
            n_bootstrap=100,
        )
        report = format_bias_report(result)
        assert isinstance(report, str)


# ============================================================
# backend/pipeline/column_selector.py — ColumnSelectorWrapper(mode, columns)
# ============================================================

class TestColumnSelectorCoverageGaps:
    def test_column_selector_with_columns(self):
        from backend.pipeline.column_selector import ColumnSelectorWrapper
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        cs = ColumnSelectorWrapper(mode="all", columns=["a", "b"])
        cs.fit(df)
        result = cs.transform(df)
        assert result.shape[1] >= 1  # 少なくとも一部が選択される

    def test_column_selector_empty_columns(self):
        from backend.pipeline.column_selector import ColumnSelectorWrapper
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cs = ColumnSelectorWrapper(mode="all", columns=["nonexist"])
        cs.fit(df)
        result = cs.transform(df)
        assert result.shape[0] == 2


# ============================================================
# backend/chem/recommender.py
# ============================================================

class TestRecommenderCoverageGaps:
    def test_get_target_names(self):
        from backend.chem.recommender import get_target_names
        names = get_target_names()
        assert isinstance(names, list)

    def test_get_all_descriptor_categories(self):
        from backend.chem.recommender import get_all_descriptor_categories
        cats = get_all_descriptor_categories()
        assert isinstance(cats, (list, dict))

    def test_get_target_recommendation(self):
        from backend.chem.recommender import get_target_recommendation_by_name, get_target_names
        names = get_target_names()
        if names:
            rec = get_target_recommendation_by_name(names[0])
            assert rec is not None


# ============================================================
# backend/chem/charge_config.py — MoleculeChargeConfig
# ============================================================

class TestChargeConfigCoverageGaps:
    def test_molecule_charge_config_default(self):
        from backend.chem.charge_config import MoleculeChargeConfig
        cfg = MoleculeChargeConfig.default()
        assert cfg.formal_charge == 0

    def test_molecule_charge_config_radical(self):
        from backend.chem.charge_config import MoleculeChargeConfig
        cfg = MoleculeChargeConfig.for_radical()
        assert cfg.spin_multiplicity >= 2


# ============================================================
# backend/chem/group_contrib_adapter.py — GroupContribAdapter
# ============================================================

class TestGroupContribCoverageGaps:
    def test_group_contrib_basic(self):
        from backend.chem.group_contrib_adapter import GroupContribAdapter
        adapter = GroupContribAdapter()
        result = adapter.compute(["CCO", "CC"])
        # result is DescriptorResult, check it has descriptors
        assert hasattr(result, "descriptors")
        assert result.descriptors.shape[0] == 2

    def test_group_contrib_invalid_smiles(self):
        from backend.chem.group_contrib_adapter import GroupContribAdapter
        adapter = GroupContribAdapter()
        result = adapter.compute(["INVALID_XYZ"])
        assert hasattr(result, "descriptors")
        assert result.descriptors.shape[0] == 1


# ============================================================
# backend/interpret/sri.py — SRIDecomposer.decompose(shap_result)
# ============================================================

class TestSRICoverageGaps:
    def test_sri_decomposer(self):
        """SRIDecomposer は ShapResult を入力とする"""
        from backend.interpret.sri import SRIDecomposer
        from backend.interpret.shap_explainer import ShapResult
        rng = np.random.RandomState(42)
        shap_result = ShapResult(
            shap_values=rng.randn(30, 3),
            expected_value=0.0,
            feature_names=["x0", "x1", "x2"],
            X_transformed=rng.randn(30, 3),
            explainer_type="tree",
        )
        dec = SRIDecomposer()
        sri_result = dec.decompose(shap_result)
        assert sri_result is not None

    def test_select_features_by_independence(self):
        from backend.interpret.sri import SRIDecomposer, select_features_by_independence
        from backend.interpret.shap_explainer import ShapResult
        rng = np.random.RandomState(42)
        shap_result = ShapResult(
            shap_values=rng.randn(30, 5),
            expected_value=0.0,
            feature_names=["a", "b", "c", "d", "e"],
            X_transformed=rng.randn(30, 5),
            explainer_type="tree",
        )
        dec = SRIDecomposer()
        sri_result = dec.decompose(shap_result)
        selected = select_features_by_independence(sri_result, top_n=3)
        assert isinstance(selected, list)
        assert len(selected) <= 3


# ============================================================
# backend/utils/optional_import.py — require → RuntimeError
# ============================================================

class TestOptionalImportCoverageGaps:
    def test_require_unavailable(self):
        from backend.utils.optional_import import require
        with pytest.raises(RuntimeError):
            require("totally_fake_module_xyz_9999", feature="test")


# ============================================================
# backend/mlops/mlflow_manager.py
# ============================================================

class TestMlflowCoverageGaps:
    def test_mlflow_manager_import(self):
        from backend.mlops.mlflow_manager import MLflowManager
        mgr = MLflowManager(tracking_uri="file:///tmp/mlruns_test")
        assert mgr is not None
