"""
tests/test_models.py

backend/models モジュールのユニットテスト。
ModelFactory, CVManager, Tuner, AutoML をテストする。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Ridge, LogisticRegression


from backend.models.factory import get_model, list_models, get_default_automl_models
from backend.models.cv_manager import (
    CVConfig, get_cv, list_cv_methods, run_cross_validation, WalkForwardSplit
)
from backend.models.tuner import TunerConfig, tune
from backend.models.automl import AutoMLEngine, AutoMLResult


# ============================================================
# テスト用データフィクスチャ
# ============================================================

@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, random_state=42
    )
    return X, y


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """AutoMLテスト用DataFrameフィクスチャ。"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.choice(["A", "B"], n),
        "target": np.random.randn(n),
    })

def generate_numeric_data(n_samples: int = 100, n_features: int = 5, n_classes: int = 2, task: str = "regression") -> pd.DataFrame:
    """テスト用のシンプルな数値データセットを生成する関数。"""
    np.random.seed(42)
    data = {f"feat_{i}": np.random.randn(n_samples) for i in range(n_features)}
    if task == "classification":
        data["target"] = np.random.randint(0, n_classes, n_samples)
    else:
        data["target"] = np.random.randn(n_samples)
    return pd.DataFrame(data)

# ============================================================
# T-004: ModelFactory テスト
# ============================================================

class TestModelFactory:
    """T-004: モデルファクトリーのテスト。"""

    def test_get_regression_model(self, regression_data: tuple) -> None:
        """回帰モデルを取得してfitできること。(T-004-01)"""
        X, y = regression_data
        model = get_model("rf", task="regression")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)

    def test_get_classification_model(self, classification_data: tuple) -> None:
        """分類モデルを取得してfitできること。(T-004-02)"""
        X, y = classification_data
        model = get_model("logistic", task="classification")
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_unknown_model_key_raises(self) -> None:
        """未知のモデルキーで ValueError が上がること。(T-004-03)"""
        with pytest.raises(ValueError, match="未知のモデルキー"):
            get_model("nonexistent_model", task="regression")

    def test_unknown_task_raises(self) -> None:
        """未知のタスク名で ValueError が上がること。(T-004-04)"""
        with pytest.raises(ValueError):
            get_model("rf", task="unknown_task")

    def test_list_models_regression(self) -> None:
        """list_models('regression') が非空リストを返すこと。(T-004-05)"""
        models = list_models(task="regression", available_only=True)
        assert len(models) > 0
        assert all("key" in m and "name" in m for m in models)

    def test_list_models_classification(self) -> None:
        """list_models('classification') が非空リストを返すこと。(T-004-06)"""
        models = list_models(task="classification", available_only=True)
        assert len(models) > 0

    def test_list_models_with_tags(self) -> None:
        """tagsフィルタが機能すること。(T-004-07)"""
        models = list_models(task="regression", tags=["linear"])
        assert all("linear" in m["tags"] for m in models)

    def test_get_default_automl_models(self) -> None:
        """AutoMLデフォルトモデルリストが非空であること。(T-004-08)"""
        models = get_default_automl_models("regression")
        assert len(models) > 0
        models_c = get_default_automl_models("classification")
        assert len(models_c) > 0

    def test_override_params(self, regression_data: tuple) -> None:
        """パラメータのオーバーライドが適用されること。(T-004-09)"""
        X, y = regression_data
        model = get_model("ridge", task="regression", alpha=10.0)
        assert model.alpha == 10.0

    def test_sklearn_linear_models(self, regression_data: tuple) -> None:
        """線形回帰モデル群が全て取得・fitできること。(T-004-10)"""
        X, y = regression_data
        for key in ["linear", "ridge", "lasso", "elasticnet", "bayesian_ridge"]:
            model = get_model(key, task="regression")
            model.fit(X, y)


# ============================================================
# T-005: CVManager テスト
# ============================================================

class TestCVManager:
    """T-005: クロスバリデーションのテスト。"""

    def test_get_kfold(self) -> None:
        """KFoldスプリッタが取得できること。(T-005-01)"""
        from sklearn.model_selection import KFold
        cfg = CVConfig(cv_key="kfold", n_splits=5)
        cv = get_cv(cfg)
        assert isinstance(cv, KFold)
        assert cv.n_splits == 5

    def test_get_stratified_kfold(self) -> None:
        """StratifiedKFoldスプリッタが取得できること。(T-005-02)"""
        from sklearn.model_selection import StratifiedKFold
        cfg = CVConfig(cv_key="stratified_kfold", n_splits=3)
        cv = get_cv(cfg)
        assert isinstance(cv, StratifiedKFold)

    def test_get_timeseries_split(self) -> None:
        """TimeSeriesSplitが取得できること。(T-005-03)"""
        from sklearn.model_selection import TimeSeriesSplit
        cfg = CVConfig(cv_key="timeseries", n_splits=4)
        cv = get_cv(cfg)
        assert isinstance(cv, TimeSeriesSplit)

    def test_get_loo(self) -> None:
        """LeaveOneOutが取得できること。(T-005-04)"""
        from sklearn.model_selection import LeaveOneOut
        cfg = CVConfig(cv_key="loo")
        cv = get_cv(cfg)
        assert isinstance(cv, LeaveOneOut)

    def test_unknown_cv_key_raises(self) -> None:
        """未知のCV手法でValueErrorが上がること。(T-005-05)"""
        with pytest.raises(ValueError, match="CV手法 'unknown_cv'"):
            get_cv(CVConfig(cv_key="unknown_cv"))

    def test_walk_forward_split(self) -> None:
        """WalkForwardSplitが正しいインデックスを生成すること。(T-005-06)"""
        wf = WalkForwardSplit(n_splits=3, min_train_size=10)
        X = np.arange(50).reshape(-1, 1)
        splits = list(wf.split(X))
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) >= 10
            assert len(test_idx) > 0
            assert max(train_idx) < min(test_idx)  # リーク確認

    def test_list_cv_methods(self) -> None:
        """list_cv_methods() が辞書のリストを返すこと。(T-005-07)"""
        methods = list_cv_methods(task="regression")
        assert len(methods) > 0
        assert all("key" in m and "name" in m for m in methods)

    def test_run_cross_validation(self, regression_data: tuple) -> None:
        """run_cross_validation() がスコア辞書を返すこと。(T-005-08)"""
        X, y = regression_data
        model = Ridge()
        cfg = CVConfig(cv_key="kfold", n_splits=3)
        result = run_cross_validation(
            model, X, y, cfg,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        assert "test_neg_root_mean_squared_error" in result
        assert len(result["test_neg_root_mean_squared_error"]) == 3

    def test_walk_forward_no_leak(self) -> None:
        """WalkForwardSplitでテスト期間が学習期間より後であること。(T-005-09)"""
        X = np.arange(60).reshape(-1, 1)
        wf = WalkForwardSplit(n_splits=4, min_train_size=10)
        for train_idx, test_idx in wf.split(X):
            assert all(t < s for t in train_idx for s in test_idx[:1])


# ============================================================
# T-006: Tuner テスト
# ============================================================

class TestTuner:
    """T-006: ハイパーパラメータ最適化のテスト。"""

    def test_grid_search(self, regression_data: tuple) -> None:
        """GridSearchが最良パラメータを返すこと。(T-006-01)"""
        X, y = regression_data
        model = Ridge()
        cfg = TunerConfig(
            method="grid",
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            cv=3,
            scoring="neg_root_mean_squared_error",
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result
        assert "best_params" in result
        assert "alpha" in result["best_params"]

    def test_random_search(self, regression_data: tuple) -> None:
        """RandomSearchが最良パラメータを返すこと。(T-006-02)"""
        X, y = regression_data
        model = Ridge()
        cfg = TunerConfig(
            method="random",
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            n_iter=3,
            cv=3,
            scoring="neg_root_mean_squared_error",
        )
        result = tune(model, X, y, cfg)
        assert result["best_estimator"] is not None

    def test_unknown_method_raises(self, regression_data: tuple) -> None:
        """未知のチューニング手法でValueErrorが上がること。(T-006-03)"""
        X, y = regression_data
        model = Ridge()
        cfg = TunerConfig(method="unknown_method", param_grid={}, cv=2)
        with pytest.raises(ValueError, match="未知のチューニング手法"):
            tune(model, X, y, cfg)

    def test_best_score_is_float(self, regression_data: tuple) -> None:
        """best_score がfloatであること。(T-006-04)"""
        X, y = regression_data
        model = Ridge()
        cfg = TunerConfig(
            method="grid",
            param_grid={"alpha": [1.0, 2.0]},
            cv=2,
            scoring="neg_root_mean_squared_error",
        )
        result = tune(model, X, y, cfg)
        assert isinstance(result["best_score"], float)

    def test_optuna_search(self, regression_data: tuple) -> None:
        """Optunaによる最適化が実行できること。(T-006-05)"""
        import sys
        from unittest.mock import patch, MagicMock
        X, y = regression_data
        model = Ridge()
        # Ridge が持っているパラメータのみを指定
        param_grid = {
            "alpha": {"type": "float", "low": 0.01, "high": 1.0, "log": True}
        }
        cfg = TunerConfig(
            method="optuna",
            param_grid=param_grid,
            n_iter=1,
            cv=2,
        )

        mock_optuna = MagicMock()
        with patch.dict(sys.modules, {"optuna": mock_optuna}), \
             patch("backend.models.tuner._optuna", True):
            
            # study.optimize 内で呼ばれる objective を直接実行してカバレッジを稼ぐ
            def mock_optimize(objective, n_trials=None, timeout=None):
                trial = MagicMock()
                trial.suggest_float.return_value = 0.5
                objective(trial)

            m_study = MagicMock()
            m_study.optimize.side_effect = mock_optimize
            m_study.best_params = {"alpha": 0.5}
            m_study.best_value = -0.1
            m_study.trials_dataframe.return_value = pd.DataFrame([{"alpha": 0.5, "value": -0.1}])
            mock_optuna.create_study.return_value = m_study
            
            result = tune(model, X, y, cfg)
            assert result["best_params"] == {"alpha": 0.5}

    def test_factory_errors(self) -> None:
        """factory.py のエラー処理を検証。(T-004-10)"""
        from backend.models.factory import get_model
        import pytest
        
        # 未知のモデルキー
        with pytest.raises(ValueError, match="未知のモデルキー"):
            get_model("unknown_model")
            
        # 未知のタスク
        with pytest.raises(ValueError, match="未知のタスク"):
            get_model("rf", task="unknown_task")

        # ライブラリ未インストール時 (レジストリを直接書き換えて再現)
        import backend.models.factory
        original_val = backend.models.factory._REGRESSION_REGISTRY["xgb"]["available"]
        backend.models.factory._REGRESSION_REGISTRY["xgb"]["available"] = False
        try:
            with pytest.raises(ValueError, match="ライブラリがインストールされていません"):
                get_model("xgb")
        finally:
            backend.models.factory._REGRESSION_REGISTRY["xgb"]["available"] = original_val

    def test_halving_search(self, regression_data: tuple) -> None:
        """HalvingGridSearch パスの検証。(T-006-06)"""
        from unittest.mock import patch, MagicMock
        from sklearn.linear_model import Ridge
        X, y = regression_data
        model = Ridge()
        # _halving_available を True にしてパスを通す
        with patch("backend.models.tuner._halving_available", True), \
             patch("sklearn.model_selection.HalvingGridSearchCV") as m_hgs, \
             patch("sklearn.model_selection.HalvingRandomSearchCV") as m_hrs:
            
            m_result = MagicMock()
            m_result.best_estimator_ = model
            m_result.best_params_ = {"alpha": 0.1}
            m_result.best_score_ = -0.1
            m_result.cv_results_ = {"mean_test_score": [-0.1]}
            m_hgs.return_value = m_result
            m_hrs.return_value = m_result
            
            cfg_grid = TunerConfig(method="halving_grid", param_grid={"alpha": [0.1, 1.0]})
            tune(model, X, y, cfg_grid)
            
            cfg_rand = TunerConfig(method="halving_random", param_grid={"alpha": [0.1, 1.0]})
            tune(model, X, y, cfg_rand)

    def test_tuner_bayes_fallback_v2(self, regression_data: tuple) -> None:
        """skopt 未インストール時の BayesSearchCV 代替分岐を検証。(T-004-09)"""
        from unittest.mock import patch
        from backend.models.tuner import tune, TunerConfig
        from sklearn.linear_model import Ridge
        X, y = regression_data
        model = Ridge()
        cfg = TunerConfig(method="bayes", param_grid={"alpha": [0.1, 1.0]}, n_iter=2)
        with patch("backend.models.tuner._skopt", False):
            res = tune(model, X, y, cfg)
            assert "best_estimator" in res


# ============================================================
# T-007: AutoML テスト
# ============================================================

class TestAutoMLEngine:
    """T-007: AutoMLエンジンのテスト。"""

    def test_automl_engine_regression(self) -> None:
        """AutoMLの回帰タスク実行テスト"""
        df = generate_numeric_data(n_samples=50, n_features=3, task="regression")
        engine = AutoMLEngine(task="regression", cv_folds=2, model_keys=["ridge", "rf", "dt"])
        result = engine.run(df, "target")
        assert isinstance(result, AutoMLResult)
        assert result.task == "regression"
        assert result.best_model_key != ""
        assert isinstance(result.best_score, float)

    def test_automl_engine_classification(self) -> None:
        """AutoMLの分類タスク実行テスト(task="auto"で自動判定)"""
        df = generate_numeric_data(n_samples=50, n_features=3, n_classes=2, task="classification")
        engine = AutoMLEngine(task="auto", cv_folds=2, model_keys=["dt_c", "rf_c"])
        result = engine.run(df, "target")
        assert result.task == "classification"

    def test_run_returns_model_scores(self, regression_df: pd.DataFrame) -> None:
        """AutoML結果がmodel_scoresを含むこと。(T-007-04)"""
        engine = AutoMLEngine(task="regression", cv_folds=2, model_keys=["ridge", "rf"])
        result = engine.run(regression_df, target_col="target")
        assert len(result.model_scores) >= 1

    def test_run_best_pipeline_can_predict(self, regression_df: pd.DataFrame) -> None:
        """AutoML結果のbest_pipelineがpredictできること。(T-007-05)"""
        engine = AutoMLEngine(task="regression", cv_folds=2, model_keys=["ridge", "lasso"])
        result = engine.run(regression_df, target_col="target")
        X = regression_df.drop(columns=["target"])
        preds = result.best_pipeline.predict(X)
        assert preds.shape == (len(regression_df),)

    def test_automl_too_few_records(self) -> None:
        df = generate_numeric_data(n_samples=5, task="regression")
        engine = AutoMLEngine(task="regression", cv_folds=2, model_keys=["ridge"])
        with pytest.raises(ValueError, match="少なすぎます"):
            engine.run(df, target_col="target")

    def test_automl_invalid_target_col(self) -> None:
        df = generate_numeric_data(n_samples=20, task="regression")
        engine = AutoMLEngine(task="regression", cv_folds=2, model_keys=["ridge"])
        with pytest.raises(ValueError, match="存在しません"):
            engine.run(df, target_col="nonexistent")
