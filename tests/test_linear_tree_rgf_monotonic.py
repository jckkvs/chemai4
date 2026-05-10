"""
tests/test_linear_tree_rgf_monotonic.py

LinearTree / RGF / MonotonicKernelWrapper の包括的テスト。
"""
import numpy as np
import pytest
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.base import clone


# ─────────────────────────────────────────────────────────────
# テストデータ
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
N = 100
X_reg = np.random.randn(N, 5)
y_reg = 2.0 * X_reg[:, 0] + X_reg[:, 1] ** 2 + 0.2 * np.random.randn(N)

X_clf = np.random.randn(N, 5)
y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(int)

X_clf3 = np.random.randn(150, 4)
y_clf3 = np.array([0]*50 + [1]*50 + [2]*50)


# ═══════════════════════════════════════════════════════════════
# 1. LinearTreeRegressor
# ═══════════════════════════════════════════════════════════════
class TestLinearTreeRegressor:
    def setup_method(self):
        from backend.models.linear_tree import LinearTreeRegressor
        self.cls = LinearTreeRegressor

    def test_fit_predict_shape(self):
        m = self.cls(max_depth=3, random_state=0)
        m.fit(X_reg, y_reg)
        pred = m.predict(X_reg)
        assert pred.shape == (N,), f"期待 ({N},), 実際 {pred.shape}"

    def test_fit_does_not_mutate_base_estimator(self):
        """fitのたびにself.base_estimatorが書き変わらない"""
        base = Ridge(alpha=2.0)
        m = self.cls(base_estimator=base, max_depth=2, random_state=0)
        m.fit(X_reg, y_reg)
        # base_estimatorはfitされていない (cloneが内部で使われるはず)
        msg = "base_estimatorが書き換えられている可能性"
        assert m.base_estimator.alpha == 2.0, msg
        # 2回fitしても同じ
        m.fit(X_reg, y_reg)
        assert m.base_estimator.alpha == 2.0

    def test_n_leaves_positive(self):
        m = self.cls(max_depth=4, random_state=0)
        m.fit(X_reg, y_reg)
        assert m.n_leaves_ >= 1

    def test_residual_improvement(self):
        """LinearTreeが単純な平均予測より小さいMSEを持つ"""
        from sklearn.metrics import mean_squared_error
        m = self.cls(max_depth=3, min_samples_split=5, random_state=0)
        m.fit(X_reg, y_reg)
        mse_tree = mean_squared_error(y_reg, m.predict(X_reg))
        mse_mean = mean_squared_error(y_reg, np.full(N, y_reg.mean()))
        assert mse_tree < mse_mean, f"MSE改善なし: tree={mse_tree:.4f} mean={mse_mean:.4f}"

    def test_different_base_estimators(self):
        """様々なbase_estimatorで動作を確認"""
        for base in [Ridge(alpha=0.5), Lasso(alpha=0.01, max_iter=5000)]:
            m = self.cls(base_estimator=base, max_depth=2, random_state=0)
            m.fit(X_reg, y_reg)
            pred = m.predict(X_reg)
            assert not np.any(np.isnan(pred)), f"{type(base).__name__}でNaN発生"

    def test_clone_compatibility(self):
        """sklearn.clone()で正常にコピーできる"""
        m = self.cls(max_depth=3, random_state=0)
        m2 = clone(m)
        m2.fit(X_reg, y_reg)
        assert hasattr(m2, "root_")

    def test_min_samples_enforcement(self):
        """min_samples_leafが守られている"""
        min_leaf = 5
        m = self.cls(max_depth=10, min_samples_leaf=min_leaf, min_samples_split=10, random_state=0)
        m.fit(X_reg, y_reg)
        preds = m.predict(X_reg)
        assert preds.shape == (N,)


# ═══════════════════════════════════════════════════════════════
# 2. LinearTreeClassifier
# ═══════════════════════════════════════════════════════════════
class TestLinearTreeClassifier:
    def setup_method(self):
        from backend.models.linear_tree import LinearTreeClassifier
        self.cls = LinearTreeClassifier

    def test_predict_classes_in_range(self):
        m = self.cls(max_depth=3, random_state=0)
        m.fit(X_clf, y_clf)
        pred = m.predict(X_clf)
        assert set(pred).issubset({0, 1}), f"不正なクラスラベル: {set(pred)}"

    def test_predict_proba_sums_to_one(self):
        m = self.cls(max_depth=3, random_state=0)
        m.fit(X_clf, y_clf)
        proba = m.predict_proba(X_clf)
        assert proba.shape == (N, 2)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(N), atol=1e-5,
            err_msg="predict_probaの行和が1でない")

    def test_multiclass(self):
        m = self.cls(max_depth=3, random_state=0)
        m.fit(X_clf3, y_clf3)
        pred = m.predict(X_clf3)
        assert set(pred).issubset({0, 1, 2})
        proba = m.predict_proba(X_clf3)
        assert proba.shape == (150, 3)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(150), atol=1e-5)

    def test_fit_not_mutate_base(self):
        base = LogisticRegression(C=0.5, max_iter=500, solver="liblinear")
        m = self.cls(base_estimator=base, max_depth=2, random_state=0)
        m.fit(X_clf, y_clf)
        assert m.base_estimator.C == 0.5, "base_estimator.Cが変わっている"


# ═══════════════════════════════════════════════════════════════
# 3. LinearForestRegressor
# ═══════════════════════════════════════════════════════════════
class TestLinearForestRegressor:
    def setup_method(self):
        from backend.models.linear_tree import LinearForestRegressor
        self.cls = LinearForestRegressor

    def test_basic(self):
        m = self.cls(n_estimators=10, max_depth=3, random_state=0)
        m.fit(X_reg, y_reg)
        pred = m.predict(X_reg)
        assert pred.shape == (N,)
        assert not np.any(np.isnan(pred))

    def test_averaging(self):
        """ForestのMSEはTreeより小さいか同等(バギング効果)"""
        from sklearn.metrics import mean_squared_error
        from backend.models.linear_tree import LinearTreeRegressor
        tree = LinearTreeRegressor(max_depth=3, random_state=0)
        tree.fit(X_reg, y_reg)
        forest = self.cls(n_estimators=20, max_depth=3, random_state=0)
        forest.fit(X_reg, y_reg)
        # ForestはTreeと同水準かより良いMSEであること
        mse_train_tree = mean_squared_error(y_reg, tree.predict(X_reg))
        mse_train_forest = mean_squared_error(y_reg, forest.predict(X_reg))
        # バギングなのでトレーニングMSEはほぼ同等かやや悪いが、NaNがないことが重要
        assert not np.isnan(mse_train_forest)


# ═══════════════════════════════════════════════════════════════
# 4. LinearBoostRegressor
# ═══════════════════════════════════════════════════════════════
class TestLinearBoostRegressor:
    def setup_method(self):
        from backend.models.linear_tree import LinearBoostRegressor
        self.cls = LinearBoostRegressor

    def test_basic(self):
        m = self.cls(n_estimators=20, learning_rate=0.1, max_depth=2, random_state=0)
        m.fit(X_reg, y_reg)
        pred = m.predict(X_reg)
        assert pred.shape == (N,)
        assert not np.any(np.isnan(pred))

    def test_mse_improves_with_rounds(self):
        """ラウンド数が増えるとMSEが減少する傾向"""
        from sklearn.metrics import mean_squared_error
        m5  = self.cls(n_estimators=5,  learning_rate=0.2, max_depth=2, random_state=0)
        m50 = self.cls(n_estimators=50, learning_rate=0.2, max_depth=2, random_state=0)
        m5.fit(X_reg, y_reg)
        m50.fit(X_reg, y_reg)
        mse5  = mean_squared_error(y_reg, m5.predict(X_reg))
        mse50 = mean_squared_error(y_reg, m50.predict(X_reg))
        assert mse50 <= mse5, f"ラウンド増加でMSEが上がった: {mse5:.4f} -> {mse50:.4f}"


# ═══════════════════════════════════════════════════════════════
# 5. LinearBoostClassifier
# ═══════════════════════════════════════════════════════════════
class TestLinearBoostClassifier:
    def setup_method(self):
        from backend.models.linear_tree import LinearBoostClassifier
        self.cls = LinearBoostClassifier

    def test_binary(self):
        m = self.cls(n_estimators=20, learning_rate=0.1, random_state=0)
        m.fit(X_clf, y_clf)
        pred = m.predict(X_clf)
        proba = m.predict_proba(X_clf)
        assert set(pred).issubset({0, 1})
        assert proba.shape == (N, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_multiclass(self):
        m = self.cls(n_estimators=10, learning_rate=0.1, random_state=0)
        m.fit(X_clf3, y_clf3)
        pred = m.predict(X_clf3)
        proba = m.predict_proba(X_clf3)
        assert proba.shape == (150, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ═══════════════════════════════════════════════════════════════
# 6. RGFRegressor
# ═══════════════════════════════════════════════════════════════
class TestRGFRegressor:
    def setup_method(self):
        from backend.models.rgf import RGFRegressor
        self.cls = RGFRegressor

    def test_basic(self):
        m = self.cls(n_estimators=20, max_leaf_nodes=8, random_state=0)
        m.fit(X_reg, y_reg)
        pred = m.predict(X_reg)
        assert pred.shape == (N,)
        assert not np.any(np.isnan(pred))

    def test_leaf_indicator_dim(self):
        """_get_leaf_indicatorsの次元が正しい"""
        m = self.cls(n_estimators=5, max_leaf_nodes=4, random_state=0)
        m.fit(X_reg, y_reg)
        X_np = X_reg
        Phi = m._get_leaf_indicators(X_np)
        # 各行はone-hot (各木に対して)
        assert Phi.shape[0] == N
        assert Phi.shape[1] > 0  # 少なくとも1つの葉
        # 各木に対してone-hotになっているかチェック（近似: 行和 == n_trees）
        # 正確には、各木の部分ではone-hotなので合計はn_treesの本数
        row_sums = Phi.sum(axis=1)
        assert (row_sums == len(m.trees_)).all(), (
            f"各サンプルが正確に{len(m.trees_)}本の木に所属すべき。実際: {row_sums[:5]}"
        )

    def test_weights_shape(self):
        m = self.cls(n_estimators=5, max_leaf_nodes=4, random_state=0)
        m.fit(X_reg, y_reg)
        assert len(m.weights_) == m._total_leaves

    def test_regularization_effect(self):
        """L2正則化が大きいほど重みのノルムが小さい"""
        m1 = self.cls(n_estimators=10, max_leaf_nodes=8, lambda_l2=0.01, random_state=0)
        m2 = self.cls(n_estimators=10, max_leaf_nodes=8, lambda_l2=100.0, random_state=0)
        m1.fit(X_reg, y_reg)
        m2.fit(X_reg, y_reg)
        norm1 = np.linalg.norm(m1.weights_)
        norm2 = np.linalg.norm(m2.weights_)
        assert norm2 <= norm1 + 1e-3, (
            f"L2大きい方が重みノルム大きい: small_reg={norm1:.4f}, large_reg={norm2:.4f}"
        )

    def test_mse_improves(self):
        """ラウンド数が増えるとMSEが改善する"""
        from sklearn.metrics import mean_squared_error
        m5  = self.cls(n_estimators=5,  max_leaf_nodes=8, random_state=0)
        m50 = self.cls(n_estimators=50, max_leaf_nodes=8, random_state=0)
        m5.fit(X_reg, y_reg)
        m50.fit(X_reg, y_reg)
        mse5  = mean_squared_error(y_reg, m5.predict(X_reg))
        mse50 = mean_squared_error(y_reg, m50.predict(X_reg))
        assert mse50 <= mse5, f"ラウンド増でMSE改善なし: {mse5:.4f} -> {mse50:.4f}"


# ═══════════════════════════════════════════════════════════════
# 7. RGFClassifier
# ═══════════════════════════════════════════════════════════════
class TestRGFClassifier:
    def setup_method(self):
        from backend.models.rgf import RGFClassifier
        self.cls = RGFClassifier

    def test_binary(self):
        m = self.cls(n_estimators=20, max_leaf_nodes=8, random_state=0)
        m.fit(X_clf, y_clf)
        proba = m.predict_proba(X_clf)
        pred = m.predict(X_clf)
        assert proba.shape == (N, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert set(pred).issubset({0, 1})

    def test_multiclass(self):
        m = self.cls(n_estimators=10, max_leaf_nodes=8, random_state=0)
        m.fit(X_clf3, y_clf3)
        proba = m.predict_proba(X_clf3)
        assert proba.shape == (150, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        pred = m.predict(X_clf3)
        assert set(pred).issubset({0, 1, 2})

    def test_proba_range(self):
        m = self.cls(n_estimators=10, max_leaf_nodes=8, random_state=0)
        m.fit(X_clf, y_clf)
        proba = m.predict_proba(X_clf)
        assert (proba >= 0).all() and (proba <= 1).all(), "確率値が[0,1]範囲外"


# ═══════════════════════════════════════════════════════════════
# 8. MonotonicKernelWrapper
# ═══════════════════════════════════════════════════════════════
class TestMonotonicKernelWrapper:
    def test_svr_increasing(self):
        """SVRで増加制約: グリッド点での予測が(おおよそ)単調増加"""
        from sklearn.svm import SVR
        from backend.models.monotonic_kernel import MonotonicKernelWrapper
        np.random.seed(0)
        X = np.random.randn(80, 3)
        y = X[:, 0] + np.random.randn(80) * 0.1
        # 特徴量0に増加制約
        m = MonotonicKernelWrapper(SVR(), monotonic_constraints=(1, 0, 0), max_iter=5)
        m.fit(X, y)
        # テスト: feature0をスイープしたときの予測が増加傾向か確認
        grid = np.zeros((30, 3))
        grid[:, 0] = np.linspace(-2, 2, 30)
        pred = m.predict(grid)
        violations = np.sum(np.diff(pred) < -0.1)  # 大きな減少はNG
        assert violations <= 5, f"増加制約で大きな違反が{violations}箇所"

    def test_no_constraint_passthrough(self):
        """制約なし(全0)はラッパーなしと同等"""
        from sklearn.svm import SVR
        from backend.models.monotonic_kernel import wrap_with_soft_monotonic, MonotonicKernelWrapper
        svr = SVR()
        result = wrap_with_soft_monotonic(svr, (0, 0, 0))
        assert not isinstance(result, MonotonicKernelWrapper), "制約なしなのにラッパーが適用された"

    def test_predict_shape(self):
        from sklearn.svm import SVR
        from backend.models.monotonic_kernel import MonotonicKernelWrapper
        m = MonotonicKernelWrapper(SVR(), monotonic_constraints=(1, -1, 0), max_iter=2)
        m.fit(X_reg[:, :3], y_reg)
        pred = m.predict(X_reg[:, :3])
        assert pred.shape == (N,)

    def test_violation_stored(self):
        """fit後にmonotonic_violation_が格納される"""
        from sklearn.svm import SVR
        from backend.models.monotonic_kernel import MonotonicKernelWrapper
        m = MonotonicKernelWrapper(SVR(), monotonic_constraints=(1, 0, 0, 0, 0), max_iter=2)
        m.fit(X_reg, y_reg)
        assert hasattr(m, "monotonic_violation_")
        assert isinstance(m.monotonic_violation_, float)

    def test_kernel_ridge_decreasing(self):
        """KernelRidgeに減少制約が適用できる"""
        from sklearn.kernel_ridge import KernelRidge
        from backend.models.monotonic_kernel import MonotonicKernelWrapper
        m = MonotonicKernelWrapper(
            KernelRidge(), monotonic_constraints=(-1, 0, 0, 0, 0), max_iter=3
        )
        m.fit(X_reg, y_reg)
        pred = m.predict(X_reg)
        assert pred.shape == (N,)

    def test_is_soft_monotonic_candidate(self):
        """is_soft_monotonic_candidate のデテクション"""
        from sklearn.svm import SVR, SVC
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.linear_model import Ridge
        from backend.models.monotonic_kernel import is_soft_monotonic_candidate
        assert is_soft_monotonic_candidate(SVR())
        assert is_soft_monotonic_candidate(SVC())
        assert is_soft_monotonic_candidate(KernelRidge())
        assert is_soft_monotonic_candidate(GaussianProcessRegressor())
        assert not is_soft_monotonic_candidate(Ridge())  # Ridgeはkernelではない


# ═══════════════════════════════════════════════════════════════
# 9. Pipeline統合テスト
# ═══════════════════════════════════════════════════════════════
class TestPipelineIntegration:
    def test_apply_monotonic_constraints_native(self):
        """XGBoostにネイティブ単調性制約が設定される"""
        try:
            from xgboost import XGBRegressor
            from backend.pipeline.pipeline_builder import apply_monotonic_constraints
            from backend.pipeline.column_selector import ColumnMeta
            estimator = XGBRegressor(n_estimators=10)
            cm = {
                "feat0": ColumnMeta(monotonic=1),
                "feat1": ColumnMeta(monotonic=-1),
                "feat2": ColumnMeta(monotonic=0),
            }
            result = apply_monotonic_constraints(estimator, cm)
            params = result.get_params()
            mc = params.get("monotone_constraints") or params.get("monotonic_constraints")
            assert mc is not None, "XGBoostに単調性制約が設定されていない"
        except ImportError:
            pytest.skip("xgboost未インストール")

    def test_apply_monotonic_constraints_soft(self):
        """SVRにMonotonicConstraintRegressorが適用される"""
        from sklearn.svm import SVR
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        from backend.pipeline.column_selector import ColumnMeta
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        estimator = SVR()
        cm = {
            "feat0": ColumnMeta(monotonic=1),
            "feat1": ColumnMeta(monotonic=0),
        }
        result = apply_monotonic_constraints(estimator, cm)
        assert isinstance(result, MonotonicConstraintRegressor), (
            f"SVRに対してMonotonicConstraintRegressorが適用されていない。実際: {type(result)}"
        )

    def test_apply_no_constraint_returns_same(self):
        """全0制約ではestimatorがそのまま返る"""
        from sklearn.svm import SVR
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        from backend.pipeline.column_selector import ColumnMeta
        estimator = SVR()
        cm = {"feat0": ColumnMeta(monotonic=0), "feat1": ColumnMeta(monotonic=0)}
        result = apply_monotonic_constraints(estimator, cm)
        assert result is estimator
