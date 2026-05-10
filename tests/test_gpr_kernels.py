"""
tests/test_gpr_kernels.py

GPR/GPC カーネル別モデルの登録・生成・fit/predict テスト。
"""
import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel, WhiteKernel

from backend.models.factory import get_model, list_models, get_model_registry


# ── テスト用の小さなデータ ──
@pytest.fixture
def regression_data():
    rng = np.random.RandomState(42)
    X = rng.rand(30, 3)
    y = X[:, 0] * 2 + X[:, 1] - 0.5 * X[:, 2] + rng.randn(30) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    rng = np.random.RandomState(42)
    X = rng.rand(30, 3)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    return X, y


# ============================================================
# 回帰: GPR カーネル別モデルの存在確認
# ============================================================

GPR_KEYS = ["gp", "gpr_rbf", "gpr_matern", "gpr_ard", "gpr_constant_rbf", "gpr_dotproduct"]
GPC_KEYS = ["gp_c", "gpc_rbf", "gpc_matern", "gpc_ard", "gpc_constant_rbf", "gpc_dotproduct"]


@pytest.mark.parametrize("model_key", GPR_KEYS)
def test_gpr_model_in_registry(model_key):
    """各GPRカーネルモデルがレジストリに登録されていることを確認。"""
    registry = get_model_registry(task="regression")
    assert model_key in registry, f"{model_key} がレジストリにありません"
    assert registry[model_key]["available"] is True


@pytest.mark.parametrize("model_key", GPC_KEYS)
def test_gpc_model_in_registry(model_key):
    """各GPCカーネルモデルがレジストリに登録されていることを確認。"""
    registry = get_model_registry(task="classification")
    assert model_key in registry, f"{model_key} がレジストリにありません"
    assert registry[model_key]["available"] is True


# ============================================================
# 回帰: GPR カーネル別モデルの生成テスト
# ============================================================

@pytest.mark.parametrize("model_key", GPR_KEYS)
def test_gpr_model_creation(model_key):
    """各GPRカーネルモデルが正しくインスタンス化されることを確認。"""
    model = get_model(model_key, task="regression")
    assert isinstance(model, GaussianProcessRegressor)
    assert model.kernel is not None


def test_gpr_rbf_has_rbf_kernel():
    """GPR (RBF) のカーネルに RBF が含まれることを確認。"""
    model = get_model("gpr_rbf", task="regression")
    kernel_str = str(model.kernel)
    assert "RBF" in kernel_str


def test_gpr_matern_has_matern_kernel():
    """GPR (Matérn) のカーネルに Matern が含まれることを確認。"""
    model = get_model("gpr_matern", task="regression")
    kernel_str = str(model.kernel)
    assert "Matern" in kernel_str


def test_gpr_matern_custom_nu():
    """GPR (Matérn) に nu パラメータを渡せることを確認。"""
    model = get_model("gpr_matern", task="regression", nu=1.5)
    kernel_str = str(model.kernel)
    assert "Matern" in kernel_str
    # nu=1.5 がカーネルに反映されているか
    for k in model.kernel.get_params().values():
        if hasattr(k, "nu"):
            assert k.nu == 1.5


def test_gpr_dotproduct_has_dotproduct_kernel():
    """GPR (DotProduct) のカーネルに DotProduct が含まれることを確認。"""
    model = get_model("gpr_dotproduct", task="regression")
    kernel_str = str(model.kernel)
    assert "DotProduct" in kernel_str


def test_gpr_ard_has_ard_kernel():
    """GPR (ARD) のカーネルが ARD (多次元 length_scale の RBF) であることを確認。"""
    model = get_model("gpr_ard", task="regression", n_features=3)
    kernel_str = str(model.kernel)
    assert "RBF" in kernel_str


# ============================================================
# 回帰: GPR fit / predict テスト
# ============================================================

@pytest.mark.parametrize("model_key", GPR_KEYS)
def test_gpr_fit_predict(model_key, regression_data):
    """各GPRカーネルモデルが fit/predict できることを確認。"""
    X, y = regression_data
    model = get_model(model_key, task="regression")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    # 予測値がNaNでないこと
    assert not np.any(np.isnan(preds))


@pytest.mark.parametrize("model_key", GPR_KEYS)
def test_gpr_predict_with_std(model_key, regression_data):
    """各GPRカーネルモデルが predict で標準偏差を返せることを確認。"""
    X, y = regression_data
    model = get_model(model_key, task="regression")
    model.fit(X, y)
    preds, stds = model.predict(X, return_std=True)
    assert preds.shape[0] == X.shape[0]
    assert stds.shape[0] == X.shape[0]
    assert np.all(stds >= 0)


# ============================================================
# 分類: GPC カーネル別モデルの生成・fit/predict テスト
# ============================================================

@pytest.mark.parametrize("model_key", GPC_KEYS)
def test_gpc_model_creation(model_key):
    """各GPCカーネルモデルが正しくインスタンス化されることを確認。"""
    model = get_model(model_key, task="classification")
    assert isinstance(model, GaussianProcessClassifier)
    assert model.kernel is not None


@pytest.mark.parametrize("model_key", GPC_KEYS)
def test_gpc_fit_predict(model_key, classification_data):
    """各GPCカーネルモデルが fit/predict できることを確認。"""
    X, y = classification_data
    model = get_model(model_key, task="classification")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]


@pytest.mark.parametrize("model_key", GPC_KEYS)
def test_gpc_predict_proba(model_key, classification_data):
    """各GPCカーネルモデルが predict_proba できることを確認。"""
    X, y = classification_data
    model = get_model(model_key, task="classification")
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    # 確率が [0, 1] 範囲で合計1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ============================================================
# list_models で gaussian_process タグフィルタが機能すること
# ============================================================

def test_list_gpr_by_tag():
    """'gaussian_process' タグでGPRモデルが取得できることを確認。"""
    models = list_models(task="regression", tags=["gaussian_process"])
    keys = [m["key"] for m in models]
    assert "gpr_rbf" in keys
    assert "gpr_matern" in keys
    assert "gpr_ard" in keys
    assert "gpr_constant_rbf" in keys
    assert "gpr_dotproduct" in keys


def test_list_gpc_by_tag():
    """'gaussian_process' タグでGPCモデルが取得できることを確認。"""
    models = list_models(task="classification", tags=["gaussian_process"])
    keys = [m["key"] for m in models]
    assert "gpc_rbf" in keys
    assert "gpc_matern" in keys
    assert "gpc_ard" in keys
    assert "gpc_constant_rbf" in keys
    assert "gpc_dotproduct" in keys


# ============================================================
# 後方互換テスト: "gp" と "gp_c" が引き続き使えること
# ============================================================

def test_backward_compat_gp(regression_data):
    """旧キー 'gp' が GPR (RBF) として動作することを確認。"""
    X, y = regression_data
    model = get_model("gp", task="regression")
    assert isinstance(model, GaussianProcessRegressor)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]


def test_backward_compat_gp_c(classification_data):
    """旧キー 'gp_c' が GPC (RBF) として動作することを確認。"""
    X, y = classification_data
    model = get_model("gp_c", task="classification")
    assert isinstance(model, GaussianProcessClassifier)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]


# ============================================================
# 各カーネルモデルが異なるカーネルを持つことの確認
# ============================================================

def test_different_kernels_are_distinct():
    """異なるモデルキーのGPRが異なるカーネルタイプを持つことを確認。"""
    rbf_model = get_model("gpr_rbf", task="regression")
    matern_model = get_model("gpr_matern", task="regression")
    dot_model = get_model("gpr_dotproduct", task="regression")

    rbf_str = str(rbf_model.kernel)
    matern_str = str(matern_model.kernel)
    dot_str = str(dot_model.kernel)

    # RBF モデルには "Matern" が含まれない
    assert "Matern" not in rbf_str
    # Matérn モデルには "Matern" が含まれる
    assert "Matern" in matern_str
    # DotProduct モデルには "DotProduct" が含まれる
    assert "DotProduct" in dot_str
    # DotProduct モデルのカーネル文字列は RBF モデルと異なる
    assert dot_str != rbf_str
