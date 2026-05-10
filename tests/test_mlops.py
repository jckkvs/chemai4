"""
tests/test_mlops.py

backend/mlops モジュールのユニットテスト。
MLflowManager をテストする。
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from backend.mlops.mlflow_manager import MLflowManager, MLRunContext

@pytest.fixture
def mock_mlflow():
    with patch("backend.mlops.mlflow_manager._mlflow", True):
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.create_experiment"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run") as m_start, \
             patch("mlflow.end_run") as m_end, \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.log_artifact"), \
             patch("mlflow.sklearn.log_model"), \
             patch("mlflow.sklearn.load_model", return_value=MagicMock()), \
             patch("mlflow.search_runs"), \
             patch("mlflow.get_experiment_by_name", return_value=None):
            m_run = MagicMock()
            m_run.info.run_id = "test_run_id"
            m_start.return_value = m_run
            yield m_start, m_end

@pytest.fixture
def mlflow_manager():
    """MLflowManager のインスタンスを返すフィクスチャ"""
    return MLflowManager(tracking_uri="http://localhost:5000", experiment_name="test_exp")

def test_mlflow_manager_init(mock_mlflow, mlflow_manager) -> None:
    assert mlflow_manager.tracking_uri == "http://localhost:5000"
    assert mlflow_manager.experiment_name == "test_exp"

def test_start_end_run(mock_mlflow, mlflow_manager) -> None:
    run_id = mlflow_manager.start_run(run_name="test_run")
    assert run_id == "test_run_id"
    assert mlflow_manager._run_id == "test_run_id"
    
    mlflow_manager.end_run()
    assert mlflow_manager._run_id is None

def test_log_params_metrics(mock_mlflow, mlflow_manager) -> None:
    mlflow_manager.start_run()
    
    with patch("mlflow.log_params") as m_log_p, \
         patch("mlflow.log_metrics") as m_log_m:
        mlflow_manager.log_params({"p1": 1})
        mlflow_manager.log_metrics({"m1": 0.5})
        
        m_log_p.assert_called_once()
        m_log_m.assert_called_once()

def test_mlrun_context(mock_mlflow, mlflow_manager) -> None:
    with MLRunContext(mlflow_manager, run_name="ctx_run") as run_id:
        assert run_id == "test_run_id"
    
    assert mlflow_manager._run_id is None

def test_run_context_fail(mock_mlflow, mlflow_manager) -> None:
    """コンテキスト内部で例外が起きたとき fail_run が呼ばれること。(T-008-05)"""
    with patch.object(mlflow_manager, "fail_run") as m_fail:
        try:
            with MLRunContext(mlflow_manager):
                raise ValueError("test error")
        except ValueError:
            pass
        m_fail.assert_called_once()

def test_save_and_load_model(mock_mlflow, mlflow_manager, tmp_path: Path) -> None:
    """モデルの保存と読み込みができること。(T-008-06)"""
    from sklearn.linear_model import Ridge
    model = Ridge()
    save_dir = tmp_path / "models"
    
    with patch("mlflow.log_artifact") as m_log_art:
        path = mlflow_manager.save_model(model, "test_ridge", save_dir=save_dir)
        assert path.exists()
        m_log_art.assert_called()
        
        loaded = mlflow_manager.load_model(path)
        assert isinstance(loaded, Ridge)

def test_get_experiment_runs(mock_mlflow, mlflow_manager) -> None:
    """実験結果の検索が呼ばれること。(T-008-07)"""
    with patch("mlflow.search_runs") as m_search:
        m_search.return_value = pd.DataFrame({"run_id": ["r1", "r2"], "metrics.rmse": [0.1, 0.2]})
        df = mlflow_manager.get_experiment_runs()
        assert len(df) == 2
        
        best = mlflow_manager.get_best_run("rmse")
        assert best["run_id"] == "r1"

def test_log_figure(mock_mlflow, mlflow_manager) -> None:
    """図の記録が呼ばれること。(T-008-08)"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    
    with patch("mlflow.log_figure") as m_log:
        mlflow_manager.log_figure(fig, "test.png")
        m_log.assert_called_once()
    plt.close(fig)

def test_best_run_none(mock_mlflow, mlflow_manager) -> None:
    """ランが無い場合にNoneを返すこと。(T-008-09)"""
    with patch("mlflow.search_runs") as m_search:
        m_search.return_value = pd.DataFrame()
        assert mlflow_manager.get_best_run("rmse") is None

def test_save_model_registry_fail(mock_mlflow, mlflow_manager, tmp_path) -> None:
    """Model Registry 登録失敗時に警告が出るが続行されること。(T-008-11)"""
    import mlflow
    from sklearn.linear_model import Ridge
    model = Ridge()
    save_dir = tmp_path / "models"
    with patch("mlflow.sklearn.log_model", side_effect=Exception("registry error")):
        path = mlflow_manager.save_model(model, "fail_model", save_dir=save_dir, register=True)
        assert path.exists()

def test_log_artifact_no_mlflow(tmp_path) -> None:
    """MLflow 未インストール時に log_artifact が何もしないこと。(T-008-12)"""
    p = tmp_path / "test.txt"
    p.write_text("dummy")
    with patch("backend.mlops.mlflow_manager._mlflow", False):
        manager = MLflowManager(tracking_uri="http://localhost:5000", experiment_name="test_exp")
        manager.log_artifact(p) # No error

def test_run_context_with_tags(mock_mlflow, mlflow_manager) -> None:
    """タグ付きランのコンテキスト。(T-008-10)"""
    m_start, _ = mock_mlflow
    tags = {"team": "chem"}
    with MLRunContext(mlflow_manager, run_name="tagged_run", tags=tags) as rid:
        assert rid == "test_run_id"
        m_start.assert_called_with(run_name="tagged_run", tags=tags)
