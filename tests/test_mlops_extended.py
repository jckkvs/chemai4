# -*- coding: utf-8 -*-
"""
tests/test_mlops_extended.py

MLflowManager と MLRunContext のユニットテスト（mock使用）。
mlflow未インストール時のGraceful Degradationも検証。
"""
from __future__ import annotations

import tempfile
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════
# MLflowManager テスト（mlflow未インストール想定）
# ═══════════════════════════════════════════════════════════════════

class TestMLflowManagerGracefulDegradation:
    """mlflowが未インストールの場合のGraceful Degradation"""

    def test_init_without_mlflow(self):
        """mlflow未インストールでも初期化成功"""
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            assert mgr._run_id is None

    def test_start_run_returns_none(self):
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            result = mgr.start_run("test_run")
            assert result is None

    def test_end_run_noop(self):
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            mgr.end_run()  # エラーなし

    def test_log_params_noop(self):
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            mgr.log_params({"lr": 0.01, "epochs": 100})  # エラーなし

    def test_log_metrics_noop(self):
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            mgr.log_metrics({"rmse": 0.05})  # エラーなし

    def test_log_artifact_noop(self):
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            mgr.log_artifact("/tmp/test.txt")  # エラーなし

    def test_log_figure_noop(self):
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            mgr.log_figure(mock.MagicMock(), "test.png")  # エラーなし

    def test_get_experiment_runs_returns_empty(self):
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            df = mgr.get_experiment_runs()
            assert isinstance(df, pd.DataFrame)
            assert df.empty

    def test_get_best_run_returns_none(self):
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            result = mgr.get_best_run("rmse")
            assert result is None

    def test_fail_run_noop(self):
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            mgr.fail_run(Exception("test error"))  # エラーなし


# ═══════════════════════════════════════════════════════════════════
# モデル保存・読み込み
# ═══════════════════════════════════════════════════════════════════

class TestModelSaveLoad:

    def test_save_and_load_model(self):
        """モデルの保存と読み込みが正しく動作する"""
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()

            model = {"type": "dummy", "params": [1, 2, 3]}
            with tempfile.TemporaryDirectory() as tmpdir:
                path = mgr.save_model(model, "test_model", save_dir=tmpdir)
                assert path.exists()
                loaded = mgr.load_model(path)
                assert loaded == model

    def test_load_model_not_found(self):
        """存在しないパスでFileNotFoundError"""
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            with pytest.raises(FileNotFoundError):
                mgr.load_model("/nonexistent/path/model.joblib")

    def test_save_creates_directory(self):
        """save_dirが存在しない場合も自動作成"""
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            with tempfile.TemporaryDirectory() as tmpdir:
                nested = Path(tmpdir) / "deep" / "nested"
                path = mgr.save_model({"x": 1}, "nested_model", save_dir=nested)
                assert path.exists()


# ═══════════════════════════════════════════════════════════════════
# MLRunContext
# ═══════════════════════════════════════════════════════════════════

class TestMLRunContext:

    def test_context_without_mlflow(self):
        """mlflow未インストール時のコンテキスト"""
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager, MLRunContext
            mgr = MLflowManager()
            with MLRunContext(mgr, run_name="test") as run_id:
                assert run_id is None

    def test_context_exception_calls_fail_run(self):
        """コンテキスト内で例外が発生した場合、fail_runが呼ばれる"""
        with mock.patch("backend.mlops.mlflow_manager._mlflow", None):
            from backend.mlops.mlflow_manager import MLflowManager, MLRunContext
            mgr = MLflowManager()
            mgr.fail_run = mock.MagicMock()
            try:
                with MLRunContext(mgr, run_name="failing"):
                    raise ValueError("test error")
            except ValueError:
                pass
            mgr.fail_run.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# CosmoAdapter mockテスト
# ═══════════════════════════════════════════════════════════════════

class TestCosmoAdapterMock:

    def test_name_and_description(self):
        from backend.chem.cosmo_adapter import CosmoAdapter
        adp = CosmoAdapter()
        assert adp.name == "cosmo_rs"
        assert "COSMO" in adp.description

    def test_descriptor_names(self):
        from backend.chem.cosmo_adapter import CosmoAdapter
        adp = CosmoAdapter()
        names = adp.get_descriptor_names()
        assert "mu_comb" in names
        assert "mu_res" in names
        assert "ln_gamma" in names

    def test_descriptor_metadata(self):
        from backend.chem.cosmo_adapter import CosmoAdapter
        adp = CosmoAdapter()
        meta = adp.get_descriptors_metadata()
        assert len(meta) == 3
        assert meta[0].name == "mu_comb"

    def test_compute_without_cosmi_files_returns_nan(self):
        """cosmi_filesなしではNaN"""
        mock_cosmors = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"opencosmorspy": mock_cosmors}):
            from backend.chem.cosmo_adapter import CosmoAdapter
            adp = CosmoAdapter()
            # is_availableをTrueに
            with mock.patch.object(adp, "is_available", return_value=True):
                result = adp.compute(["CCO", "c1ccccc1"])
                assert result.descriptors.shape[0] == 2
                # 全てNaN
                assert result.descriptors.isna().all().all()
