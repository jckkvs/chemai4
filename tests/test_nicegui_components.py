"""
tests/test_nicegui_components.py

NiceGUIフロントエンドのロジック部分のユニットテスト。
UIコンポーネント自体はテスト対象外（NiceGUIランタイム不要）。
データ処理・ステート管理のロジックのみをテストする。
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# backendへのパスを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ================================================================
# data_tab.py のロジックテスト
# ================================================================

class TestAutoDetectColumns:
    """_auto_detect_columns のロジックテスト"""

    def _get_func(self):
        from frontend_nicegui.components.data_tab import _auto_detect_columns
        return _auto_detect_columns

    def test_detects_smiles_column_by_name(self):
        """SMILES列が列名で自動検出されること"""
        func = self._get_func()
        df = pd.DataFrame({
            "SMILES": ["C", "CC", "CCC"],
            "target": [1.0, 2.0, 3.0],
        })
        state = {"df": df}
        func(state)
        assert state["smiles_col"] == "SMILES"
        assert state["target_col"] == "target"
        assert state["task_type"] == "regression"

    def test_detects_smiles_column_case_insensitive(self):
        """smiles列が小文字でも検出されること"""
        func = self._get_func()
        df = pd.DataFrame({
            "smiles": ["C", "CC", "CCC"],
            "value": [1.0, 2.0, 3.0],
        })
        state = {"df": df}
        func(state)
        assert state["smiles_col"] == "smiles"

    def test_target_col_is_last_column(self):
        """目的変数が最後の列に自動設定されること"""
        func = self._get_func()
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "target_value": [0.1, 0.2, 0.3],
        })
        state = {"df": df}
        func(state)
        assert state["target_col"] == "target_value"

    def test_classification_task_detection(self):
        """分類タスクが自動判定されること（最後の列がint型）"""
        func = self._get_func()
        df = pd.DataFrame({
            "SMILES": ["C", "CC", "CCC"],
            "is_toxic": [0, 1, 0],
        })
        state = {"df": df}
        func(state)
        assert state["task_type"] == "classification"

    def test_regression_task_detection(self):
        """回帰タスクが自動判定されること（最後の列がfloat型）"""
        func = self._get_func()
        df = pd.DataFrame({
            "SMILES": ["C", "CC", "CCC"],
            "logS": [1.5, -2.3, 0.8],
        })
        state = {"df": df}
        func(state)
        assert state["task_type"] == "regression"

    def test_no_smiles_column(self):
        """SMILES列が存在しない場合に空文字が設定されること"""
        func = self._get_func()
        df = pd.DataFrame({
            "temperature": [20.0, 30.0, 40.0],
            "yield": [75.0, 80.0, 85.0],
        })
        state = {"df": df}
        func(state)
        assert state["smiles_col"] == ""

    def test_none_df(self):
        """dfがNoneの場合にエラーにならないこと"""
        func = self._get_func()
        state = {"df": None}
        # None の場合は早期リターン
        func(state)
        # target_col は設定されない（state に無い or 変更なし）


class TestToggleModel:
    """_toggle_model のステート操作テスト"""

    def _get_func(self):
        from frontend_nicegui.components.data_tab import _toggle_model
        return _toggle_model

    def test_add_model(self):
        """モデルを選択リストに追加できること"""
        func = self._get_func()
        state = {"selected_models": ["ridge", "lasso"]}
        func(state, "xgboost", True)
        assert "xgboost" in state["selected_models"]

    def test_remove_model(self):
        """モデルを選択リストから削除できること"""
        func = self._get_func()
        state = {"selected_models": ["ridge", "lasso", "xgboost"]}
        func(state, "xgboost", False)
        assert "xgboost" not in state["selected_models"]

    def test_add_duplicate_model(self):
        """既に存在するモデルを追加しても重複しないこと"""
        func = self._get_func()
        state = {"selected_models": ["ridge", "lasso"]}
        func(state, "ridge", True)
        assert state["selected_models"].count("ridge") == 1

    def test_remove_nonexistent_model(self):
        """存在しないモデルの削除でエラーにならないこと"""
        func = self._get_func()
        state = {"selected_models": ["ridge"]}
        func(state, "nonexistent", False)
        assert state["selected_models"] == ["ridge"]

    def test_empty_list(self):
        """空リストにモデルを追加できること"""
        func = self._get_func()
        state = {"selected_models": []}
        func(state, "ridge", True)
        assert state["selected_models"] == ["ridge"]


class TestOnTargetChange:
    """_on_target_change のタスク自動判定テスト"""

    def _get_func(self):
        from frontend_nicegui.components.data_tab import _on_target_change
        return _on_target_change

    def test_regression_detection(self):
        """float型の列で回帰タスクが設定されること"""
        func = self._get_func()
        df = pd.DataFrame({
            "SMILES": ["C", "CC", "CCC"],
            "logS": [1.5, -2.3, 0.8],
        })
        state = {"df": df, "target_col": "", "task_type": "", "precalc_done": True}
        func("logS", state)
        assert state["target_col"] == "logS"
        assert state["task_type"] == "regression"
        assert state["precalc_done"] is False

    def test_classification_detection(self):
        """int型の列で分類タスクが設定されること"""
        func = self._get_func()
        df = pd.DataFrame({
            "SMILES": ["C", "CC", "CCC"],
            "is_active": [0, 1, 1],
        })
        state = {"df": df, "target_col": "", "task_type": "", "precalc_done": True}
        func("is_active", state)
        assert state["target_col"] == "is_active"
        assert state["task_type"] == "classification"

    def test_column_not_in_df(self):
        """存在しない列名で呼んでもクラッシュしないこと"""
        func = self._get_func()
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })
        state = {"df": df, "target_col": "", "task_type": "", "precalc_done": True}
        func("nonexistent", state)
        # target_colは設定されるが、task_typeは変更されない
        assert state["target_col"] == "nonexistent"


class TestSampleSmiles:
    """SAMPLE_SMILES 定数のテスト"""

    def test_sample_smiles_not_empty(self):
        from frontend_nicegui.components.data_tab import SAMPLE_SMILES
        assert len(SAMPLE_SMILES) > 0

    def test_sample_smiles_are_strings(self):
        from frontend_nicegui.components.data_tab import SAMPLE_SMILES
        assert all(isinstance(s, str) for s in SAMPLE_SMILES)

    def test_sample_smiles_unique(self):
        from frontend_nicegui.components.data_tab import SAMPLE_SMILES
        assert len(SAMPLE_SMILES) == len(set(SAMPLE_SMILES))


# ================================================================
# analysis_runner.py のロジックテスト
# ================================================================

class TestRunEngineSync:
    """_run_engine_sync のテスト"""

    def test_function_exists(self):
        """_run_engine_sync 関数が存在すること"""
        from frontend_nicegui.components.analysis_runner import _run_engine_sync
        assert callable(_run_engine_sync)

    def test_analysis_running_flag_exists(self):
        """_analysis_running フラグが存在すること"""
        from frontend_nicegui.components import analysis_runner
        assert hasattr(analysis_runner, "_analysis_running")

    @patch("backend.models.automl.AutoMLEngine", autospec=True)
    def test_run_engine_sync_calls_engine(self, MockEngine):
        """_run_engine_sync が AutoMLEngine を正しく呼び出すこと"""
        import queue
        from frontend_nicegui.components.analysis_runner import _run_engine_sync

        # モック設定
        mock_result = MagicMock()
        mock_result.best_model_key = "ridge"
        mock_result.best_score = 0.85
        mock_result.elapsed_seconds = 10.0
        mock_result.task = "regression"
        MockEngine.return_value.run.return_value = mock_result

        df = pd.DataFrame({"a": [1, 2, 3], "target": [0.1, 0.2, 0.3]})
        q = queue.Queue()

        result = _run_engine_sync(
            df_work=df,
            target_col="target",
            smiles_col=None,
            group_col=None,
            task="regression",
            model_keys=["ridge"],
            cv_folds=3,
            timeout=60,
            selected_desc=None,
            progress_queue=q,
        )

        assert result.best_model_key == "ridge"
        assert result.best_score == 0.85
        MockEngine.return_value.run.assert_called_once()

    @patch("backend.models.automl.AutoMLEngine", autospec=True)
    def test_progress_callback_sends_to_queue(self, MockEngine):
        """進捗コールバックがキューに送信すること"""
        import queue
        from frontend_nicegui.components.analysis_runner import _run_engine_sync

        q = queue.Queue()

        # engine.run の中で progress_callback を呼ぶモック
        def mock_run(*args, **kwargs):
            # エンジン作成時の progress_callback を取得して呼ぶ
            cb = MockEngine.call_args[1].get("progress_callback")
            if cb:
                cb(1, 5, "Step 1")
                cb(2, 5, "Step 2")
            mock_result = MagicMock()
            mock_result.best_model_key = "test"
            mock_result.best_score = 0.9
            return mock_result

        MockEngine.return_value.run.side_effect = mock_run

        df = pd.DataFrame({"a": [1, 2], "t": [0.1, 0.2]})

        _run_engine_sync(
            df_work=df,
            target_col="t",
            smiles_col=None,
            group_col=None,
            task="regression",
            model_keys=None,
            cv_folds=3,
            timeout=60,
            selected_desc=None,
            progress_queue=q,
        )

        # キューに進捗が入っていること
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert len(items) == 2
        assert items[0] == ("progress", 1, 5, "Step 1")
        assert items[1] == ("progress", 2, 5, "Step 2")


# ================================================================
# エンジン定義のテスト
# ================================================================

class TestAllEngines:
    """_ALL_ENGINES 定数のテスト"""

    def test_engines_list_not_empty(self):
        from frontend_nicegui.components.data_tab import _ALL_ENGINES
        assert len(_ALL_ENGINES) > 0

    def test_engine_tuple_structure(self):
        """各エンジンが (name, module, class_name, kwargs) のタプルであること"""
        from frontend_nicegui.components.data_tab import _ALL_ENGINES
        for eng in _ALL_ENGINES:
            assert len(eng) == 4
            name, mod, cls, kwargs = eng
            assert isinstance(name, str)
            assert isinstance(mod, str)
            assert isinstance(cls, str)
            assert isinstance(kwargs, dict)

    def test_engine_count(self):
        """14エンジンが定義されていること"""
        from frontend_nicegui.components.data_tab import _ALL_ENGINES
        assert len(_ALL_ENGINES) == 14



# ================================================================
# descriptor_plugins_ui.py の _ENGINE_INFO テスト
# ================================================================

class TestEngineInfo:
    """_ENGINE_INFO 定数のテスト (manual_url フィールド含む)"""

    def _get_engine_info(self):
        from frontend_nicegui.components.descriptor_plugins_ui import _ENGINE_INFO
        return _ENGINE_INFO

    def test_engine_info_not_empty(self):
        """_ENGINE_INFO が空でないこと"""
        engines = self._get_engine_info()
        assert len(engines) > 0

    def test_engine_info_count(self):
        """14エンジンが定義されていること"""
        engines = self._get_engine_info()
        assert len(engines) == 14

    def test_all_engines_have_manual_url(self):
        """全エンジンに manual_url フィールドが存在すること"""
        engines = self._get_engine_info()
        for eng in engines:
            assert "manual_url" in eng, f"{eng['cls']} に manual_url がありません"

    def test_all_manual_urls_are_non_empty(self):
        """manual_url が空文字でないこと"""
        engines = self._get_engine_info()
        for eng in engines:
            url = eng.get("manual_url", "")
            assert url, f"{eng['cls']} の manual_url が空です"

    def test_all_manual_urls_start_with_http(self):
        """manual_url が http:// または https:// で始まること"""
        engines = self._get_engine_info()
        for eng in engines:
            url = eng.get("manual_url", "")
            assert url.startswith("http://") or url.startswith("https://"), \
                f"{eng['cls']} の manual_url ({url}) が無効なURL形式です"

    def test_required_fields_present(self):
        """必須フィールド (cls, label, category, dims, speed, desc, manual_url) が全て存在すること"""
        required_keys = {"cls", "label", "category", "dims", "speed", "desc", "manual_url"}
        engines = self._get_engine_info()
        for eng in engines:
            missing = required_keys - set(eng.keys())
            assert not missing, f"{eng['cls']} にフィールドがありません: {missing}"

    def test_engine_cls_names_unique(self):
        """cls 名がすべて一意であること"""
        engines = self._get_engine_info()
        cls_names = [e["cls"] for e in engines]
        assert len(cls_names) == len(set(cls_names)), "重複する cls 名があります"

    def test_rdkit_adapter_url(self):
        """RDKitAdapter のマニュアル URL が正しいこと"""
        engines = self._get_engine_info()
        rdkit_eng = next((e for e in engines if e["cls"] == "RDKitAdapter"), None)
        assert rdkit_eng is not None
        assert "rdkit.org" in rdkit_eng["manual_url"]

    def test_xtb_adapter_url(self):
        """XTBAdapter の manual_url が xtb-docs.readthedocs.io であること"""
        engines = self._get_engine_info()
        xtb_eng = next((e for e in engines if e["cls"] == "XTBAdapter"), None)
        assert xtb_eng is not None
        assert "xtb-docs" in xtb_eng["manual_url"]


# ================================================================
# auto_params_ui.py のインポートテスト
# ================================================================

class TestAutoParamsUiImport:
    """auto_params_ui のインポートテスト"""

    def test_render_param_editor_importable(self):
        from frontend_nicegui.components.auto_params_ui import render_param_editor
        assert callable(render_param_editor)

    def test_render_model_param_editor_importable(self):
        from frontend_nicegui.components.auto_params_ui import render_model_param_editor
        assert callable(render_model_param_editor)

    def test_render_adapter_param_editor_importable(self):
        from frontend_nicegui.components.auto_params_ui import render_adapter_param_editor
        assert callable(render_adapter_param_editor)
