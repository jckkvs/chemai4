"""
tests/test_preset_manager.py

backend/preset_manager.py のユニットテスト。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.preset_manager import (
    PIPELINE_KEYS,
    save_preset,
    load_preset,
    list_presets,
    delete_preset,
    export_state_summary,
    _make_serializable,
)


@pytest.fixture
def tmp_preset_dir(tmp_path):
    """テスト用のプリセットディレクトリ。"""
    return tmp_path / "presets"


@pytest.fixture
def sample_state():
    """テスト用のstate辞書。"""
    return {
        "user_mode": "advanced",
        "task_type": "regression",
        "cv_key": "kfold",
        "cv_folds": 5,
        "timeout": 300,
        "num_scaler": "standard",
        "num_imputer": "median",
        "selected_models": ["Ridge", "RF", "LGBM"],
        "do_eda": True,
        "do_shap": True,
        # 非保存対象
        "df": "should_not_be_saved",
        "automl_result": "should_not_be_saved",
    }


class TestSavePreset:
    def test_save_creates_file(self, tmp_preset_dir, sample_state):
        path = save_preset("test_preset", sample_state, preset_dir=tmp_preset_dir)
        assert path.exists()
        assert path.suffix in (".yaml", ".json")

    def test_save_content(self, tmp_preset_dir, sample_state):
        save_preset("my_preset", sample_state, description="テスト", tags=["回帰"], preset_dir=tmp_preset_dir)
        files = list(tmp_preset_dir.iterdir())
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "my_preset" in content
        assert "テスト" in content

    def test_save_empty_name_raises(self, tmp_preset_dir, sample_state):
        with pytest.raises(ValueError, match="プリセット名は必須"):
            save_preset("", sample_state, preset_dir=tmp_preset_dir)

    def test_save_excludes_non_pipeline_keys(self, tmp_preset_dir, sample_state):
        save_preset("check", sample_state, preset_dir=tmp_preset_dir)
        files = list(tmp_preset_dir.iterdir())
        content = files[0].read_text(encoding="utf-8")
        assert "should_not_be_saved" not in content

    def test_save_special_chars_in_name(self, tmp_preset_dir, sample_state):
        path = save_preset("テスト/プリセット", sample_state, preset_dir=tmp_preset_dir)
        assert path.exists()


class TestLoadPreset:
    def test_load_restores_state(self, tmp_preset_dir, sample_state):
        save_preset("restore_test", sample_state, preset_dir=tmp_preset_dir)
        new_state: dict = {}
        meta = load_preset("restore_test", new_state, preset_dir=tmp_preset_dir)
        assert new_state["cv_folds"] == 5
        assert new_state["num_scaler"] == "standard"
        assert new_state["selected_models"] == ["Ridge", "RF", "LGBM"]
        assert "name" in meta
        assert "keys_loaded" in meta

    def test_load_nonexistent_raises(self, tmp_preset_dir):
        with pytest.raises(FileNotFoundError, match="見つかりません"):
            load_preset("nonexistent", {}, preset_dir=tmp_preset_dir)

    def test_load_does_not_overwrite_non_pipeline_keys(self, tmp_preset_dir, sample_state):
        save_preset("safe", sample_state, preset_dir=tmp_preset_dir)
        target_state = {"df": "keep_me", "automl_result": "keep_me"}
        load_preset("safe", target_state, preset_dir=tmp_preset_dir)
        assert target_state["df"] == "keep_me"
        assert target_state["automl_result"] == "keep_me"


class TestListPresets:
    def test_list_empty(self, tmp_preset_dir):
        result = list_presets(preset_dir=tmp_preset_dir)
        assert result == []

    def test_list_multiple(self, tmp_preset_dir, sample_state):
        save_preset("preset_a", sample_state, description="A", preset_dir=tmp_preset_dir)
        save_preset("preset_b", sample_state, description="B", tags=["test"], preset_dir=tmp_preset_dir)
        result = list_presets(preset_dir=tmp_preset_dir)
        assert len(result) == 2
        names = {p["name"] for p in result}
        assert "preset_a" in names
        assert "preset_b" in names
        assert all("n_settings" in p for p in result)


class TestDeletePreset:
    def test_delete_existing(self, tmp_preset_dir, sample_state):
        save_preset("to_delete", sample_state, preset_dir=tmp_preset_dir)
        assert delete_preset("to_delete", preset_dir=tmp_preset_dir) is True
        assert list_presets(preset_dir=tmp_preset_dir) == []

    def test_delete_nonexistent(self, tmp_preset_dir):
        assert delete_preset("ghost", preset_dir=tmp_preset_dir) is False


class TestExportStateSummary:
    def test_export(self, sample_state):
        summary = export_state_summary(sample_state)
        assert "cv_folds" in summary
        assert "df" not in summary
        assert "automl_result" not in summary

    def test_export_empty(self):
        summary = export_state_summary({})
        assert summary == {}


class TestMakeSerializable:
    def test_numpy_types(self):
        import numpy as np
        assert _make_serializable(np.int64(42)) == 42
        assert isinstance(_make_serializable(np.int64(42)), int)
        assert _make_serializable(np.float32(3.14)) == pytest.approx(3.14, abs=0.01)
        assert _make_serializable(np.bool_(True)) is True
        assert _make_serializable(np.array([1, 2, 3])) == [1, 2, 3]

    def test_nested_dict(self):
        import numpy as np
        result = _make_serializable({"a": np.int64(1), "b": [np.float32(2.0)]})
        assert result == {"a": 1, "b": [pytest.approx(2.0, abs=0.01)]}

    def test_plain_types(self):
        assert _make_serializable("hello") == "hello"
        assert _make_serializable(42) == 42
        assert _make_serializable(None) is None


class TestRecordAnalysis:
    def test_record_creates_file(self, tmp_path, sample_state):
        from backend.preset_manager import record_analysis

        class MockResult:
            task = "regression"
            best_model_key = "LGBM"
            best_score = 0.85
            model_scores = {"LGBM": 0.85, "RF": 0.80}
            elapsed_seconds = 30.0
            processed_X = None

        import pandas as pd
        sample_state["df"] = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = record_analysis(sample_state, MockResult(), history_dir=tmp_path)
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["best_model"] == "LGBM"
        assert data["best_score"] == 0.85


class TestListHistory:
    def test_list_empty(self, tmp_path):
        from backend.preset_manager import list_history
        assert list_history(history_dir=tmp_path) == []

    def test_list_after_record(self, tmp_path, sample_state):
        from backend.preset_manager import record_analysis, list_history
        import pandas as pd

        class MockResult:
            task = "regression"
            best_model_key = "RF"
            best_score = 0.80
            model_scores = {"RF": 0.80}
            elapsed_seconds = 10.0
            processed_X = None

        sample_state["df"] = pd.DataFrame({"x": [1]})
        record_analysis(sample_state, MockResult(), history_dir=tmp_path)
        records = list_history(history_dir=tmp_path)
        assert len(records) == 1
        assert records[0]["best_model"] == "RF"


class TestExportImportConfigYaml:
    def test_export_yaml(self, sample_state):
        from backend.preset_manager import export_config_yaml
        yaml_text = export_config_yaml(sample_state)
        assert "cv_folds" in yaml_text
        assert "chemai2_config" in yaml_text

    def test_import_yaml(self, sample_state):
        from backend.preset_manager import export_config_yaml, import_config_yaml
        yaml_text = export_config_yaml(sample_state)
        new_state: dict = {}
        count = import_config_yaml(yaml_text, new_state)
        assert count > 0
        assert new_state.get("cv_folds") == 5

    def test_roundtrip(self, sample_state):
        from backend.preset_manager import export_config_yaml, import_config_yaml
        yaml_text = export_config_yaml(sample_state)
        new_state: dict = {}
        import_config_yaml(yaml_text, new_state)
        assert new_state.get("num_scaler") == sample_state["num_scaler"]
        assert new_state.get("selected_models") == sample_state["selected_models"]
