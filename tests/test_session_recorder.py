"""
tests/test_session_recorder.py

SessionRecorder のユニットテスト。
"""
import json
import tempfile
from pathlib import Path

import pytest

from backend.utils.session_recorder import (
    EnvironmentInfo,
    SessionRecord,
    SessionRecorder,
)


@pytest.fixture
def tmp_session_dir(tmp_path):
    return tmp_path / "sessions"


@pytest.fixture
def recorder(tmp_session_dir) -> SessionRecorder:
    return SessionRecorder(save_dir=tmp_session_dir)


# ── EnvironmentInfo ──

def test_environment_capture():
    env = EnvironmentInfo.capture()
    assert len(env.python_version) > 0
    assert len(env.os_name) > 0
    assert "numpy" in env.packages


# ── セッション開始 ──

def test_start_session(recorder):
    session = recorder.start_session(
        smiles_list=["CCO", "c1ccccc1"],
        config={"calc_type": "opt", "gfn": 2},
        notes="テスト実行",
    )
    assert session.session_id.startswith("chemai2_")
    assert len(session.timestamp) > 0
    assert session.input_config["n_molecules"] == 2
    assert "smiles_hash" in session.input_config
    assert session.notes == "テスト実行"


def test_start_session_no_smiles(recorder):
    session = recorder.start_session(config={"calc_type": "sp"})
    assert "n_molecules" not in session.input_config
    assert "computation_params" in session.input_config


# ── イベントログ ──

def test_log_event(recorder):
    session = recorder.start_session()
    recorder.log_event(session, "compute_start", {"n": 10})
    recorder.log_event(session, "compute_done", {"success": 0.95})
    
    events = session.computation_log["events"]
    assert len(events) == 2
    assert events[0]["type"] == "compute_start"
    assert events[1]["details"]["success"] == 0.95


# ── 結果記録 ──

def test_record_output_dataframe(recorder):
    import pandas as pd
    
    session = recorder.start_session()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    recorder.record_output(session, df, "features")
    
    summary = session.output_summary["features"]
    assert summary["type"] == "DataFrame"
    assert summary["shape"] == [3, 2]
    assert "a" in summary["columns"]
    assert summary["null_counts"] == 0


def test_record_output_dict(recorder):
    session = recorder.start_session()
    recorder.record_output(session, {"key": "val"}, "config")
    
    summary = session.output_summary["config"]
    assert summary["type"] == "dict"
    assert summary["n_keys"] == 1


def test_record_output_list(recorder):
    session = recorder.start_session()
    recorder.record_output(session, [1, 2, 3], "indices")
    
    summary = session.output_summary["indices"]
    assert summary["type"] == "list"
    assert summary["length"] == 3


# ── 保存・読み込み ──

def test_save_and_load(recorder):
    session = recorder.start_session(
        smiles_list=["CCO"],
        config={"gfn": 2},
        notes="保存テスト",
    )
    recorder.log_event(session, "test_event")
    
    path = recorder.save_session(session)
    assert path is not None
    assert path.exists()
    
    loaded = recorder.load_session(session.session_id)
    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert loaded.notes == "保存テスト"


def test_save_creates_valid_json(recorder):
    session = recorder.start_session(smiles_list=["CCO"])
    path = recorder.save_session(session)
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "session_id" in data
    assert "timestamp" in data
    assert "environment" in data


def test_load_nonexistent_returns_none(recorder):
    result = recorder.load_session("nonexistent_session_id")
    assert result is None


def test_save_no_dir_returns_none():
    recorder = SessionRecorder(save_dir=None)
    session = recorder.start_session()
    result = recorder.save_session(session)
    assert result is None


# ── セッション一覧 ──

def test_list_sessions(recorder):
    s1 = recorder.start_session(notes="first")
    s2 = recorder.start_session(notes="second")
    recorder.save_session(s1)
    recorder.save_session(s2)
    
    sessions = recorder.list_sessions()
    assert len(sessions) == 2
    assert any(s["notes"] == "first" for s in sessions)
    assert any(s["notes"] == "second" for s in sessions)


def test_list_sessions_empty_dir():
    recorder = SessionRecorder(save_dir=None)
    assert recorder.list_sessions() == []


# ── 再現バンドル ──

def test_reproducibility_bundle(recorder):
    session = recorder.start_session(
        smiles_list=["CCO", "CCCO"],
        config={"calc_type": "opt", "gfn": 2},
    )
    
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2]})
    recorder.record_output(session, df, "features")
    
    bundle = recorder.generate_reproducibility_bundle(session)
    assert "session_id" in bundle
    assert "reproducibility_hash" in bundle
    assert "computation_params" in bundle
    assert bundle["computation_params"]["calc_type"] == "opt"


# ── ハッシュの安定性 ──

def test_hash_deterministic(recorder):
    s1 = recorder.start_session(smiles_list=["CCO", "c1ccccc1"])
    s2 = recorder.start_session(smiles_list=["CCO", "c1ccccc1"])
    assert s1.input_config["smiles_hash"] == s2.input_config["smiles_hash"]


def test_hash_changes_with_input(recorder):
    s1 = recorder.start_session(smiles_list=["CCO"])
    s2 = recorder.start_session(smiles_list=["CCCO"])
    assert s1.input_config["smiles_hash"] != s2.input_config["smiles_hash"]


# ── to_dict ──

def test_session_to_dict(recorder):
    session = recorder.start_session(smiles_list=["CCO"])
    d = session.to_dict()
    assert isinstance(d, dict)
    assert "session_id" in d
    # JSON serializable
    json.dumps(d)
