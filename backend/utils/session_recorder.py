"""
backend/utils/session_recorder.py

計算セッションの完全記録・再現機能モジュール。

研究用途で不可欠な以下を提供:
- 入力データ、計算設定、環境情報の自動記録
- 結果のメタデータ付きシリアライズ
- 再現用バンドル（設定 + データハッシュ）の生成
- セッションログの永続化（JSON形式）

既存モジュールへの影響: なし（完全新規）
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """実行環境情報。"""
    python_version: str = ""
    os_name: str = ""
    os_version: str = ""
    machine: str = ""
    packages: dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> EnvironmentInfo:
        """現在の実行環境を取得する。"""
        info = cls(
            python_version=sys.version,
            os_name=platform.system(),
            os_version=platform.version(),
            machine=platform.machine(),
        )
        # 主要パッケージのバージョン
        for pkg_name in ["numpy", "pandas", "sklearn", "rdkit", "nicegui"]:
            try:
                mod = __import__(pkg_name)
                info.packages[pkg_name] = getattr(mod, "__version__", "unknown")
            except ImportError:
                pass
        return info


@dataclass
class SessionRecord:
    """
    1つの計算セッションの完全記録。

    Attributes:
        session_id: ユニークなセッションID (UUID)
        timestamp: 記録作成時刻 (ISO8601)
        environment: 実行環境情報
        input_config: 入力設定（SMILES、電荷、計算パラメータ等）
        computation_log: 計算過程の記録
        output_summary: 出力のサマリー（ハッシュ、形状等）
        notes: ユーザーメモ
    """
    session_id: str = ""
    timestamp: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    input_config: dict[str, Any] = field(default_factory=dict)
    computation_log: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """シリアライズ可能な辞書に変換する。"""
        return asdict(self)


class SessionRecorder:
    """
    計算セッションの記録と再現を管理するクラス。

    使い方::

        recorder = SessionRecorder(save_dir="sessions/")

        # セッション開始
        session = recorder.start_session(
            smiles_list=["CCO", "c1ccccc1"],
            config={"calc_type": "opt", "gfn": 2},
        )

        # 計算ログの追加
        recorder.log_event(session, "xtb_compute_start", {"n_molecules": 2})
        recorder.log_event(session, "xtb_compute_done", {"success_rate": 1.0})

        # 結果の記録
        recorder.record_output(session, df_features, "features")

        # セッション保存
        recorder.save_session(session)
    """

    def __init__(self, save_dir: str | Path | None = None):
        self._save_dir = Path(save_dir) if save_dir else None
        if self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    def start_session(
        self,
        smiles_list: list[str] | None = None,
        config: dict[str, Any] | None = None,
        notes: str = "",
    ) -> SessionRecord:
        """
        新しい計算セッションを開始する。

        Args:
            smiles_list: 入力SMILES（ハッシュのみ記録）。
            config: 計算設定辞書。
            notes: ユーザーメモ。

        Returns:
            SessionRecord インスタンス。
        """
        session_id = f"chemai2_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        env = EnvironmentInfo.capture()

        input_config: dict[str, Any] = {}
        if smiles_list is not None:
            input_config["n_molecules"] = len(smiles_list)
            input_config["smiles_hash"] = self._hash_list(smiles_list)
            input_config["source"] = "manual"
        if config:
            input_config["computation_params"] = config

        record = SessionRecord(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            environment=asdict(env),
            input_config=input_config,
            computation_log={"events": []},
            notes=notes,
        )

        logger.info("セッション開始: %s", session_id)
        return record

    def log_event(
        self,
        session: SessionRecord,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        セッションにイベントログを追加する。

        Args:
            session: 対象セッション。
            event_type: イベント種別（例: "xtb_compute_start"）。
            details: イベント詳細。
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "details": details or {},
        }
        if "events" not in session.computation_log:
            session.computation_log["events"] = []
        session.computation_log["events"].append(event)

    def record_output(
        self,
        session: SessionRecord,
        data: Any,
        label: str = "output",
    ) -> None:
        """
        計算結果のメタデータをセッションに記録する。

        DataFrameの場合は形状とカラムハッシュを記録する。
        データそのものは保存しない。

        Args:
            session: 対象セッション。
            data: 出力データ（DataFrame等）。
            label: ラベル名。
        """
        import pandas as pd

        summary: dict[str, Any] = {"label": label}

        if isinstance(data, pd.DataFrame):
            summary["type"] = "DataFrame"
            summary["shape"] = list(data.shape)
            summary["columns"] = list(data.columns)
            summary["columns_hash"] = self._hash_list(list(data.columns))
            summary["null_counts"] = int(data.isnull().sum().sum())
        elif isinstance(data, dict):
            summary["type"] = "dict"
            summary["n_keys"] = len(data)
        elif isinstance(data, list):
            summary["type"] = "list"
            summary["length"] = len(data)
        else:
            summary["type"] = type(data).__name__

        session.output_summary[label] = summary

    def save_session(self, session: SessionRecord) -> Path | None:
        """
        セッション記録をJSONファイルに保存する。

        Returns:
            保存先のPath。save_dir未設定の場合はNone。
        """
        if self._save_dir is None:
            logger.warning("save_dir未設定: セッション '%s' は保存されません", session.session_id)
            return None

        file_path = self._save_dir / f"{session.session_id}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info("セッション保存: %s", file_path)
            return file_path
        except Exception as e:
            logger.error("セッション保存失敗: %s", e)
            return None

    def load_session(self, session_id: str) -> SessionRecord | None:
        """
        保存済みセッションを読み込む。

        Args:
            session_id: セッションID。

        Returns:
            SessionRecord またはNone。
        """
        if self._save_dir is None:
            return None

        file_path = self._save_dir / f"{session_id}.json"
        if not file_path.exists():
            logger.warning("セッション '%s' が見つかりません", session_id)
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            record = SessionRecord(**data)
            return record
        except Exception as e:
            logger.error("セッション読み込み失敗: %s", e)
            return None

    def list_sessions(self) -> list[dict[str, str]]:
        """
        保存済みセッション一覧を返す。

        Returns:
            [{"session_id": ..., "timestamp": ..., "notes": ...}, ...]
        """
        if self._save_dir is None:
            return []

        sessions = []
        for p in sorted(self._save_dir.glob("chemai2_*.json"), reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id", p.stem),
                    "timestamp": data.get("timestamp", ""),
                    "notes": data.get("notes", ""),
                })
            except Exception:
                pass
        return sessions

    def generate_reproducibility_bundle(
        self,
        session: SessionRecord,
    ) -> dict[str, Any]:
        """
        再現に必要な情報をまとめたバンドルを生成する。

        Returns:
            再現用の設定辞書。他ユーザーに共有可能。
        """
        return {
            "session_id": session.session_id,
            "created_at": session.timestamp,
            "python_version": session.environment.get("python_version", ""),
            "input_config": session.input_config,
            "computation_params": session.input_config.get("computation_params", {}),
            "output_shape": {
                k: v.get("shape") for k, v in session.output_summary.items()
                if isinstance(v, dict) and "shape" in v
            },
            "reproducibility_hash": self._hash_dict(session.input_config),
        }

    # ────────────────────────────────────────────────────────
    # ユーティリティ
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _hash_list(items: list) -> str:
        """リストのSHA256ハッシュ（先頭12文字）を返す。"""
        content = json.dumps(items, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    @staticmethod
    def _hash_dict(d: dict) -> str:
        """辞書のSHA256ハッシュ（先頭12文字）を返す。"""
        content = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
