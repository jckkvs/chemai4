"""
backend/session/version_manager.py

SQLite ベースの実験バージョン管理システム。
hashlib.sha256 によるハッシュキー生成と、
sqlite3 による永続化を行う。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(".chemai_experiments.db")
_SCHEMA_PATH = Path(__file__).parent.parent / "db" / "schema.sql"


@dataclass
class ExperimentRecord:
    """1回の実験結果を表すデータクラス。"""
    exp_hash: str
    exp_name: str
    task_type: str
    target_col: str
    best_model_key: str
    best_score: float
    scoring_metric: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_name: str = ""
    n_samples: int = 0
    n_features: int = 0
    cv_folds: int = 5
    cv_strategy: str = "KFold"
    metrics_json: str = "{}"
    hyperparams_json: str = "{}"
    preprocess_json: str = "{}"
    engines_json: str = "[]"
    elapsed_seconds: float = 0.0
    notes: str = ""


class VersionManager:
    """実験設定・結果のバージョン管理を行うクラス。

    Parameters
    ----------
    db_path : str | Path
        SQLite データベースファイルのパス。
        存在しない場合は新規作成する。
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """スレッドセーフな接続を返す。"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        """スキーマを読み込み、テーブルを作成（冪等）。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if _SCHEMA_PATH.exists():
            schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        else:
            # スキーマファイルが見つからない場合は最小限のテーブルを作成する
            schema_sql = """
            CREATE TABLE IF NOT EXISTS experiments (
                exp_hash         TEXT PRIMARY KEY,
                created_at       TEXT NOT NULL DEFAULT (datetime('now')),
                exp_name         TEXT NOT NULL DEFAULT 'Unnamed',
                dataset_name     TEXT,
                n_samples        INTEGER,
                n_features       INTEGER,
                task_type        TEXT NOT NULL,
                target_col       TEXT NOT NULL,
                cv_folds         INTEGER NOT NULL DEFAULT 5,
                cv_strategy      TEXT NOT NULL DEFAULT 'KFold',
                best_model_key   TEXT NOT NULL,
                best_score       REAL NOT NULL,
                scoring_metric   TEXT NOT NULL,
                metrics_json     TEXT,
                hyperparams_json TEXT,
                preprocess_json  TEXT,
                engines_json     TEXT,
                elapsed_seconds  REAL,
                notes            TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_created_at
                ON experiments (created_at DESC);
            """
        with self._connect() as conn:
            conn.executescript(schema_sql)

    @staticmethod
    def compute_hash(hyperparams: dict[str, Any], preprocess_params: dict[str, Any]) -> str:
        """ハイパーパラメータと前処理設定から SHA-256 ハッシュを生成する。

        Parameters
        ----------
        hyperparams : dict
            モデルのハイパーパラメータ辞書（順序不問）。
        preprocess_params : dict
            前処理パラメータ辞書（スケーラー名、欠損補完方法等）。

        Returns
        -------
        str
            64文字の hex ダイジェスト文字列。
        """
        payload = {
            "hyperparams": dict(sorted(hyperparams.items())),
            "preprocess": dict(sorted(preprocess_params.items())),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save(self, record: ExperimentRecord) -> str:
        """実験レコードを SQLite に保存する。

        同一ハッシュが既に存在する場合は REPLACE (上書き) する。

        Parameters
        ----------
        record : ExperimentRecord
            保存する実験レコード。

        Returns
        -------
        str
            保存した exp_hash。
        """
        row = asdict(record)
        placeholders = ", ".join(["?"] * len(row))
        columns = ", ".join(row.keys())
        values = list(row.values())

        sql = f"INSERT OR REPLACE INTO experiments ({columns}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, values)
        logger.info("実験を保存しました: hash=%s name=%s", record.exp_hash[:8], record.exp_name)
        return record.exp_hash

    def save_from_automl_result(
        self,
        ar: Any,
        state: dict[str, Any],
        exp_name: str = "",
    ) -> str:
        """AutoMLResult オブジェクトと state からレコードを生成して保存する。

        Parameters
        ----------
        ar : AutoMLResult
            AutoMLEngine.run() の戻り値。
        state : dict
            フロントエンドの共有ステート辞書。
        exp_name : str
            実験の表示名。空の場合は「{best_model_key}_{日時}」を使用。

        Returns
        -------
        str
            保存した exp_hash。
        """
        hyperparams: dict = {}
        best_detail = getattr(ar, "model_details", {}).get(ar.best_model_key, {})
        if best_detail:
            hyperparams = best_detail.get("params", {}) or {}

        preprocess_params: dict = {
            k: state.get(k, "")
            for k in ["num_scaler", "num_imputer", "cat_encoder", "feature_selector"]
        }

        exp_hash = self.compute_hash(hyperparams, preprocess_params)

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = exp_name or f"{ar.best_model_key}_{timestamp_str}"

        proc_X = getattr(ar, "processed_X", None)
        n_feats = proc_X.shape[1] if proc_X is not None and hasattr(proc_X, "shape") else 0
        n_samp = proc_X.shape[0] if proc_X is not None and hasattr(proc_X, "shape") else 0

        # 評価指標の収集
        metrics: dict = {}
        if ar.oof_true is not None and ar.oof_predictions is not None:
            import numpy as np
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            y_t = np.asarray(ar.oof_true).ravel()
            y_p = np.asarray(ar.oof_predictions).ravel()
            if ar.task == "regression":
                metrics["R2"]   = float(r2_score(y_t, y_p))
                metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_t, y_p)))
                metrics["MAE"]  = float(mean_absolute_error(y_t, y_p))
            else:
                from sklearn.metrics import accuracy_score, f1_score
                metrics["Accuracy"] = float(accuracy_score(y_t, y_p))
                metrics["F1"]       = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

        # 使用エンジンの収集
        engine_map = {
            "use_rdkit": "RDKitAdapter", "use_mordred": "MordredAdapter",
            "use_skfp": "SkfpAdapter", "use_molai": "MolAIAdapter",
            "use_descriptastorus": "DescriptaStorusAdapter",
        }
        engines = [v for k, v in engine_map.items() if state.get(k)]

        record = ExperimentRecord(
            exp_hash=exp_hash,
            exp_name=name,
            task_type=ar.task,
            target_col=state.get("target_col", ""),
            best_model_key=ar.best_model_key,
            best_score=float(ar.best_score),
            scoring_metric=getattr(ar, "scoring", ""),
            dataset_name=state.get("_csv_filename", ""),
            n_samples=n_samp,
            n_features=n_feats,
            cv_folds=getattr(ar, "cv_folds", 5),
            cv_strategy=state.get("cv_key", "KFold"),
            metrics_json=json.dumps(metrics, ensure_ascii=False),
            hyperparams_json=json.dumps(hyperparams, ensure_ascii=False),
            preprocess_json=json.dumps(preprocess_params, ensure_ascii=False),
            engines_json=json.dumps(engines, ensure_ascii=False),
            elapsed_seconds=float(getattr(ar, "elapsed_seconds", 0.0)),
            notes="",
        )
        return self.save(record)

    def list_experiments(
        self,
        limit: int = 50,
        task_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """保存済み実験の一覧を取得する（最新順）。

        Parameters
        ----------
        limit : int
            取得件数の上限。
        task_type : str | None
            "regression" または "classification" でフィルタ。None で全件。

        Returns
        -------
        list[dict]
            実験レコードの辞書リスト。
        """
        if task_type:
            sql = (
                "SELECT * FROM experiments WHERE task_type = ? "
                "ORDER BY created_at DESC LIMIT ?"
            )
            params = (task_type, limit)
        else:
            sql = "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?"
            params = (limit,)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_experiment(self, exp_hash: str) -> dict[str, Any] | None:
        """ハッシュで単一の実験レコードを取得する。"""
        sql = "SELECT * FROM experiments WHERE exp_hash = ?"
        with self._connect() as conn:
            row = conn.execute(sql, (exp_hash,)).fetchone()
        return dict(row) if row else None

    def delete_experiment(self, exp_hash: str) -> bool:
        """指定ハッシュの実験レコードを削除する。"""
        sql = "DELETE FROM experiments WHERE exp_hash = ?"
        with self._connect() as conn:
            cursor = conn.execute(sql, (exp_hash,))
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("実験を削除しました: hash=%s", exp_hash[:8])
        return deleted

    def update_notes(self, exp_hash: str, notes: str) -> bool:
        """実験のメモを更新する。"""
        sql = "UPDATE experiments SET notes = ? WHERE exp_hash = ?"
        with self._connect() as conn:
            cursor = conn.execute(sql, (notes, exp_hash))
        return cursor.rowcount > 0
