-- backend/db/schema.sql
-- ChemAI ML Studio: SQLite実験バージョン管理スキーマ
-- hashlib.sha256 で生成した実験ハッシュを主キーとして使用する

CREATE TABLE IF NOT EXISTS experiments (
    -- 実験の一意識別子 (hashlib.sha256 of hyperparams+config)
    exp_hash            TEXT PRIMARY KEY,

    -- タイムスタンプ (ISO 8601形式 UTC)
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),

    -- 実験名 (ユーザーが任意に付与可能)
    exp_name            TEXT NOT NULL DEFAULT 'Unnamed',

    -- データセット情報
    dataset_name        TEXT,
    n_samples           INTEGER,
    n_features          INTEGER,

    -- タスク設定
    task_type           TEXT NOT NULL,   -- 'regression' | 'classification'
    target_col          TEXT NOT NULL,
    cv_folds            INTEGER NOT NULL DEFAULT 5,
    cv_strategy         TEXT NOT NULL DEFAULT 'KFold',

    -- 最良モデル情報
    best_model_key      TEXT NOT NULL,
    best_score          REAL NOT NULL,
    scoring_metric      TEXT NOT NULL,

    -- 全評価指標 (JSON文字列)
    -- {"R2": 0.95, "RMSE": 0.12, "MAE": 0.09, "MAPE": 1.5}
    metrics_json        TEXT,

    -- ハイパーパラメータ (JSON文字列)
    -- {"n_estimators": 300, "max_depth": 7, ...}
    hyperparams_json    TEXT,

    -- 前処理設定 (JSON文字列)
    -- {"scaler": "standard", "imputer": "median", ...}
    preprocess_json     TEXT,

    -- SMILES記述子エンジン情報 (JSON文字列)
    -- ["RDKitAdapter", "MordredAdapter"]
    engines_json        TEXT,

    -- 所要時間 (秒)
    elapsed_seconds     REAL,

    -- ユーザーメモ (自由記述)
    notes               TEXT DEFAULT ''
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_task_type  ON experiments (task_type);
CREATE INDEX IF NOT EXISTS idx_experiments_best_score ON experiments (best_score DESC);
