-- 混合物機能のためのスキーマ定義

CREATE TABLE IF NOT EXISTS mixture_sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    ratio_type TEXT NOT NULL, -- 'weight', 'mole', or 'other'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mixture_components (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    smiles TEXT NOT NULL,
    ratio_input REAL NOT NULL,
    ratio_weight_fraction REAL,
    ratio_mole_fraction REAL,
    FOREIGN KEY(session_id) REFERENCES mixture_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS feature_weighting_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL UNIQUE,
    weighting_type TEXT NOT NULL, -- 'weight', 'mole', 'context'
    rationale TEXT
);
