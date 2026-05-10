"""
backend/utils/config.py

アプリケーション全体の設定を一元管理するモジュール。
random_stateは全モジュールでこのファイルの値を参照すること。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================
# プロジェクトルート
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DESCRIPTOR_RULES_DIR = PROJECT_ROOT / "descriptor_rules"
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"

# ============================================================
# 再現性
# ============================================================
RANDOM_STATE: int = 42

# ============================================================
# 変数型自動判定の閾値
# ============================================================
TYPE_DETECTOR_CARDINALITY_THRESHOLD: int = 20   # これ以下はカテゴリ(少)
TYPE_DETECTOR_SKEWNESS_THRESHOLD: float = 0.5   # これ以上は対数/冪乗候補
TYPE_DETECTOR_OUTLIER_IQR_FACTOR: float = 1.5   # IQR外れ値判定係数

# ============================================================
# AutoML 設定
# ============================================================
AUTOML_CV_FOLDS: int = 5
AUTOML_TIMEOUT_SECONDS: int = 600               # 最大学習時間(秒)
AUTOML_N_JOBS: int = -1                         # 並列数(-1=全コア)

# ============================================================
# SHAP 設定
# ============================================================
SHAP_MAX_DISPLAY: int = 20                      # SummaryPlotの上位特徴量数
SHAP_KERNEL_NSAMPLES: int = 100                 # KernelExplainerのサンプル数
SHAP_KERNEL_NSAMPLES_MAX: int = 500             # これ以上の行数なら代表点サンプリング実行

# ============================================================
# MLflow 設定
# ============================================================
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_TRACKING_DIR}/mlflow.db"
)
MLFLOW_EXPERIMENT_NAME: str = "ml_gui_app"

# ============================================================
# Django / Celery 設定 (環境変数で上書き可能)
# ============================================================
DJANGO_SECRET_KEY: str = os.getenv(
    "DJANGO_SECRET_KEY",
    "UNSAFE-default-key-change-me-in-production"
)
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{PROJECT_ROOT}/db.sqlite3"   # デフォルトはSQLite（ローカル用）
)
CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# ============================================================
# 化合物特徴量化設定
# ============================================================
# stlite（ブラウザ環境）かどうかの判定フラグ（Streamlitから注入される）
IS_STLITE: bool = bool(os.getenv("ML_GUI_STLITE", ""))

# xtb実行可能ファイルのパス（環境変数で指定）
XTB_EXECUTABLE: str = os.getenv("XTB_EXECUTABLE", "xtb")
CREST_EXECUTABLE: str = os.getenv("CREST_EXECUTABLE", "crest")

XTB_TIMEOUT_SECONDS_PER_MOL: int = 30           # 1分子あたりのxTB実行タイムアウト(秒)
MAX_DESCRIPTOR_SELECTION: int = 100             # UI等での記述子選択上限数

# ============================================================
# パフォーマンス要件
# ============================================================
EDA_SMALL_NROWS: int = 1_000        # EDA高速モードの閾値
EDA_LARGE_NROWS: int = 50_000       # 巨大データ警告の閾値
PCA_DEFAULT_COMPONENTS: int = 2     # デフォルトのPCA次元数


@dataclass
class AppConfig:
    """
    アプリケーション全体の設定をまとめるデータクラス。
    環境に応じた設定の切り替えができる。
    """
    random_state: int = RANDOM_STATE
    automl_cv_folds: int = AUTOML_CV_FOLDS
    automl_timeout: int = AUTOML_TIMEOUT_SECONDS
    n_jobs: int = AUTOML_N_JOBS
    shap_max_display: int = SHAP_MAX_DISPLAY
    shap_kernel_nsamples: int = SHAP_KERNEL_NSAMPLES
    shap_kernel_nsamples_max: int = SHAP_KERNEL_NSAMPLES_MAX
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI
    mlflow_experiment_name: str = MLFLOW_EXPERIMENT_NAME
    is_stlite: bool = IS_STLITE
    cardinality_threshold: int = TYPE_DETECTOR_CARDINALITY_THRESHOLD
    skewness_threshold: float = TYPE_DETECTOR_SKEWNESS_THRESHOLD
    outlier_iqr_factor: float = TYPE_DETECTOR_OUTLIER_IQR_FACTOR
    xtb_timeout_per_mol: int = XTB_TIMEOUT_SECONDS_PER_MOL
    max_descriptor_selection: int = MAX_DESCRIPTOR_SELECTION
    extra: dict = field(default_factory=dict)


# デフォルトのグローバル設定インスタンス
default_config = AppConfig()
