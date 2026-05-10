"""
backend/mlops/mlflow_manager.py

MLflow を使った実験トラッキング・モデル管理モジュール。
UI非依存の純粋Pythonとして実装し、Streamlit/Djangoから呼び出す。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.utils.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_DIR,
)
from backend.utils.optional_import import safe_import

_mlflow = safe_import("mlflow", "mlflow")

logger = logging.getLogger(__name__)


class MLflowManager:
    """
    MLflow との連携を管理するクラス。

    Implements: 要件定義書 §3.12 MLOps

    mlflow が未インストールの場合は全メソッドが警告ログを出力して
    ノーオペレーションを返す（Graceful Degradation）。

    Args:
        tracking_uri: MLflow Tracking ServerのURI
        experiment_name: 実験名
    """

    def __init__(
        self,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._run_id: str | None = None

        if not _mlflow:
            logger.warning("mlflow が未インストールのため実験トラッキングは無効です。")
            return

        import mlflow  # type: ignore
        mlflow.set_tracking_uri(tracking_uri)
        # 実験が存在しない場合は作成
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow 初期化: {tracking_uri} / {experiment_name}")

    # ---- ラン制御 ----

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """
        新しい MLflow ランを開始して run_id を返す。

        Args:
            run_name: ラン名（省略時は自動命名）
            tags: MLflowタグ

        Returns:
            run_id (str) または None（mlflow未使用時）
        """
        if not _mlflow:
            return None

        import mlflow  # type: ignore
        run = mlflow.start_run(run_name=run_name, tags=tags or {})
        self._run_id = run.info.run_id
        logger.info(f"MLflow ラン開始: run_id={self._run_id}")
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """現在のランを終了する。"""
        if not _mlflow or self._run_id is None:
            return

        import mlflow  # type: ignore
        mlflow.end_run(status=status)
        logger.info(f"MLflow ラン終了: run_id={self._run_id}")
        self._run_id = None

    def fail_run(self, exc: Exception | None = None) -> None:
        """ランをFAILED状態で終了する。"""
        if exc:
            logger.error(f"MLflow ランをFAILEDとして終了: {exc}")
        self.end_run("FAILED")

    # ---- ロギング ----

    def log_params(self, params: dict[str, Any]) -> None:
        """
        パラメータを現在のランに記録する。

        Args:
            params: {パラメータ名: 値} の辞書（値はstr/int/float/bool）
        """
        if not _mlflow:
            return
        import mlflow  # type: ignore
        # MLflowはパラメータ値を文字列に変換する（最大500文字）
        flat = {k: str(v)[:500] for k, v in params.items()}
        mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        メトリクスを記録する。

        Args:
            metrics: {メトリクス名: 値} の辞書
            step: ステップ番号（学習曲線等で使用）
        """
        if not _mlflow:
            return
        import mlflow  # type: ignore
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """
        ファイルをアーティファクトとして記録する。

        Args:
            local_path: ローカルファイルパス
            artifact_path: MLflow内の保存ディレクトリ名
        """
        if not _mlflow:
            return
        import mlflow  # type: ignore
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_figure(self, fig: Any, filename: str) -> None:
        """
        matplotlib/plotly figureをアーティファクトとして記録する。

        Args:
            fig: matplotlib.figure.Figure または plotly.graph_objects.Figure
            filename: 保存ファイル名（拡張子含む）
        """
        if not _mlflow:
            return
        import mlflow  # type: ignore
        mlflow.log_figure(fig, filename)

    # ---- モデル管理 ----

    def save_model(
        self,
        model: Any,
        model_name: str,
        save_dir: str | Path | None = None,
        register: bool = False,
    ) -> Path:
        """
        モデルを joblib で保存し、MLflow にアーティファクトとして登録する。

        Args:
            model: 保存するモデル
            model_name: モデル名（ファイル名のベース）
            save_dir: 保存ディレクトリ（省略時は mlruns/ 内に作成）
            register: True の場合 MLflow Model Registry に登録する

        Returns:
            保存されたファイルのパス
        """
        save_dir = Path(save_dir) if save_dir else MLFLOW_TRACKING_DIR / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{model_name}.joblib"

        joblib.dump(model, model_path)
        logger.info(f"モデル保存: {model_path}")

        if _mlflow:
            import mlflow  # type: ignore
            self.log_artifact(model_path, artifact_path="models")
            if register:
                try:
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path=f"model_{model_name}",
                        registered_model_name=model_name,
                    )
                    logger.info(f"MLflow Model Registry に登録: {model_name}")
                except Exception as e:
                    logger.warning(f"Model Registry 登録失敗: {e}")

        return model_path

    def load_model(self, model_path: str | Path) -> Any:
        """
        joblib 形式のモデルを読み込む。

        Args:
            model_path: モデルファイルのパス

        Returns:
            読み込まれたモデル
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
        model = joblib.load(path)
        logger.info(f"モデル読み込み: {path}")
        return model

    # ---- 実験一覧 ----

    def get_experiment_runs(
        self,
        max_results: int = 100,
        order_by: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        実験のラン一覧をDataFrameで返す（GUI表示用）。

        Args:
            max_results: 最大取得件数
            order_by: ソート条件（例: ["metrics.rmse ASC"]）

        Returns:
            ランのDataFrame（空のDataFrameを返す場合もあり）
        """
        if not _mlflow:
            return pd.DataFrame()

        import mlflow  # type: ignore
        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            max_results=max_results,
            order_by=order_by or ["start_time DESC"],
        )
        return runs

    def get_best_run(
        self,
        metric_name: str,
        ascending: bool = True,
    ) -> dict[str, Any] | None:
        """
        指定メトリクスで最良のランを返す。

        Args:
            metric_name: 評価メトリクス名（例: "rmse"）
            ascending: True で昇順（小さいほど良い場合）

        Returns:
            ランの情報辞書、またはNone
        """
        if not _mlflow:
            return None

        df = self.get_experiment_runs(max_results=1000)
        col = f"metrics.{metric_name}"
        if col not in df.columns or df.empty:
            return None

        df_sorted = df.dropna(subset=[col]).sort_values(col, ascending=ascending)
        if df_sorted.empty:
            return None

        best = df_sorted.iloc[0]
        return best.to_dict()


# ── コンテキストマネージャとしての使用をサポートするユーティリティ ──

class MLRunContext:
    """
    with 文で MLflow ランを安全に管理するコンテキストマネージャ。

    Usage:
        with MLRunContext(manager, run_name="exp1") as run_id:
            manager.log_params({"lr": 0.01})
            manager.log_metrics({"rmse": 0.05})
    """

    def __init__(
        self,
        manager: MLflowManager,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.manager = manager
        self.run_name = run_name
        self.tags = tags
        self.run_id: str | None = None

    def __enter__(self) -> str | None:
        self.run_id = self.manager.start_run(self.run_name, self.tags)
        return self.run_id

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.manager.fail_run(exc_val)
        else:
            self.manager.end_run()
