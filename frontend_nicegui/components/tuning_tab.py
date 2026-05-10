"""
frontend_nicegui/components/tuning_tab.py

ハイパーパラメータチューニングUI。

EstimatorConfigDialogで設定したGridSearchCV/OptunaSearchCV探索空間を
使って、ベストモデルの精密チューニングを実行する。

Usage:
    from frontend_nicegui.components.tuning_tab import render_tuning_tab
    render_tuning_tab(state)

Implements:
    F-TUN01: GridSearchCV / Optuna チューニング実行UI
    F-TUN02: 探索空間ビジュアライズ + 編集
    F-TUN03: チューニング結果表示
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui, run

logger = logging.getLogger(__name__)


def render_tuning_tab(state: dict[str, Any]) -> None:
    """ハイパーパラメータチューニングタブを描画する。

    Args:
        state: アプリケーション共有状態
    """
    ar = state.get("automl_result")
    if ar is None:
        with ui.card().classes("glass-card q-pa-xl full-width text-center"):
            ui.icon("tune", color="grey-7", size="xl").classes("q-mb-md")
            ui.label("先に順解析を実行してください").classes("text-h6 text-grey-5")
            ui.label(
                "ベストモデルが決まった後に、Grid/Optunaでの精密チューニングが可能です。"
            ).classes("text-grey-6 q-mt-sm")
        return

    best_key = ar.best_model_key
    model_configs = state.get("model_configs", {})

    # 1. ベストモデルの設定状況
    with ui.card().classes("glass-card q-pa-md full-width q-mb-md"):
        with ui.row().classes("items-center q-gutter-md"):
            ui.icon("emoji_events", color="amber", size="md")
            ui.label(f"🏆 ベストモデル: {best_key}").classes("text-h6 text-bold")
            ui.badge(f"スコア: {ar.best_score:.4f}", color="cyan").props("dense")

        # 探索空間の設定状況
        best_config = model_configs.get(best_key)
        has_grid = bool(best_config and hasattr(best_config, "grid_space") and best_config.grid_space)
        has_optuna = bool(best_config and hasattr(best_config, "optuna_space") and best_config.optuna_space)

        with ui.row().classes("q-gutter-sm q-mt-sm"):
            ui.badge(
                f"GridSearch: {'✅ {0}パラメータ設定済'.format(len(best_config.grid_space)) if has_grid else '❌ 未設定'}",
                color="teal" if has_grid else "grey",
            ).props("outline")
            ui.badge(
                f"Optuna: {'✅ {0}パラメータ設定済'.format(len(best_config.optuna_space)) if has_optuna else '❌ 未設定'}",
                color="purple" if has_optuna else "grey",
            ).props("outline")

        if not has_grid and not has_optuna:
            ui.label(
                "💡 パイプライン設定 → 推定器タブの ⚙️ ボタンから"
                f" {best_key} の Grid/Optuna 探索範囲を設定してください。"
            ).classes("text-caption text-amber q-mt-sm")

            # 自動設定ボタン
            def _auto_setup():
                _auto_generate_tuning_config(state, best_key)
                ui.notify("✅ デフォルトの探索空間を自動生成しました", type="positive")

            ui.button(
                "🔧 ベストモデルの探索空間を自動生成",
                on_click=_auto_setup,
            ).props("outline color=cyan no-caps").classes("q-mt-sm")

    # 2. チューニング手法の選択
    tuning_method = state.get("_tuning_method", "grid")
    with ui.card().classes("glass-card q-pa-md full-width q-mb-md"):
        ui.label("⚙️ チューニング手法").classes("text-subtitle1 text-bold q-mb-sm")

        with ui.row().classes("q-gutter-md items-center"):
            method_select = ui.select(
                label="手法",
                options={
                    "grid": "🔍 GridSearchCV（全探索）",
                    "random": "🎲 RandomizedSearchCV",
                    "optuna": "⚡ Optuna（ベイズ最適化）",
                    "halving_grid": "📊 HalvingGridSearchCV",
                    "halving_random": "📊 HalvingRandomSearchCV",
                },
                value=tuning_method,
                on_change=lambda e: state.update({"_tuning_method": e.value}),
            ).classes("w-64").props("outlined dense")

        with ui.row().classes("q-gutter-sm q-mt-sm"):
            ui.number(
                "CV folds",
                value=state.get("_tuning_cv", 5),
                min=2, max=30, step=1,
                on_change=lambda e: state.update({"_tuning_cv": int(e.value)}),
            ).classes("w-28").props("outlined dense")

            ui.number(
                "試行回数（Random/Optuna）",
                value=state.get("_tuning_n_iter", 50),
                min=5, max=1000, step=5,
                on_change=lambda e: state.update({"_tuning_n_iter": int(e.value)}),
            ).classes("w-40").props("outlined dense")

            ui.select(
                label="スコアリング",
                options=[
                    "neg_root_mean_squared_error",
                    "neg_mean_absolute_error",
                    "neg_mean_squared_error",
                    "r2",
                    "accuracy",
                    "f1_weighted",
                ],
                value=state.get("_tuning_scoring", "neg_root_mean_squared_error"),
                on_change=lambda e: state.update({"_tuning_scoring": e.value}),
            ).classes("w-64").props("outlined dense")

    # 3. 探索空間プレビュー
    best_config = model_configs.get(best_key)
    if best_config:
        with ui.card().classes("glass-card q-pa-md full-width q-mb-md"):
            ui.label("🔍 探索空間プレビュー").classes("text-subtitle1 text-bold q-mb-sm")

            method = state.get("_tuning_method", "grid")
            if method in ("grid", "halving_grid") and hasattr(best_config, "grid_space"):
                _render_grid_preview(best_config.grid_space)
            elif method in ("optuna", "random", "halving_random") and hasattr(best_config, "optuna_space"):
                _render_optuna_preview(best_config.optuna_space)
            else:
                ui.label("該当する探索空間が未設定です").classes("text-grey-5")

            # 直接編集ボタン
            def _open_editor():
                from frontend_nicegui.components.estimator_config_dialog import EstimatorConfigDialog
                from backend.models.factory import get_model_registry
                task = state.get("task_type", "regression")
                registry = get_model_registry(task=task)
                entry = registry.get(best_key, {})
                cls = entry.get("class")
                if cls is None:
                    ui.notify("このモデルのクラスが取得できません", type="warning")
                    return
                dialog = EstimatorConfigDialog(
                    model_key=best_key,
                    model_cls=cls,
                    model_name=best_key,
                    initial_config=best_config,
                    on_save=lambda cfg: state["model_configs"].update({best_key: cfg}),
                )
                dialog.open()

            ui.button(
                "✏️ 探索空間を編集",
                on_click=_open_editor,
            ).props("outline color=cyan size=sm no-caps").classes("q-mt-sm")

    # 4. 実行ボタン
    tuning_container = ui.column().classes("full-width")

    async def _run_tuning():
        await _execute_tuning(state, tuning_container)

    with ui.row().classes("q-gutter-sm q-mt-md"):
        ui.button(
            "🚀 チューニング実行",
            on_click=_run_tuning,
        ).props("color=primary no-caps").classes("q-px-lg text-bold")

    # 5. 前回の結果表示
    prev_result = state.get("_tuning_result")
    if prev_result:
        with ui.card().classes("glass-card q-pa-md full-width q-mt-md"):
            _render_tuning_result(prev_result)


# ============================================================
# 探索空間プレビュー
# ============================================================

def _render_grid_preview(grid_space: dict[str, list[Any]]) -> None:
    """GridSearchCV探索空間のプレビュー。"""
    if not grid_space:
        ui.label("GridSearch探索空間が空です").classes("text-grey-5")
        return

    total_combos = 1
    rows = []
    for name, values in grid_space.items():
        total_combos *= len(values)
        rows.append({
            "パラメータ": name,
            "候補値": ", ".join(str(v) for v in values),
            "候補数": str(len(values)),
        })

    cols = [
        {"name": c, "label": c, "field": c, "align": "left" if c == "パラメータ" else "center", "sortable": True}
        for c in ["パラメータ", "候補値", "候補数"]
    ]
    ui.table(columns=cols, rows=rows).classes("full-width").props("dense flat bordered")
    ui.label(f"合計: {total_combos:,}通りの組み合わせ").classes("text-caption text-grey-5 q-mt-xs")


def _render_optuna_preview(optuna_space: dict[str, dict[str, Any]]) -> None:
    """OptunaSearchCV探索空間のプレビュー。"""
    if not optuna_space:
        ui.label("Optuna探索空間が空です").classes("text-grey-5")
        return

    rows = []
    for name, spec in optuna_space.items():
        t = spec.get("type", "float")
        if t == "categorical":
            detail = f"候補: {', '.join(str(c) for c in spec.get('choices', []))}"
        else:
            detail = f"[{spec.get('low', '?')}, {spec.get('high', '?')}]"
            if spec.get("log"):
                detail += " (log)"
            if spec.get("step"):
                detail += f" step={spec['step']}"
        rows.append({
            "パラメータ": name,
            "型": t,
            "範囲": detail,
        })

    cols = [
        {"name": c, "label": c, "field": c, "align": "left", "sortable": True}
        for c in ["パラメータ", "型", "範囲"]
    ]
    ui.table(columns=cols, rows=rows).classes("full-width").props("dense flat bordered")


# ============================================================
# チューニング実行
# ============================================================

async def _execute_tuning(state: dict, container) -> None:
    """チューニングを実行する。"""
    ar = state.get("automl_result")
    if ar is None:
        ui.notify("先に順解析を実行してください", type="warning")
        return

    best_key = ar.best_model_key
    model_configs = state.get("model_configs", {})
    best_config = model_configs.get(best_key)

    # 探索空間がなければ自動生成
    if best_config is None:
        _auto_generate_tuning_config(state, best_key)
        best_config = model_configs.get(best_key)

    if best_config is None:
        ui.notify("探索空間の生成に失敗しました", type="negative")
        return

    method = state.get("_tuning_method", "grid")
    cv = state.get("_tuning_cv", 5)
    n_iter = state.get("_tuning_n_iter", 50)
    scoring = state.get("_tuning_scoring", "neg_root_mean_squared_error")

    # param_gridの決定
    if method in ("grid", "halving_grid"):
        if not hasattr(best_config, "grid_space") or not best_config.grid_space:
            ui.notify("GridSearch探索空間が設定されていません。⚙️ ボタンから設定してください。", type="warning")
            return
        param_grid = best_config.grid_space
    else:
        if not hasattr(best_config, "optuna_space") or not best_config.optuna_space:
            ui.notify("Optuna探索空間が設定されていません。⚙️ ボタンから設定してください。", type="warning")
            return
        param_grid = best_config.optuna_space

    # モデル取得（Pipelineごと渡す → tune()が自動でestimator__プレフィックスを付与）
    pipeline = getattr(ar, "best_pipeline", None)
    if pipeline is None:
        ui.notify("パイプラインが取得できません", type="warning")
        return

    # データ取得
    X = getattr(ar, "processed_X", None)
    y = getattr(ar, "y_train", None) or getattr(ar, "oof_true", None)
    if X is None or y is None:
        ui.notify("データの取得に失敗しました", type="warning")
        return

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = np.asarray(y).ravel()

    # 進捗UI
    container.clear()
    with container:
        with ui.card().classes("glass-card q-pa-md full-width"):
            progress_label = ui.label(
                f"⏳ {method.upper()} チューニング実行中..."
            ).classes("text-lg text-bold")
            progress_bar = ui.linear_progress(value=0, show_value=False).props(
                "color=purple rounded"
            )
            progress_detail = ui.label("").classes("text-caption text-grey-5")

    start_time = time.time()

    try:
        from backend.models.tuner import TunerConfig, tune

        config = TunerConfig(
            method=method,
            param_grid=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )

        progress_bar.value = 0.3
        progress_label.text = f"⏳ {method.upper()} チューニング中... (CV={cv}, scoring={scoring})"

        # Pipeline全体を渡す（tune()が estimator__ プレフィックスを自動処理）
        result = await run.io_bound(
            tune,
            pipeline,
            X_arr,
            y_arr,
            config,
        )

        elapsed = time.time() - start_time

        # 結果保存
        tuning_result = {
            "method": method,
            "best_params": result["best_params"],
            "best_score": result["best_score"],
            "cv_results": result.get("cv_results"),
            "elapsed": elapsed,
            "best_estimator": result.get("best_estimator"),
            "model_key": best_key,
        }
        state["_tuning_result"] = tuning_result

        # 成功表示
        container.clear()
        with container:
            with ui.card().classes("glass-card q-pa-md full-width best-model-glow"):
                _render_tuning_result(tuning_result)

        ui.notify(
            f"✅ チューニング完了！ スコア: {result['best_score']:.4f} ({elapsed:.1f}秒)",
            type="positive",
            timeout=5000,
        )

    except Exception as ex:
        elapsed = time.time() - start_time
        container.clear()
        with container:
            with ui.card().classes("q-pa-md full-width").style(
                "border: 1px solid rgba(248,113,113,0.3); background: rgba(60,20,20,0.2);"
            ):
                ui.label(f"❌ チューニングエラー: {str(ex)[:200]}").classes("text-red")
                import traceback
                with ui.expansion("🔍 詳細", icon="bug_report"):
                    ui.code(traceback.format_exc()[-800:]).classes("full-width")
        ui.notify(f"チューニングエラー: {str(ex)[:100]}", type="negative")
        logger.error("Tuning error: %s", ex, exc_info=True)


# ============================================================
# チューニング結果表示
# ============================================================

def _render_tuning_result(result: dict) -> None:
    """チューニング結果を表示する。"""
    with ui.row().classes("items-center q-gutter-md"):
        ui.icon("auto_awesome", color="purple", size="md")
        ui.label(f"🎯 チューニング結果: {result['model_key']}").classes("text-h6 text-bold")
        ui.badge(f"手法: {result['method']}", color="purple").props("outline")

    with ui.row().classes("q-gutter-md q-mt-sm"):
        for val, lbl in [
            (f"{result['best_score']:.4f}", "ベストスコア"),
            (f"{result['elapsed']:.1f}秒", "所要時間"),
            (f"{len(result['best_params'])}個", "最適化パラメータ数"),
        ]:
            with ui.card().classes("q-pa-xs").style(
                "min-width: 100px; background: rgba(156,39,176,0.1); border-radius: 8px;"
            ):
                ui.label(val).classes("text-subtitle1 text-bold hero-gradient")
                ui.label(lbl).classes("text-caption text-grey-5")

    # ベストパラメータ
    ui.separator().classes("q-my-sm")
    ui.label("🔧 最適パラメータ").classes("text-subtitle2 text-bold q-mt-sm")
    rows = [
        {"パラメータ": k, "最適値": str(v)}
        for k, v in result["best_params"].items()
    ]
    cols = [
        {"name": c, "label": c, "field": c, "align": "left"}
        for c in ["パラメータ", "最適値"]
    ]
    ui.table(columns=cols, rows=rows).classes("full-width").props("dense flat bordered")

    # CV結果テーブル（top 10）
    cv_df = result.get("cv_results")
    if cv_df is not None and isinstance(cv_df, pd.DataFrame):
        with ui.expansion("📊 全試行結果（上位10）", icon="table_chart").classes("full-width q-mt-sm"):
            if "mean_test_score" in cv_df.columns:
                show_df = cv_df.nlargest(10, "mean_test_score")
            elif "value" in cv_df.columns:
                show_df = cv_df.nlargest(10, "value")
            else:
                show_df = cv_df.head(10)

            display_cols = [c for c in show_df.columns if not c.startswith("split") and "time" not in c.lower()][:8]
            rows_cv = show_df[display_cols].round(4).to_dict("records")
            for r in rows_cv:
                for k, v in r.items():
                    if isinstance(v, float):
                        r[k] = f"{v:.4f}"
                    else:
                        r[k] = str(v)

            cv_cols = [
                {"name": c, "label": c, "field": c, "align": "left", "sortable": True}
                for c in display_cols
            ]
            ui.table(columns=cv_cols, rows=rows_cv).classes("full-width").props("dense flat bordered")


# ============================================================
# 探索空間自動生成
# ============================================================

def _auto_generate_tuning_config(state: dict, model_key: str) -> None:
    """ベストモデルの探索空間を自動生成する。"""
    from backend.models.factory import get_model_registry
    from backend.ui.param_schema import introspect_params
    from backend.models.search_space_generator import (
        generate_grid_space,
        generate_optuna_space,
    )
    from frontend_nicegui.components.estimator_config_dialog import EstimatorConfig

    task = state.get("task_type", "regression")
    try:
        registry = get_model_registry(task=task)
    except Exception:
        registry = {}

    entry = registry.get(model_key, {})
    cls = entry.get("class")
    if cls is None:
        logger.warning("モデルクラスが取得できません: %s", model_key)
        return

    specs = introspect_params(cls)
    grid = generate_grid_space(specs, include_advanced=True)
    optuna = generate_optuna_space(specs, include_advanced=True)

    config = EstimatorConfig(
        model_key=model_key,
        model_cls=cls,
        grid_space=grid,
        optuna_space=optuna,
    )

    if "model_configs" not in state:
        state["model_configs"] = {}
    state["model_configs"][model_key] = config
