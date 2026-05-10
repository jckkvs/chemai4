"""
frontend_nicegui/components/analysis_runner.py

解析実行コンポーネント: AutoMLEngine呼び出しとリアルタイム進捗表示。
run.io_bound で重い計算をバックグラウンドスレッドにオフロードし、
NiceGUI の WebSocket heartbeat をブロックしない。
"""
from __future__ import annotations

import asyncio
import logging
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui, run

# LLMアドバイザー（遅延インポート）
try:
    from backend.llm.analysis_advisor import AnalysisAdvisor
    _HAS_ADVISOR = True
except ImportError:
    _HAS_ADVISOR = False

logger = logging.getLogger(__name__)

# バックエンドパスを追加（スレッドでも有効）
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    logger.info(f"プロジェクトルートを追加: {_project_root}")
logger.debug(f"sys.path: {sys.path[:3]}...")

# 解析ロック（二重実行防止）
_analysis_running = False
_cancel_requested = False


class AnalysisCancelled(Exception):
    """解析キャンセル時に送出される例外。"""
    pass


def _run_engine_sync(
    df_work: pd.DataFrame,
    target_col: str,
    smiles_col: str | None,
    group_col: str | None,
    task: str,
    model_keys: list[str] | None,
    cv_folds: int,
    timeout: int,
    selected_desc: list[str] | None,
    progress_queue: queue.Queue,
    *,
    active_engines: list[str] | None = None,
    cv_key: str = "auto",
    model_params: dict[str, dict] | None = None,
    preprocess_params: dict[str, Any] | None = None,
    monotonic_constraints: dict[str, int] | None = None,
    column_meta_dict: dict | None = None,
    count_normalization: str = "density",
    # Morgan フィンガープリント オプション
    morgan_count: bool = False,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
    morgan_order_by_appearance: bool = False,
) -> Any:
    """
    バックグラウンドスレッドで AutoMLEngine を実行する同期関数。
    """
    from backend.models.automl import AutoMLEngine

    def progress_callback(step: int, total: int, msg: str) -> None:
        global _cancel_requested
        if _cancel_requested:
            raise AnalysisCancelled("ユーザーが解析をキャンセルしました")
        try:
            progress_queue.put_nowait(("progress", step, total, msg))
        except queue.Full:
            pass 

    engine = AutoMLEngine(
        task=task,
        cv_folds=cv_folds,
        cv_key=cv_key,
        model_keys=model_keys if model_keys else None,
        model_params=model_params,
        preprocess_params=preprocess_params,
        timeout_seconds=timeout,
        progress_callback=progress_callback,
        selected_descriptors=selected_desc,
        active_engines=active_engines,
        monotonic_constraints_dict=monotonic_constraints,
        column_meta_dict=column_meta_dict,
        count_normalization=count_normalization,
        # Morgan フィンガープリント オプション
        morgan_count=morgan_count,
        morgan_radius=morgan_radius,
        morgan_bits=morgan_bits,
        morgan_order_by_appearance=morgan_order_by_appearance,
    )

    result = engine.run(
        df_work,
        target_col=target_col,
        smiles_col=smiles_col,
        group_col=group_col,
    )

    return result


async def run_analysis(state: dict[str, Any], status_container, on_complete=None) -> None:
    """
    AutoML解析を実行し、結果をstateに保存する。

    重い計算は run.io_bound でバックグラウンドスレッドにオフロードし、
    NiceGUI の WebSocket heartbeat をブロックしない。
    進捗は queue.Queue + ui.timer でポーリング更新する。

    Args:
        state: 共有ステート辞書
        status_container: 進捗表示を描画するUIコンテナ
        on_complete: 完了時のコールバック
    """
    global _analysis_running

    df = state.get("df")
    target_col = state.get("target_col")

    if df is None or not target_col:
        ui.notify("データと目的変数を設定してください", type="warning")
        return

    if _analysis_running:
        ui.notify("⏳ 解析が既に実行中です。下のボタンで中断できます。", type="info")
        # 強制停止ボタンを控えめに表示
        with status_container:
            with ui.row().classes("items-center q-gutter-sm q-mt-sm"):
                ui.label("⏳ 前回の解析がまだ実行中です").classes("text-caption text-grey-5")
                def _force_stop():
                    global _cancel_requested, _analysis_running
                    _cancel_requested = True
                    _analysis_running = False
                    ui.notify("🛑 解析を強制停止しました。再実行できます。", type="warning")
                ui.button("🛑 前回の解析を中断", on_click=_force_stop).props(
                    "flat dense size=sm no-caps color=orange"
                ).tooltip("実行中の解析にキャンセル要求を送り、新しい解析を開始可能にします")
        return

    _analysis_running = True
    _cancel_requested = False

    # キャンセルハンドラ
    def _on_cancel():
        global _cancel_requested
        _cancel_requested = True
        cancel_btn.disable()
        cancel_btn.text = "⏳ 中断処理中..."
        progress_label.text = "⏳ キャンセル要求を送信しました..."
        ui.notify("🛑 解析キャンセルを要求しました。現在のステップ完了後に停止します。", type="warning", timeout=5000)

    # 進捗表示の構築
    status_container.clear()
    with status_container:
        with ui.card().classes("full-width glass-card q-pa-md q-mb-sm"):
            progress_header = ui.row().classes("items-center full-width justify-between")
            with progress_header:
                progress_label = ui.label("⏳ 解析を開始しています...").classes("text-body2")
                with ui.row().classes("items-center q-gutter-sm"):
                    progress_pct = ui.label("").classes("text-h6 text-bold hero-gradient")
                    cancel_btn = ui.button(
                        "🛑 中断", on_click=_on_cancel,
                    ).props("outline color=red size=sm no-caps").tooltip(
                        "現在のモデル学習ステップ完了後に解析を安全に停止します"
                    )
            progress_bar = ui.linear_progress(value=0, show_value=False).classes("q-mb-xs").props("color=cyan rounded")
            with ui.row().classes("justify-between full-width"):
                progress_detail = ui.label("").classes("text-caption text-grey-5")
                progress_eta = ui.label("").classes("text-caption text-grey-5")
            # ── リソースモニター（CPU / メモリ）──
            with ui.row().classes("items-center q-gutter-md q-mt-xs").style("opacity: 0.7;"):
                resource_cpu_lbl = ui.label("CPU: —").classes("text-caption text-grey-6").style("font-size: 0.72rem;")
                resource_mem_lbl = ui.label("MEM: —").classes("text-caption text-grey-6").style("font-size: 0.72rem;")

        # ── LLM解析アドバイス表示エリア──
        if _HAS_ADVISOR:
            with ui.card().classes("full-width glass-card q-pa-sm q-mt-sm").style(
                "border-left: 3px solid rgba(123, 47, 247, 0.7);"
            ):
                advice_header = ui.row().classes("items-center justify-between full-width")
                with advice_header:
                    ui.label("🤖 AI解析アドバイス").classes("text-body2 text-purple font-weight-bold")
                    advice_status = ui.label("⏳ 生成中...").classes("text-caption text-grey-5")
                advice_content = ui.label("").classes("text-caption").style("white-space: pre-wrap;")
                advice_content.visible = False

    # ── リソースモニタータイマー（1秒間隔）──
    def _poll_resources():
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            resource_cpu_lbl.text = f"CPU: {cpu:.0f}%"
            resource_mem_lbl.text = f"MEM: {mem.percent:.0f}% ({mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB)"
        except Exception:
            pass
    resource_timer = ui.timer(1.0, _poll_resources)

    # 進捗キュー（スレッドセーフ）
    progress_queue: queue.Queue = queue.Queue(maxsize=100)
    _start_time = time.time()

    # 進捗ポーリングタイマー（最新のステップのみ表示、羅列しない）
    def _poll_progress():
        """キューから進捗情報を取得してUI更新（最新のみ）"""
        latest = None
        while True:
            try:
                item = progress_queue.get_nowait()
                if item[0] == "progress":
                    latest = item
            except queue.Empty:
                break
        if latest:
            _, step, total, msg = latest
            pct = step / total if total > 0 else 0
            progress_bar.value = pct
            progress_pct.text = f"{int(pct * 100)}%"
            progress_label.text = f"⏳ {msg}"
            progress_detail.text = f"ステップ {step}/{total}"
            # 推定残り時間
            elapsed = time.time() - _start_time
            if pct > 0.05:
                eta_sec = elapsed / pct * (1 - pct)
                if eta_sec < 60:
                    progress_eta.text = f"残り約{eta_sec:.0f}秒"
                else:
                    progress_eta.text = f"残り約{eta_sec/60:.1f}分"
            else:
                progress_eta.text = "推定中..."

    timer = ui.timer(0.5, _poll_progress)

    # ── LLM解析アドバイスの非同期生成 ──
    async def _generate_advice_async():
        """LLMで解析アドバイスを生成し、UIを更新する。"""
        if not _HAS_ADVISOR:
            return
        try:
            advisor = AnalysisAdvisor()
            if not advisor.is_available:
                advice_status.text = "⚠️ LLM未準備"
                return

            advice_status.text = "⏳ アドバイス生成中..."

            # LLM呼び出し（io_boundで非同期実行）
            advice = await run.io_bound(advisor.generate_advice, state)

            # UI更新
            advice_content.text = advice
            advice_content.visible = True
            advice_status.text = "✅ 生成完了"
            ui.notify("🤖 AI解析アドバイスが準備できました", type="positive", timeout=3000)

        except Exception as e:
            logger.warning(f"[AnalysisAdvisor] アドバイス生成エラー: {e}")
            advice_status.text = "⚠️ 生成失敗"
            advice_content.text = f"アドバイスの生成に失敗しました:\n{str(e)[:200]}"
            advice_content.visible = True

    # アドバイス生成を開始（非同期、解析の邪魔をしない）
    if _HAS_ADVISOR:
        asyncio.create_task(_generate_advice_async())

    try:
        # タスク判定
        task = state.get("task_type", "auto")

        # モデル選択
        model_keys = state.get("selected_models")
        if not model_keys:
            from backend.models.factory import get_default_automl_models
            effective_task = task
            if effective_task == "auto":
                effective_task = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
            model_keys = get_default_automl_models(task=effective_task)

        # SMILES列
        smiles_col = state.get("smiles_col") or None

        # 除外列の処理
        exclude_cols = state.get("exclude_cols", [])
        df_work = df.copy()
        if exclude_cols:
            df_work = df_work.drop(columns=[c for c in exclude_cols if c in df_work.columns], errors="ignore")

        # ── パイプライン設定をstateから抽出 ──
        preprocess_params = {}
        for key in [
            "num_scaler", "num_imputer", "num_transform",
            "cat_encoder", "cat_imputer",
            "feature_selector", "n_features_to_select",
            "do_polynomial", "poly_degree", "poly_interaction_only",
        ]:
            if key in state:
                preprocess_params[key] = state[key]

        # model_params: 直接指定 + EstimatorConfigDialog で設定された値を統合
        model_params = dict(state.get("model_params") or {})
        model_configs = state.get("model_configs", {})
        for mkey, mcfg in model_configs.items():
            if hasattr(mcfg, "default_params") and mcfg.default_params:
                if mkey not in model_params:
                    model_params[mkey] = {}
                model_params[mkey].update(mcfg.default_params)
        model_params = model_params or None

        # ── 新しい単調性制約システムの反映 ──
        monotonic_constraints = state.get("feature_constraints", {})
        
        try:
            from frontend_nicegui.components.column_meta_editor import build_column_meta_dict
            column_meta_dict = build_column_meta_dict(state)
        except Exception as _e:
            logger.warning(f"column_meta の変換に失敗: {_e}")
            import traceback
            logger.debug(traceback.format_exc())
            column_meta_dict = None
        cv_key = state.get("cv_key", "auto")

        # ══════════════════════════════════════════════════════
        # 複数セット対応: active=True のセットをループ
        # ══════════════════════════════════════════════════════
        desc_sets = state.get("descriptor_sets", {})
        active_sets = {
            name: info for name, info in desc_sets.items()
            if info.get("active", True)
        }

        # activeセットがなければ現在の選択で1回実行
        if not active_sets:
            active_sets = {"デフォルト": {"descriptors": state.get("selected_descriptors")}}

        all_results: dict[str, Any] = {}
        best_result = None
        best_set_name = ""
        best_score = -float("inf")
        total_sets = len(active_sets)

        for set_idx, (set_name, set_info) in enumerate(active_sets.items()):
            # ── キャンセルチェック ──
            if _cancel_requested:
                progress_label.text = "🛑 ユーザーにより解析がキャンセルされました"
                progress_detail.text = f"{set_idx}/{total_sets}セット完了時点で中断"
                ui.notify("🛑 解析をキャンセルしました", type="warning")
                break

            set_descs = set_info.get("descriptors")
            # 記述子リストのバリデーション:
            # set_descsはprecalc_dfの列名だが、SmilesDescriptorTransformerが
            # 計算時に存在しない記述子をフィルタしてしまう可能性がある。
            # 空リストやNoneの場合は全記述子を使用（フォールバック）。
            if set_descs and isinstance(set_descs, (list, tuple)) and len(set_descs) > 0:
                selected_desc = list(set_descs)
            else:
                selected_desc = state.get("selected_descriptors")

            # 進捗更新
            progress_label.text = f"⏳ [{set_idx + 1}/{total_sets}] セット「{set_name}」を解析中..."
            progress_bar.value = set_idx / total_sets
            progress_pct.text = f"{int(set_idx / total_sets * 100)}%"

            set_queue: queue.Queue = queue.Queue(maxsize=100)

            # セット固有の進捗ポーリング
            def _poll_set_progress(sq=set_queue, sn=set_name, si=set_idx):
                latest = None
                while True:
                    try:
                        item = sq.get_nowait()
                        if item[0] == "progress":
                            latest = item
                    except queue.Empty:
                        break
                if latest:
                    _, step, total, msg = latest
                    set_pct = (si + step / max(total, 1)) / total_sets
                    progress_bar.value = set_pct
                    progress_pct.text = f"{int(set_pct * 100)}%"
                    progress_label.text = f"⏳ [{si + 1}/{total_sets}] {sn}: {msg}"
                    progress_detail.text = f"セット {si + 1}/{total_sets} | ステップ {step}/{total}"
                    elapsed = time.time() - _start_time
                    if set_pct > 0.05:
                        eta_sec = elapsed / set_pct * (1 - set_pct)
                        progress_eta.text = f"残り約{eta_sec:.0f}秒" if eta_sec < 60 else f"残り約{eta_sec/60:.1f}分"

            set_timer = ui.timer(0.5, _poll_set_progress)

            try:
                # アクティブエンジンの抽出
                _engine_map = {
                    "use_rdkit": "RDKitAdapter", "use_xtb": "XTBAdapter", "use_mordred": "MordredAdapter",
                    "use_skfp": "SkfpAdapter", "use_mol2vec": "Mol2VecAdapter", "use_groupcontrib": "GroupContribAdapter",
                    "use_molai": "MolAIAdapter", "use_uma": "UMAAdapter", "use_padel": "PaDELAdapter",
                    "use_descriptastorus": "DescriptaStorusAdapter", "use_molfeat": "MolfeatAdapter",
                    "use_chemprop": "ChempropAdapter", "use_cosmo": "CosmoAdapter", "use_unipka": "UniPkaAdapter"
                }
                active_engines = [cls_name for k, cls_name in _engine_map.items() if state.get(k)]

                result = await run.io_bound(
                    _run_engine_sync,
                    df_work,
                    target_col,
                    smiles_col if smiles_col and smiles_col in df_work.columns else None,
                    state.get("group_col"),
                    task,
                    model_keys if model_keys else None,
                    state.get("cv_folds", 5),
                    state.get("timeout", 300),
                    selected_desc,
                    set_queue,
                    active_engines=active_engines,
                    cv_key=cv_key,
                    model_params=model_params,
                    preprocess_params=preprocess_params if preprocess_params else None,
                    monotonic_constraints=monotonic_constraints,
                    column_meta_dict=column_meta_dict,
                    count_normalization=state.get("count_normalization", "density"),
                    # Morgan フィンガープリント オプション
                    morgan_count=state.get("morgan_count", False),
                    morgan_radius=state.get("morgan_radius", 2),
                    morgan_bits=state.get("morgan_bits", 2048),
                    morgan_order_by_appearance=state.get("morgan_order_by_appearance", False),
                )
                all_results[set_name] = result


                # ベストスコア追跡
                if hasattr(result, "best_score") and result.best_score > best_score:
                    best_score = result.best_score
                    best_result = result
                    best_set_name = set_name

            except AnalysisCancelled:
                logger.info(f"セット「{set_name}」がキャンセルされました")
                all_results[set_name] = None
                set_timer.deactivate()
                break
            except Exception as set_ex:
                import traceback as _tb
                _tb_text = _tb.format_exc()
                logger.error(f"セット「{set_name}」の解析エラー: {set_ex}\n{_tb_text}")
                all_results[set_name] = None
                # UIにもエラー詳細を表示
                try:
                    _short = str(set_ex)[:200]
                    _full = _tb_text[-1500:] if len(_tb_text) > 1500 else _tb_text
                    progress_label.text = f"⚠️ セット「{set_name}」でエラー: {_short}"
                    # 詳細エラーを展開可能に表示
                    with status_container:
                        with ui.expansion("🔍 エラー詳細（クリックで展開）", icon="bug_report").classes("full-width q-mt-xs"):
                            ui.code(_full).classes("full-width").style("font-size: 0.7rem; max-height: 300px; overflow-y: auto;")
                except Exception:
                    pass
            finally:
                set_timer.deactivate()

        # ── 結果の保存 ──
        state["automl_results"] = all_results  # 全セット結果
        # 後方互換: 最良セットを automl_result にも保存
        if best_result:
            state["automl_result"] = best_result
            state["best_set_name"] = best_set_name
            state["pipeline_result"] = type("PipelineResult", (), {"elapsed": best_result.elapsed_seconds})()

            # ── 評価エンジン（StratifiedMetricCalculator）の初期化と一括計算 ──
            try:
                from backend.metrics.stratified_evaluator import StratifiedMetricCalculator
                
                metric_calc = StratifiedMetricCalculator(min_group_size=10)
                
                # y_true, y_pred を取得
                y_train = getattr(best_result, 'y_train', None)
                if y_train is None:
                    logger.warning("best_result に y_train がありません。層化評価をスキップします。")
                    return
                y_train_pred = getattr(best_result, 'train_predictions', None)
                if y_train_pred is None:
                    pipeline = getattr(best_result, "best_pipeline", None)
                    X_train = getattr(best_result, 'X_train', None)
                    if pipeline is not None and X_train is not None:
                        y_train_pred = pipeline.predict(X_train)
                    else:
                        logger.warning("学習予測値またはパイプライン/X_trainが取得できません。層化評価をスキップします。")
                        return
                
                if y_train_pred is not None:
                    metadata = state.get("df_encoded") if state else None
                    
                    stratified_metrics = metric_calc.compute_all(
                        y_true=y_train,
                        y_pred=y_train_pred,
                        metadata=metadata,
                        auto_cluster=True,
                        n_clusters=5,
                        cluster_method="kmeans"
                    )
                    
                    best_result.stratified_metrics = stratified_metrics
                    state["stratified_metrics"] = stratified_metrics
                
            except Exception as ev_ex:
                logger.warning(f"メトリック評価エンジンの初期化・計算に失敗: {ev_ex}")

            # ── 自動解決された単調性制約を column_meta に反映し、UIへフィードバック ──
            res_constraints = getattr(best_result, "resolved_constraints", {})
            cm = state.setdefault("column_meta", {})
            for col, val in res_constraints.items():
                if col not in cm:
                    cm[col] = {"monotonic": 0}
                cm[col]["resolved_monotonic"] = val

            # 新しい単調性制約フィードバック
            feature_constraints = state.get("feature_constraints", {})
            if feature_constraints:
                active = [f for f, c in feature_constraints.items() if c.get("direction") not in ("none", None)]
                if active:
                    model_key = best_result.best_model_key.lower().replace(" ", "")
                    if any(s in model_key for s in ["lgbm", "xgb", "catboost", "hist"]):
                        ui.notify(f"✅ 単調性制約適用: {best_result.best_model_key} ({len(active)}変数: {', '.join(active[:3])}{'...' if len(active)>3 else ''})", type="positive", timeout=4000)
                    else:
                        ui.notify(f"⚠️ 単調性制約スキップ: {best_result.best_model_key} は非対応です（設定は保持されます）", type="warning", timeout=5000)

            # ── バックグラウンドでプロット（Plotly + matplotlib）を自動保存 ──
            try:
                from backend.utils.plot_export_hooks import export_all_plots_from_result
                import time as _time
                _session_id = f"session_{int(_time.time())}"
                state["_plot_session_id"] = _session_id
                export_all_plots_from_result(
                    best_result, state, session_id=_session_id, run_async=True
                )
                logger.info("[PlotExport] バックグラウンドでプロット保存開始: %s", _session_id)
            except Exception as _pe:
                logger.warning("[PlotExport] プロット保存フック失敗: %s", _pe)

        # 成功表示
        elapsed_total = time.time() - _start_time
        progress_bar.value = 1.0
        progress_pct.text = "100%"

        n_success = sum(1 for v in all_results.values() if v is not None)
        if best_result:
            progress_label.text = (
                f"✅ 解析完了！ {n_success}/{total_sets}セット成功 | "
                f"最良: {best_set_name} → {best_result.best_model_key}"
            )
            n_models = len(best_result.model_scores)
            proc_X = getattr(best_result, "processed_X", None)
            n_feats = proc_X.shape[1] if proc_X is not None and hasattr(proc_X, "shape") else "?"
            progress_detail.text = (
                f"スコア: {best_result.best_score:.4f} | "
                f"所要時間: {elapsed_total:.1f}秒 | "
                f"{n_models}モデル比較 | {n_feats}特徴量"
            )
            progress_eta.text = f"タスク: {best_result.task}"
            ui.notify(
                f"✅ {n_success}セット解析完了！ 最良: {best_set_name} ({best_result.best_score:.4f})",
                type="positive",
                timeout=5000,
            )
        else:
            progress_label.text = "❌ 全セットの解析に失敗しました"
            progress_detail.text = "設定を確認して再実行してください"

        # ── 失敗セットの詳細表示 ──
        failed_sets = {k: v for k, v in all_results.items() if v is None}
        if failed_sets and best_result:
            with status_container:
                with ui.card().classes("full-width q-pa-sm q-mt-sm").style(
                    "border: 1px solid rgba(251,191,36,0.3); border-radius: 8px;"
                    "background: rgba(60,40,0,0.15);"
                ):
                    ui.label(f"⚠️ {len(failed_sets)}セットで解析失敗（スキップ済み）").classes(
                        "text-subtitle2 text-bold text-amber"
                    )
                    with ui.expansion("失敗セットの詳細を確認", icon="warning").classes("full-width q-mt-xs"):
                        for sname in failed_sets:
                            ui.label(f"❌ {sname}").classes("text-caption text-red q-ml-sm")
                        ui.label(
                            "💡 失敗したセットはスキップされ、成功したセットの結果が表示されています。"
                            "記述子の設定やデータを確認してください。"
                        ).classes("text-caption text-grey q-mt-xs")

        # ── 結果タブへ自動切り替え＋再描画 ──
        # 重要: on_complete（タブ遷移）の前に _switch_to_results（再描画含む）を呼ぶ。
        # これにより「空の結果タブ」バグを防止する。
        switch_results = state.get("_switch_to_results")
        if switch_results and best_result:
            try:
                switch_results()
            except Exception as _re:
                logger.warning("結果タブ切替に失敗: %s", _re)
                refresh_only = state.get("_refresh_results")
                if refresh_only:
                    try:
                        refresh_only()
                    except Exception:
                        pass

        if on_complete:
            on_complete()

        # 解析履歴を自動記録
        try:
            from backend.preset_manager import record_analysis
            if best_result:
                record_analysis(state, best_result)
        except Exception as hist_ex:
            logger.warning("解析履歴の保存に失敗: %s", hist_ex)

        # ── SQLite バージョン管理への自動保存 ──
        try:
            from backend.session.version_manager import VersionManager
            if best_result:
                vm = VersionManager()
                saved_hash = vm.save_from_automl_result(best_result, state)
                logger.info("実験を保存しました: hash=%s", saved_hash[:8])
                # 実験比較ダッシュボードの再描画
                refresh_comp = state.get("_refresh_experiment_comparison")
                if refresh_comp:
                    try:
                        refresh_comp()
                    except Exception:
                        pass
        except Exception as vm_ex:
            logger.warning("VersionManager保存に失敗: %s", vm_ex)

        # ── タスク3-1: 事前設定済みの自動処理を実行 ──
        if best_result:
            await _run_post_analysis_tasks(state, status_container)


    except AnalysisCancelled:
        progress_label.text = "🛑 解析がキャンセルされました"
        progress_pct.text = "—"
        progress_detail.text = "ユーザーの要求により中断しました"
        progress_eta.text = ""
        ui.notify("🛑 解析をキャンセルしました", type="warning")

    except Exception as ex:
        error_msg = str(ex)
        tb_text = traceback.format_exc()
        short_msg = error_msg[:200] + "..." if len(error_msg) > 200 else error_msg

        # エラー種別に応じた対処法
        remedy = _get_error_remedy(error_msg, tb_text)

        progress_pct.text = "❌"
        progress_label.text = "❌ エラーが発生しました"
        progress_detail.text = short_msg
        progress_eta.text = ""

        # 対処法パネル
        with status_container:
            with ui.card().classes("full-width q-pa-sm q-mt-sm").style(
                "border: 1px solid rgba(248,113,113,0.3); border-radius: 8px; background: rgba(60,20,20,0.2);"
            ):
                ui.label("💡 対処法").classes("text-subtitle2 text-bold text-amber")
                ui.label(remedy).classes("text-caption")
                with ui.expansion("🔍 詳細エラー情報", icon="bug_report").classes("full-width q-mt-xs"):
                    ui.code(tb_text[-1000:]).classes("full-width").style("font-size: 0.7rem;")

        ui.notify(f"解析エラー: {short_msg}", type="negative", timeout=8000)
        logger.error(f"AutoML実行エラー: {tb_text}")

    finally:
        _analysis_running = False
        timer.deactivate()
        try:
            resource_timer.deactivate()
        except Exception:
            pass


def _get_error_remedy(error_msg: str, tb_text: str) -> str:
    """エラーメッセージからユーザー向けの対処法を生成する。"""
    msg_lower = error_msg.lower()
    tb_lower = tb_text.lower()

    if "memory" in msg_lower or "memoryerror" in tb_lower:
        return (
            "メモリ不足です。以下を試してください:\n"
            "• データの行数を減らす（サンプリング）\n"
            "• 記述子エンジンを少なくする\n"
            "• 使わない列を「除外」に設定する"
        )
    elif "smiles" in msg_lower or "rdkit" in tb_lower or "mol" in msg_lower:
        return (
            "SMILES列の解析に失敗しました:\n"
            "• SMILES列に無効な分子構造が含まれている可能性があります\n"
            "• SMILES列を「なし」に設定して再解析するか、データを確認してください"
        )
    elif "target" in msg_lower and ("not found" in msg_lower or "見つかりません" in msg_lower):
        return (
            "目的変数が見つかりません:\n"
            "• 「列の役割」タブで目的変数が正しく設定されているか確認してください"
        )
    elif "timeout" in msg_lower or "timed out" in msg_lower:
        return (
            "解析がタイムアウトしました:\n"
            "• タイムアウト値を増やしてください（パイプライン設定）\n"
            "• モデル数を減らすと高速化できます"
        )
    elif "fit" in msg_lower or "convergence" in tb_lower:
        return (
            "モデルの学習に失敗しました:\n"
            "• データに欠損値やInfが多い可能性があります — EDAタブで確認\n"
            "• スケーラーを「robust」に変更してみてください"
        )
    elif "nan" in msg_lower or "inf" in msg_lower or "missing" in msg_lower:
        return (
            "データに NaN/Inf が含まれています:\n"
            "• EDAタブの「欠損行削除」や「高欠損列削除」を試してください\n"
            "• 前処理の欠損値補完方法を変更してみてください"
        )
    elif "shape" in msg_lower or "dimension" in msg_lower:
        return (
            "データの次元に問題があります:\n"
            "• 全行が同じ値の定数列があれば「除外」してください\n"
            "• 特徴量数がサンプル数より大きい場合、特徴量選択を有効にしてください"
        )
    else:
        return (
            "予期しないエラーが発生しました:\n"
            "• データの形式やサイズを確認してください\n"
            "• 詳細エラー情報（下）を展開して原因を確認してください\n"
            "• 問題が解決しない場合は、設定を変更して再試行してください"
        )


async def _run_post_analysis_tasks(state: dict, status_container) -> None:
    """
    順解析完了後に事前設定された自動処理を実行する。

    Implements: F-3-1 | 順解析完了時の自動逆解析実行
    事前設定 (state["_post_analysis"]) に基づき:
      - auto_inverse=True → 逆解析を自動実行
      - auto_report=True → レポートを自動生成（将来）
      - auto_shap=True → SHAP解析を自動実行（将来）
    """
    pa = state.get("_post_analysis")
    if not pa:
        return

    ar = state.get("automl_result")
    if ar is None:
        return

    tasks_run = []

    # ═══════════════════════════════════════════
    # 自動逆解析
    # ═══════════════════════════════════════════
    if pa.get("auto_inverse"):
        try:
            logger.info("[PostAnalysis] 自動逆解析を開始...")

            with status_container:
                with ui.card().classes("full-width glass-card q-pa-sm q-mb-xs q-mt-sm").style(
                    "border-left: 3px solid rgba(123, 47, 247, 0.7);"
                ):
                    pa_label = ui.label("🔮 逆解析を自動実行中...").classes("text-body2 text-purple")
                    pa_progress = ui.linear_progress(value=0, show_value=False).props(
                        "color=purple rounded"
                    ).style("height: 4px;")

            # 逆解析設定の構築
            df = state.get("df")
            target_col = state.get("target_col", "")
            precalc_df = state.get("precalc_df")

            if df is None or not target_col:
                pa_label.text = "⚠️ 逆解析スキップ: データまたは目的変数が未設定"
                return

            # 逆解析の設定を _inv に反映
            if "_inv" not in state:
                state["_inv"] = {
                    "target_mode": "range",
                    "constraints": {},
                    "method": "random",
                    "method_params": {},
                    "results": None,
                }
            inv = state["_inv"]

            # 目標モードの設定
            inv["target_mode"] = pa.get("inv_target_mode", "maximize")
            inv["target_min"] = pa.get("inv_target_min")
            inv["target_max"] = pa.get("inv_target_max")

            # 手法の設定
            inv["method"] = pa.get("inv_method", "random")
            inv["method_params"] = dict(pa.get("inv_method_params", {}))

            # 使用モデル
            inv["selected_model"] = ar.best_model_key if hasattr(ar, "best_model_key") else None

            # 制約の自動設定
            if pa.get("inv_auto_constraints", True):
                import pandas as _pd
                expand = pa.get("inv_constraint_expand", 0.2)
                exclude = set(state.get("exclude_cols", []))
                smiles_col = state.get("smiles_col", "")
                source_df = precalc_df if precalc_df is not None else df

                for col in source_df.columns:
                    if col == target_col or col == smiles_col or col in exclude:
                        continue
                    if not _pd.api.types.is_numeric_dtype(source_df[col]):
                        continue
                    col_data = source_df[col].dropna()
                    if len(col_data) == 0:
                        continue
                    col_min = float(col_data.min())
                    col_max = float(col_data.max())
                    span = (col_max - col_min) * expand
                    inv["constraints"][col] = {
                        "min": col_min - span,
                        "max": col_max + span,
                        "fixed": False,
                        "fixed_val": float(col_data.median()),
                        "active": True,
                    }

            pa_progress.value = 0.3
            pa_label.text = "🔮 逆解析の最適化を実行中..."

            # 逆解析の実行
            try:
                from backend.inverse.optimizer import InverseOptimizer

                # パイプラインとメタ情報の取得
                pipeline = getattr(ar, "best_pipeline", None)
                if pipeline is None:
                    pa_label.text = "⚠️ 逆解析スキップ: パイプラインが取得できません"
                    return

                # 特徴量名の取得
                proc_X = getattr(ar, "processed_X", None)
                if proc_X is not None and hasattr(proc_X, "columns"):
                    feature_names = list(proc_X.columns)
                else:
                    feature_names = list(inv["constraints"].keys())

                # 制約の変換
                bounds = {}
                fixed_values = {}
                for col, c in inv["constraints"].items():
                    if col not in feature_names:
                        continue
                    if c.get("fixed"):
                        fixed_values[col] = c.get("fixed_val", 0)
                    else:
                        bounds[col] = (c.get("min", -1e6), c.get("max", 1e6))

                method = inv.get("method", "random")
                method_params = inv.get("method_params", {})

                optimizer = InverseOptimizer(
                    pipeline=pipeline,
                    feature_names=feature_names,
                    bounds=bounds,
                    fixed_values=fixed_values,
                    target_mode=inv.get("target_mode", "maximize"),
                    target_range=(inv.get("target_min"), inv.get("target_max")),
                )

                pa_progress.value = 0.5

                result = await run.io_bound(
                    optimizer.optimize,
                    method=method,
                    **method_params,
                )

                inv["results"] = result
                pa_progress.value = 1.0

                n_candidates = len(result) if hasattr(result, "__len__") else 0
                pa_label.text = f"✅ 逆解析完了！ {n_candidates}候補を生成"
                tasks_run.append("逆解析")

                # 逆解析タブを再描画
                refresh_inv = state.get("_refresh_inverse")
                if refresh_inv:
                    try:
                        refresh_inv()
                    except Exception:
                        pass

                logger.info("[PostAnalysis] 自動逆解析完了: %d候補", n_candidates)

            except ImportError:
                pa_label.text = "⚠️ 逆解析モジュールが見つかりません"
                logger.warning("[PostAnalysis] InverseOptimizer import failed")
            except Exception as inv_ex:
                pa_label.text = f"⚠️ 逆解析エラー: {str(inv_ex)[:100]}"
                logger.warning("[PostAnalysis] 逆解析エラー: %s", inv_ex)

        except Exception as e:
            logger.warning("[PostAnalysis] 自動逆解析の全体エラー: %s", e)

    # ═══════════════════════════════════════════
    # サマリー通知
    # ═══════════════════════════════════════════
    if tasks_run:
        ui.notify(
            f"🤖 自動処理完了: {', '.join(tasks_run)}",
            type="positive", timeout=5000,
        )

