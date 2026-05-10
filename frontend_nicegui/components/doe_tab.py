"""
frontend_nicegui/components/doe_tab.py

実験計画法（DoE）タブ。
D最適 / E最適 / I最適 / Maximin / Minimax / 直交表に対応。
① ゼロから設計  ② 既存データを活用して追加実験を計画
"""
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────

_CRITERIA = {
    "D": "D最適（情報量最大化 / det(X'X) 最大）",
    "E": "E最適（最小固有値最大化）",
    "I": "I最適（平均予測分散最小化）",
    "MAXIMIN": "Maximin（空間充填: 最小点間距離最大化）",
    "MINIMAX": "Minimax（空間充填: 最大カバー率最適化）",
    "OA": "直交表（Orthogonal Array）",
}

_CARD_STYLE = "border:1px solid rgba(99,102,241,0.25); border-radius:10px;"
_SUCCESS_STYLE = "border:1px solid rgba(74,222,128,0.5); border-radius:8px;"
_WARN_STYLE = "border:1px solid rgba(251,191,36,0.5); border-radius:8px;"


# ─────────────────────────────────────────────────────────────────────────────
# メインエントリ
# ─────────────────────────────────────────────────────────────────────────────

def render_doe_tab(app_state: dict | None = None) -> None:
    """DoEタブ全体をレンダリングする。"""
    state: dict[str, Any] = {
        "mode": "scratch",       # "scratch" | "existing"
        "factors": [],           # list[dict] — 因子定義
        "existing_df": None,     # pd.DataFrame | None
        "n_new": 8,
        "criterion": "D",
        "oa_name": "L8(2\u2077)",  # 直交表名
        "max_candidates": 5000,
        "random_seed": 42,
        "n_starts": 5,
        "max_iter": 200,
        "result": None,          # DoEResult | None
        "_app_state": app_state,  # app_stateへの参照を保持
    }

    # ── ヘッダー ─────────────────────────────────────────────────────────────
    with ui.row().classes("items-center q-gutter-md q-mb-sm"):
        ui.icon("science", color="indigo").classes("text-h4")
        with ui.column().classes("q-gutter-none"):
            ui.label("実験計画法（DoE）").classes("text-h5 text-bold")
            ui.label(
                "D最適 / E最適 / I最適 / Maximin / Minimax / 直交表で効率的な実験計画を生成します"
            ).classes("text-grey-5 text-caption")

    # ── モード選択 ──────────────────────────────────────────────────────────
    ui.separator()
    with ui.row().classes("items-center q-gutter-sm q-mb-md"):
        ui.label("モード:").classes("text-subtitle2")
        mode_toggle = ui.toggle(
            {"scratch": "① ゼロから設計", "existing": "② 既存データを活用"},
            value="scratch",
            on_change=lambda e: _on_mode_change(e.value, state, mode_panels),
        ).props("dense")

    # ── モード別パネル ───────────────────────────────────────────────────────
    mode_panels: dict[str, ui.column] = {}

    with ui.column().classes("full-width") as _container:
        # ① ゼロから設計
        with ui.column().classes("full-width") as panel_scratch:
            _render_scratch_mode(state)
        mode_panels["scratch"] = panel_scratch

        # ② 既存データを活用
        with ui.column().classes("full-width") as panel_existing:
            _render_existing_mode(state)
        mode_panels["existing"] = panel_existing

    # 初期表示: scratchのみ表示
    panel_existing.set_visibility(False)

    # ── 共通設定 + 実行ボタン ─────────────────────────────────────────────────
    ui.separator().classes("q-my-md")
    _render_common_settings(state)

    # ── 結果エリア ───────────────────────────────────────────────────────────
    result_area = ui.column().classes("full-width q-mt-md")
    state["_result_area"] = result_area


def _on_mode_change(new_mode: str, state: dict, panels: dict) -> None:
    state["mode"] = new_mode
    for name, panel in panels.items():
        panel.set_visibility(name == new_mode)


# ─────────────────────────────────────────────────────────────────────────────
# ① ゼロから設計
# ─────────────────────────────────────────────────────────────────────────────

def _render_scratch_mode(state: dict) -> None:
    ui.label("因子の定義").classes("text-subtitle2 q-mt-sm")
    ui.label(
        "実験で制御する因子（説明変数）を追加してください。"
        "連続値には最小・最大・水準数を、カテゴリには値をカンマ区切りで入力します。"
    ).classes("text-grey-5 text-caption q-mb-xs")

    factor_list = ui.column().classes("full-width q-gutter-xs")

    def _add_factor():
        idx = len(state["factors"])
        fdef = {
            "name": f"Factor_{idx+1}",
            "type": "continuous",
            "low": 0.0, "high": 1.0, "n_levels": 5,
            "categories": "A,B,C",
        }
        state["factors"].append(fdef)
        _render_factor_row(factor_list, fdef, state, on_remove=lambda: _refresh_factors(factor_list, state))

    with ui.row().classes("q-gutter-sm q-mb-xs"):
        ui.button("＋ 因子を追加", on_click=_add_factor).props("unelevated dense no-caps color=indigo size=sm")
        ui.button("💡 サンプル3因子を挿入", on_click=lambda: _insert_sample_factors(factor_list, state)).props(
            "flat dense no-caps color=grey size=sm"
        )

    # デフォルト3因子を挿入
    _insert_sample_factors(factor_list, state)


def _insert_sample_factors(factor_list: ui.column, state: dict) -> None:
    state["factors"].clear()
    samples = [
        {"name": "温度 (°C)", "type": "continuous", "low": 60.0, "high": 120.0, "n_levels": 5,  "categories": ""},
        {"name": "圧力 (MPa)", "type": "continuous", "low": 1.0,  "high": 10.0, "n_levels": 5,  "categories": ""},
        {"name": "触媒種",    "type": "categorical", "low": 0.0,  "high": 1.0,  "n_levels": 3,  "categories": "A,B,C"},
    ]
    state["factors"].extend(samples)
    factor_list.clear()
    for fdef in state["factors"]:
        _render_factor_row(factor_list, fdef, state, on_remove=lambda: _refresh_factors(factor_list, state))


def _refresh_factors(factor_list: ui.column, state: dict) -> None:
    factor_list.clear()
    for fdef in state["factors"]:
        _render_factor_row(factor_list, fdef, state, on_remove=lambda: _refresh_factors(factor_list, state))


def _render_factor_row(parent: ui.column, fdef: dict, state: dict, on_remove) -> None:
    with parent:
        with ui.card().classes("full-width q-pa-xs").style(_CARD_STYLE):
            with ui.row().classes("items-center q-gutter-sm full-width"):
                # 因子名
                ui.input(
                    "因子名",
                    value=fdef["name"],
                    on_change=lambda e, d=fdef: d.update({"name": e.value}),
                ).props("dense outlined").style("width:160px;")

                # 型
                ui.select(
                    {"continuous": "連続値", "categorical": "カテゴリ"},
                    value=fdef["type"],
                    label="型",
                    on_change=lambda e, d=fdef: d.update({"type": e.value}),
                ).props("dense outlined").style("width:100px;")

                # 連続値設定
                with ui.row().classes("items-center q-gutter-xs") as cont_row:
                    ui.number("最小値", value=fdef["low"], format="%.4g",
                              on_change=lambda e, d=fdef: d.update({"low": float(e.value or 0)}),
                              ).props("dense outlined").style("width:90px;")
                    ui.number("最大値", value=fdef["high"], format="%.4g",
                              on_change=lambda e, d=fdef: d.update({"high": float(e.value or 1)}),
                              ).props("dense outlined").style("width:90px;")
                    ui.number("水準数", value=fdef["n_levels"], min=2, max=20, step=1,
                              on_change=lambda e, d=fdef: d.update({"n_levels": int(e.value or 5)}),
                              ).props("dense outlined integer").style("width:75px;").tooltip("候補として使う等間隔水準数")

                # カテゴリ設定
                with ui.row().classes("items-center q-gutter-xs") as cat_row:
                    ui.input(
                        "水準（カンマ区切り）",
                        value=fdef["categories"],
                        on_change=lambda e, d=fdef: d.update({"categories": e.value}),
                        placeholder="例: A,B,C または 低,中,高",
                    ).props("dense outlined").style("width:260px;")

                # 削除ボタン
                ui.button(
                    icon="delete",
                    on_click=lambda _f=fdef: (_factor_remove(state, _f), on_remove()),
                ).props("flat round dense color=red size=sm")

            # 型に応じて表示切替はJSより状態で管理（NiceGUIの制約）
            # 簡易: カテゴリ行を非表示にする
            is_cat = fdef["type"] == "categorical"
            cont_row.set_visibility(not is_cat)
            cat_row.set_visibility(is_cat)


def _factor_remove(state: dict, fdef: dict) -> None:
    if fdef in state["factors"]:
        state["factors"].remove(fdef)


# ─────────────────────────────────────────────────────────────────────────────
# ② 既存データを活用
# ─────────────────────────────────────────────────────────────────────────────

def _render_existing_mode(state: dict) -> None:
    ui.label("既存実験データの読み込み").classes("text-subtitle2 q-mt-sm")
    ui.label(
        "既存の実験結果CSVを読み込み、各列を因子として使います。"
        "読み込んだデータは固定し、その上に追加実験点を最適化します。"
    ).classes("text-grey-5 text-caption q-mb-xs")

    file_info = ui.label("").classes("text-caption text-grey-5")
    factor_table_area = ui.column().classes("full-width")

    def _on_upload(e):
        try:
            content = e.content.read()
            df = pd.read_csv(io.BytesIO(content))
            state["existing_df"] = df
            file_info.set_text(f"✅ {e.name}  ({len(df)}行 × {len(df.columns)}列)")
            _show_existing_factor_editor(factor_table_area, df, state)
        except Exception as ex:
            ui.notify(f"読み込みエラー: {ex}", type="negative")

    ui.upload(
        label="CSVをドラッグ＆ドロップ または クリックして選択",
        on_upload=_on_upload,
        auto_upload=True,
    ).props("accept=.csv flat outlined").classes("full-width q-mb-xs")

    file_info
    factor_table_area

    # 現在のapp_stateのデータを使うボタン（app_stateが渡されていれば）
    with ui.row().classes("items-center q-gutter-sm q-mt-xs"):
        ui.label("または:").classes("text-caption text-grey-5")
        ui.button(
            "📂 現在の解析設定のデータを使用",
            on_click=lambda: _use_app_state_data(state, factor_table_area, file_info),
        ).props("flat dense no-caps color=teal size=sm")


def _use_app_state_data(state: dict, area: ui.column, file_info: ui.label) -> None:
    """app_stateのDataFrameをそのまま使う（render_doe_tabで渡されたapp_stateから取得）。"""
    app_st = state.get("_app_state")
    if app_st is None:
        ui.notify("データ連携未設定です。", type="warning")
        return
    df = app_st.get("df")
    if df is None or len(df) == 0:
        ui.notify("データが読み込まれていません。先に「解析設定」タブでデータを読み込んでください。", type="warning")
        return
    state["existing_df"] = df.copy()
    file_info.set_text(f"✅ 現在のデータ ({len(df)}行 × {len(df.columns)}列)")
    _show_existing_factor_editor(area, df, state)


def _show_existing_factor_editor(area: ui.column, df: pd.DataFrame, state: dict) -> None:
    """読み込んだデータから因子設定UIを生成する。"""
    area.clear()

    # 因子定義をDataFrameから自動推定
    state["factors"].clear()
    for col in df.columns:
        col_data = df[col].dropna()
        if pd.api.types.is_numeric_dtype(col_data) and col_data.nunique() > 5:
            fdef = {
                "name": col,
                "type": "continuous",
                "low": float(col_data.min()),
                "high": float(col_data.max()),
                "n_levels": 5,
                "categories": "",
                "_role": "factor",  # "factor" | "target" | "ignore"
            }
        else:
            fdef = {
                "name": col,
                "type": "categorical",
                "low": 0.0, "high": 1.0, "n_levels": 3,
                "categories": ",".join(str(v) for v in sorted(col_data.unique())),
                "_role": "factor",
            }
        state["factors"].append(fdef)

    with area:
        ui.label("📋 列の役割設定と因子範囲").classes("text-subtitle2 q-mb-xs")
        ui.label(
            "各列の「役割」を選択してください。"
            "「因子」を選んだ列が実験計画の変数になります。目的変数列は「目的変数」にしてください。"
        ).classes("text-grey-5 text-caption q-mb-sm")

        for fdef in state["factors"]:
            with ui.card().classes("full-width q-pa-xs q-mb-xs").style(_CARD_STYLE):
                with ui.row().classes("items-center q-gutter-sm full-width"):
                    ui.label(fdef["name"]).classes("text-body2 text-bold").style("min-width:120px;")

                    role_sel = ui.select(
                        {"factor": "因子", "target": "目的変数", "ignore": "無視"},
                        value=fdef.get("_role", "factor"),
                        label="役割",
                        on_change=lambda e, d=fdef: d.update({"_role": e.value}),
                    ).props("dense outlined").style("width:100px;")

                    type_sel = ui.select(
                        {"continuous": "連続値", "categorical": "カテゴリ"},
                        value=fdef["type"],
                        label="型",
                        on_change=lambda e, d=fdef: d.update({"type": e.value}),
                    ).props("dense outlined").style("width:100px;")

                    if fdef["type"] == "continuous":
                        ui.number("最小値", value=fdef["low"], format="%.4g",
                                  on_change=lambda e, d=fdef: d.update({"low": float(e.value or d["low"])}),
                                  ).props("dense outlined").style("width:90px;")
                        ui.number("最大値", value=fdef["high"], format="%.4g",
                                  on_change=lambda e, d=fdef: d.update({"high": float(e.value or d["high"])}),
                                  ).props("dense outlined").style("width:90px;")
                        ui.number("水準数", value=fdef["n_levels"], min=2, max=20, step=1,
                                  on_change=lambda e, d=fdef: d.update({"n_levels": int(e.value or 5)}),
                                  ).props("dense outlined integer").style("width:70px;")
                    else:
                        ui.input(
                            "水準（カンマ区切り）",
                            value=fdef["categories"],
                            on_change=lambda e, d=fdef: d.update({"categories": e.value}),
                        ).props("dense outlined").style("width:280px;")


# ─────────────────────────────────────────────────────────────────────────────
# 共通設定 + 実行ボタン
# ─────────────────────────────────────────────────────────────────────────────

def _render_common_settings(state: dict) -> None:
    from backend.doe import list_oa_names

    with ui.card().classes("full-width q-pa-sm").style(_CARD_STYLE):
        ui.label("⚙️ 実験計画の設定").classes("text-subtitle2 q-mb-xs")

        with ui.row().classes("q-gutter-md items-start flex-wrap"):
            # 手法
            crit_options = {k: v for k, v in _CRITERIA.items()}
            crit_sel = ui.select(
                crit_options,
                value=state["criterion"],
                label="計画手法",
                on_change=lambda e: _on_criterion_change(e.value, state, oa_row),
            ).props("dense outlined").style("min-width:320px;")

            # 追加実験数
            ui.number(
                "追加実験数 (n_runs)",
                value=state["n_new"], min=2, max=500, step=1,
                on_change=lambda e: state.update({"n_new": int(e.value or 8)}),
            ).props("dense outlined integer").style("width:160px;").tooltip(
                "新規に行う実験の件数。既存データがある場合はそれに追加される。"
            )

            # 直交表選択（OA時のみ表示）
            oa_names = list_oa_names()
            with ui.row().classes("items-center q-gutter-xs") as oa_row:
                ui.label("直交表:").classes("text-caption text-grey")
                ui.select(
                    {n: n for n in oa_names},
                    value=oa_names[1] if len(oa_names) > 1 else oa_names[0],
                    label="",
                    on_change=lambda e: state.update({"oa_name": e.value}),
                ).props("dense outlined").style("min-width:160px;")
            state["oa_name"] = oa_names[1] if len(oa_names) > 1 else oa_names[0]
            oa_row.set_visibility(state["criterion"] == "OA")

        # 詳細設定（アコーディオン）
        with ui.expansion("🔬 詳細設定（候補集合・探索パラメータ）", icon="tune").classes("full-width q-mt-xs"):
            with ui.row().classes("q-gutter-md items-center flex-wrap"):
                ui.number(
                    "最大候補点数",
                    value=state["max_candidates"], min=100, max=200000, step=1000,
                    on_change=lambda e: state.update({"max_candidates": int(e.value or 5000)}),
                ).props("dense outlined integer").style("width:160px;").tooltip(
                    "因子水準の全直積がこの値を超える場合、ランダムサンプリングで候補を絞ります。\n"
                    "大きくすると精度↑・計算時間↑、小さくすると高速化。"
                )
                ui.number(
                    "乱数シード",
                    value=state["random_seed"], min=0, max=99999, step=1,
                    on_change=lambda e: state.update({"random_seed": int(e.value or 42)}),
                ).props("dense outlined integer").style("width:120px;").tooltip("再現性のためのシード値")
                ui.number(
                    "マルチスタート数",
                    value=state["n_starts"], min=1, max=50, step=1,
                    on_change=lambda e: state.update({"n_starts": int(e.value or 5)}),
                ).props("dense outlined integer").style("width:140px;").tooltip(
                    "局所最適を避けるために繰り返す初期値の数。大きいほど品質↑・計算時間↑"
                )
                ui.number(
                    "最大反復数",
                    value=state["max_iter"], min=10, max=2000, step=10,
                    on_change=lambda e: state.update({"max_iter": int(e.value or 200)}),
                ).props("dense outlined integer").style("width:130px;")

    # 実行ボタン
    with ui.row().classes("q-gutter-sm q-mt-md items-center"):
        run_btn = ui.button(
            "🚀 実験計画を生成",
            on_click=lambda: _run_doe(state, run_btn),
        ).props("unelevated no-caps color=indigo").classes("text-h6")

        ui.label("").classes("text-caption text-grey-5 q-ml-sm").bind_text_from(
            state, "_status_text", backward=lambda v: v or ""
        )
    state["_status_text"] = ""


def _on_criterion_change(new_crit: str, state: dict, oa_row: ui.row) -> None:
    state["criterion"] = new_crit
    oa_row.set_visibility(new_crit == "OA")


# ─────────────────────────────────────────────────────────────────────────────
# DoE実行
# ─────────────────────────────────────────────────────────────────────────────

async def _run_doe(state: dict, run_btn) -> None:
    from backend.doe import Factor, FactorType, DoEOptimizer, apply_orthogonal_array

    run_btn.disable()
    run_btn.text = "⏳ 計算中..."

    result_area: ui.column = state.get("_result_area")
    if result_area is None:
        # フォールバック
        result_area = ui.column().classes("full-width")
        state["_result_area"] = result_area

    result_area.clear()

    try:
        # 因子オブジェクト構築
        factors = _build_factors(state)
        if not factors:
            ui.notify("因子が定義されていません", type="warning")
            return

        # 既存データ（②モードのみ）
        existing_df: pd.DataFrame | None = None
        if state["mode"] == "existing" and state.get("existing_df") is not None:
            raw_df = state["existing_df"]
            factor_names = [f.name for f in factors]
            cols = [c for c in factor_names if c in raw_df.columns]
            if cols:
                existing_df = raw_df[cols].dropna().reset_index(drop=True)

        criterion = state["criterion"]

        if criterion == "OA":
            # 直交表
            from nicegui import run as ng_run
            design_df, warn = await ng_run.io_bound(
                apply_orthogonal_array, state["oa_name"], factors
            )
            with result_area:
                _render_results_oa(design_df, warn, state)
        else:
            # D/E/I最適
            opt = DoEOptimizer(
                factors=factors,
                n_new=state["n_new"],
                criterion=criterion,
                max_candidates=state["max_candidates"],
                random_seed=state["random_seed"],
                n_starts=state["n_starts"],
                max_iter=state["max_iter"],
                existing_df=existing_df,
            )
            from nicegui import run as ng_run
            result = await ng_run.io_bound(opt.optimize)
            state["result"] = result

            with result_area:
                _render_results_optimal(result, state)

    except Exception as ex:
        import traceback
        with result_area:
            ui.label(f"⚠️ エラー: {ex}").classes("text-red text-body2")
            with ui.expansion("詳細トレースバック", icon="bug_report"):
                ui.label(traceback.format_exc()).classes("text-caption text-red").style(
                    "white-space:pre; font-family:monospace;"
                )
        logger.exception("DoE実行エラー")
    finally:
        run_btn.enable()
        run_btn.text = "🚀 実験計画を生成"


# ─────────────────────────────────────────────────────────────────────────────
# 因子オブジェクト構築
# ─────────────────────────────────────────────────────────────────────────────

def _build_factors(state: dict):
    from backend.doe import Factor, FactorType

    factors = []
    for fdef in state["factors"]:
        # ②モードで role が "target" / "ignore" の列はスキップ
        if fdef.get("_role") in ("target", "ignore"):
            continue

        name = fdef["name"].strip()
        ftype = fdef["type"]

        if ftype == "categorical":
            cats_raw = fdef.get("categories", "")
            cats = [c.strip() for c in cats_raw.split(",") if c.strip()]
            if not cats:
                cats = ["A", "B"]
            factors.append(Factor.categorical(name, cats))
        else:
            low = float(fdef.get("low", 0))
            high = float(fdef.get("high", 1))
            n_levels = int(fdef.get("n_levels", 5))
            factors.append(Factor.continuous(name, low, high, n_levels))

    return factors


# ─────────────────────────────────────────────────────────────────────────────
# 結果表示
# ─────────────────────────────────────────────────────────────────────────────

def _render_results_optimal(result, state: dict) -> None:
    from backend.doe import DoEResult
    res: DoEResult = result

    n_exist = sum(1 for v in res.is_new if not v)
    n_new = sum(1 for v in res.is_new if v)

    # ── サマリーカード ────────────────────────────────────────────────────────
    with ui.card().classes("full-width q-pa-sm q-mb-md").style(_SUCCESS_STYLE):
        with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
            ui.icon("check_circle", color="green").classes("text-h5")
            ui.label(f"✅ {res.criterion_name}最適計画 — 生成完了").classes("text-subtitle1 text-bold text-green")

        with ui.row().classes("q-gutter-lg"):
            _metric_chip("実験数（合計）", f"{len(res.design_df)}件")
            _metric_chip("既存（固定）", f"{n_exist}件", "blue-grey")
            _metric_chip("新規追加", f"{n_new}件", "green")
            _metric_chip("D効率", f"{res.d_efficiency:.4f}", "purple")

            if res.criterion_name == "D":
                _metric_chip("log det(X'X)", f"{res.criterion_value:.4f}", "indigo")
            elif res.criterion_name == "E":
                _metric_chip("最小固有値", f"{res.criterion_value:.4f}", "indigo")
            elif res.criterion_name == "I":
                _metric_chip("I基準値", f"{res.criterion_value:.4f}", "indigo")

        info = res.info
        ui.label(
            f"パラメータ: 候補{info.get('n_candidates',0)}点, "
            f"スタート{info.get('n_starts',0)}回, モデル{info.get('n_params',0)}列"
        ).classes("text-caption text-grey q-mt-xs")

    # ── 設計行列テーブル ──────────────────────────────────────────────────────
    _render_design_table(res.design_df, res.is_new)

    # ── ダウンロード ──────────────────────────────────────────────────────────
    _render_download(res.design_df, res.is_new, f"doe_{res.criterion_name.lower()}_result")


def _render_results_oa(design_df: pd.DataFrame, warn: str, state: dict) -> None:
    with ui.card().classes("full-width q-pa-sm q-mb-md").style(_SUCCESS_STYLE):
        with ui.row().classes("items-center q-gutter-sm"):
            ui.icon("check_circle", color="green").classes("text-h5")
            ui.label(f"✅ 直交表 — 生成完了").classes("text-subtitle1 text-bold text-green")

        with ui.row().classes("q-gutter-lg"):
            _metric_chip("実験数", f"{len(design_df)}件")
            _metric_chip("因子数", f"{len(design_df.columns)}個")

        if warn:
            ui.label(warn).classes("text-amber text-caption q-mt-xs")

    if not design_df.empty:
        is_new = [True] * len(design_df)
        _render_design_table(design_df, is_new)
        _render_download(design_df, is_new, "doe_oa_result")
    else:
        ui.label("設計点が生成できませんでした。因子数を確認してください。").classes("text-red text-body2")


def _render_design_table(df: pd.DataFrame, is_new: list[bool]) -> None:
    ui.label("📋 実験計画表").classes("text-subtitle2 q-mb-xs")

    # NiceGUI aggriadの代わりにシンプルなHTMLテーブルで表示
    df_display = df.copy()
    df_display.index = range(1, len(df) + 1)
    df_display.index.name = "No."

    rows = []
    for i, (idx, row) in enumerate(df_display.iterrows()):
        style = "" if is_new[i] else "background:rgba(100,100,100,0.12);"
        tag = "🆕" if is_new[i] else "📌"
        row_vals = [f"<td style='padding:4px 8px;'>{tag}</td>"]
        row_vals.append(f"<td style='padding:4px 8px; text-align:center;'>{idx}</td>")
        for val in row.values:
            if isinstance(val, float):
                row_vals.append(f"<td style='padding:4px 8px; text-align:right;'>{val:.4g}</td>")
            else:
                row_vals.append(f"<td style='padding:4px 8px;'>{val}</td>")
        rows.append(f"<tr style='{style}'>{''.join(row_vals)}</tr>")

    header_cells = ["<th style='padding:4px 8px; border-bottom:1px solid #444;'>状態</th>"]
    header_cells.append("<th style='padding:4px 8px; border-bottom:1px solid #444;'>No.</th>")
    for col in df.columns:
        header_cells.append(f"<th style='padding:4px 8px; border-bottom:1px solid #444;'>{col}</th>")

    html = (
        "<div style='overflow-x:auto;'>"
        "<table style='border-collapse:collapse; font-size:0.82rem; width:100%;'>"
        f"<thead><tr>{''.join(header_cells)}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "<div style='font-size:0.72rem; color:#888; margin-top:4px;'>"
        "📌 = 既存（固定）  🆕 = 新規追加</div>"
        "</div>"
    )
    ui.html(html).classes("full-width q-mb-md")


def _render_download(df: pd.DataFrame, is_new: list[bool], filename_base: str) -> None:
    def _make_csv():
        out = df.copy()
        out.insert(0, "is_new", ["新規" if n else "既存" for n in is_new])
        return out.to_csv(index=True, encoding="utf-8-sig")

    with ui.row().classes("q-gutter-sm"):
        ui.button(
            "⬇️ CSVダウンロード",
            on_click=lambda: ui.download(
                _make_csv().encode("utf-8-sig"),
                filename=f"{filename_base}.csv",
                media_type="text/csv",
            ),
        ).props("outline no-caps color=teal dense")

        ui.button(
            "⬇️ 実験計画のみ（新規点）",
            on_click=lambda: ui.download(
                df[[v for v, n in zip(df.index, is_new) if n]].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                if any(is_new) else b"",
                filename=f"{filename_base}_new_only.csv",
                media_type="text/csv",
            ),
        ).props("outline no-caps color=indigo dense")


def _metric_chip(label: str, value: str, color: str = "teal") -> None:
    with ui.card().classes("q-pa-xs").style(
        f"border:1px solid rgba(0,0,0,0.1); border-radius:6px; min-width:90px;"
    ):
        ui.label(label).classes(f"text-caption text-grey-5")
        ui.label(value).classes(f"text-subtitle2 text-bold text-{color}")
