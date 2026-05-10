# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/column_meta_editor.py

変数メタ情報エディタ。
各説明変数について以下のメタ情報をユーザーが設定できるUIコンポーネント。

  - monotonic: 単調性制約 (なし / 単調増加 / 単調減少)
  - linearity: 線形性ヒント (不明 / 線形 / 非線形)
  - group: グループラベル (GroupLasso等で使用)
  - scale_hint: スケーラー推奨ヒント
  - fixed: 特徴量選択で常に保持するか
  - description: 列の説明テキスト

state["column_meta"] = {列名: ColumnMeta.to_dict()} の形式で保存される。
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)

# ================================================================
# 定数・ラベル定義
# ================================================================

_MONOTONIC_OPTIONS = {
    "0":  "➖ なし",
    "1":  "📈 増加",
    "-1": "📉 減少",
    "2":  "🔄 自動",
}

_LINEARITY_OPTIONS = {
    "unknown":  "❓ 不明",
    "linear":   "📏 線形",
    "nonlinear": "〰️ 非線形",
}

_SCALE_HINT_OPTIONS = {
    "":            "🔄 自動",
    "standard":    "σ Standard",
    "minmax":      "📐 MinMax",
    "robust":      "💪 Robust",
    "maxabs":      "⬆️ MaxAbs",
    "power_yj":    "🔀 PowerYJ",
    "quantile_normal": "🔔 Quantile",
    "log":         "📊 Log",
    "none":        "🚫 なし",
}


def _get_meta(state: dict, col: str) -> dict:
    """state から列のメタ情報を取得（なければデフォルト）。"""
    cm = state.setdefault("column_meta", {})
    if col not in cm:
        cm[col] = {"monotonic": 0, "constraint_strength": None,
                   "linearity": "unknown", "group": None,
                   "scale_hint": None, "description": "", "fixed": False}
    return cm[col]


def _set_meta(state: dict, col: str, key: str, value: Any) -> None:
    """state の列メタ情報を更新する。"""
    cm = state.setdefault("column_meta", {})
    if col not in cm:
        cm[col] = {"monotonic": 0, "constraint_strength": None,
                   "linearity": "unknown", "group": None,
                   "scale_hint": None, "description": "", "fixed": False}
    cm[col][key] = value


# ================================================================
# メインレンダリング関数
# ================================================================

def render_column_meta_editor(state: dict, df: pd.DataFrame | None = None) -> None:
    """
    説明変数ごとのメタ情報設定UIをレンダリングする。

    Args:
        state: 共有ステート辞書。state["column_meta"] に結果を保存する。
        df: 現在のデータフレーム（列名取得に使用）。 None の場合は表示しない。
    """
    target_col = state.get("target_col", "")
    exclude_cols = set(state.get("exclude_cols", []))

    if df is None:
        ui.label("データを読み込むと変数メタ情報を設定できます").classes("text-caption text-grey")
        return

    # 説明変数の一覧取得（目的変数・除外列を除く）
    feature_cols = [
        c for c in df.columns
        if c != target_col and c not in exclude_cols
    ]

    if not feature_cols:
        ui.label("説明変数が見つかりません").classes("text-caption text-grey")
        return

    # ── ツールバー ──
    with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
        ui.icon("tune", color="cyan").classes("text-h6")
        ui.label("変数メタ情報設定").classes("text-subtitle1 text-bold")
        ui.badge(f"{len(feature_cols)} 変数", color="blue-grey").props("dense")

        # 一括クリアボタン
        def _clear_all():
            state["column_meta"] = {}
            ui.notify("メタ情報をすべてクリアしました", type="info")

        ui.button("🗑 全クリア", on_click=_clear_all).props(
            "flat dense size=xs color=grey no-caps"
        ).classes("q-ml-auto")

    # ── 検索フィルタ ──
    search_input = ui.input(
        placeholder="列名で検索...",
    ).props("outlined dense clearable").classes("full-width q-mb-sm").style("max-width:320px;")

    # ── 凡例 ──
    with ui.row().classes("items-center q-gutter-xs q-mb-xs text-caption text-grey"):
        ui.label("列名")
        ui.separator().props("vertical")
        ui.label("単調性")
        ui.separator().props("vertical")
        ui.label("制約強度")
        ui.separator().props("vertical")
        ui.label("線形性")
        ui.separator().props("vertical")
        ui.label("グループ")
        ui.separator().props("vertical")
        ui.label("スケーラーHint")
        ui.separator().props("vertical")
        ui.label("常時保持")
        ui.separator().props("vertical")
        ui.label("説明(任意)")

    # ── 変数リスト ──
    list_container = ui.column().classes("full-width")

    def _render_list(search: str = ""):
        list_container.clear()
        filtered = [c for c in feature_cols if not search or search.lower() in c.lower()]
        with list_container:
            if not filtered:
                ui.label("一致する変数がありません").classes("text-caption text-grey q-pa-sm")
                return

            for col in filtered:
                meta = _get_meta(state, col)
                _render_row(state, col, meta)

    def _render_row(state: dict, col: str, meta: dict) -> None:
        """1変数分の設定行をレンダリングする。"""
        mono_val = str(meta.get("monotonic", 0))
        lin_val = meta.get("linearity", "unknown")
        grp_val = meta.get("group") or ""
        scale_val = meta.get("scale_hint") or ""
        fixed_val = bool(meta.get("fixed", False))
        desc_val = meta.get("description", "")

        with ui.card().classes("full-width q-pa-xs q-mb-xs").style(
            "border:1px solid rgba(0,188,212,0.15); border-radius:6px; background:rgba(0,20,40,0.15);"
        ):
            with ui.row().classes("items-center full-width q-gutter-xs flex-nowrap"):
                # 列名
                ui.label(col).classes("text-body2 text-bold").style(
                    "min-width:150px; max-width:200px; overflow:hidden; text-overflow:ellipsis;"
                ).tooltip(col)

                # 単調性 (toggle buttons)
                mono_btns = {}
                with ui.row().classes("items-center q-gutter-none").style("min-width:140px;"):
                    for val, label in [("0", "➖"), ("1", "📈"), ("-1", "📉"), ("2", "🔄")]:
                        active = (val == mono_val)
                        btn_style = (
                            "background:rgba(0,188,212,0.3); border:1px solid rgba(0,188,212,0.6);"
                            if active else
                            "background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);"
                        )
                        def _on_mono(_, c=col, v=val):
                            _set_meta(state, c, "monotonic", int(v))
                        
                        # 自動(2)かつ解決済みの場合は表示を変更
                        display_label = label
                        if val == "2" and active:
                            res = meta.get("resolved_monotonic")
                            if res == 1:
                                display_label = "🔄(📈)"
                            elif res == -1:
                                display_label = "🔄(📉)"

                        b = ui.button(display_label, on_click=_on_mono).props(
                            "flat dense no-caps size=xs"
                        ).style(btn_style + "min-width:28px; margin:1px; border-radius:4px;")
                        b.tooltip({"0": "制約なし", "1": "単調増加", "-1": "単調減少", "2": "単調(自動検出)"}[val])

                # 制約強度 (weak/strong)
                strength_val = meta.get("constraint_strength") or ""
                _STRENGTH_OPTIONS = {"": "─ デフォルト", "weak": "🟢 弱い", "strong": "🟠 強い"}
                def _on_strength(e, c=col):
                    v = e.value if e.value else None
                    _set_meta(state, c, "constraint_strength", v)
                ui.select(
                    list(_STRENGTH_OPTIONS.keys()),
                    value=strength_val,
                    label=None,
                    on_change=_on_strength,
                ).props("outlined dense options-dense").style("min-width:110px;").without_auto_validation

                # 線形性 select
                def _on_lin(e, c=col):
                    _set_meta(state, c, "linearity", e.value)
                ui.select(
                    list(_LINEARITY_OPTIONS.keys()),
                    value=lin_val,
                    label=None,
                    on_change=_on_lin,
                ).props("outlined dense options-dense").style("min-width:120px;").without_auto_validation

                # グループ input
                def _on_grp(e, c=col):
                    v = e.value.strip() if e.value else None
                    _set_meta(state, c, "group", v if v else None)
                ui.input(
                    value=grp_val,
                    placeholder="グループ名",
                    on_change=_on_grp,
                ).props("outlined dense").style("min-width:100px; max-width:140px;")

                # スケーラーヒント
                def _on_scale(e, c=col):
                    v = e.value if e.value else None
                    _set_meta(state, c, "scale_hint", v)
                ui.select(
                    list(_SCALE_HINT_OPTIONS.keys()),
                    value=scale_val,
                    label=None,
                    on_change=_on_scale,
                ).props("outlined dense options-dense").style("min-width:120px;").without_auto_validation

                # 常時保持チェックボックス
                def _on_fixed(e, c=col):
                    _set_meta(state, c, "fixed", e.value)
                ui.checkbox(
                    value=fixed_val,
                    on_change=_on_fixed,
                ).tooltip("✅ チェックで特徴量選択から除外（常時使用）")

                # 説明テキスト
                def _on_desc(e, c=col):
                    _set_meta(state, c, "description", e.value or "")
                ui.input(
                    value=desc_val,
                    placeholder="説明（任意）",
                    on_change=_on_desc,
                ).props("outlined dense").style("min-width:160px; flex:1;")

    # 初期表示
    _render_list()

    # 検索入力で再描画
    search_input.on("update:model-value", lambda e: _render_list(e.args or ""))


# ================================================================
# state → ColumnMeta dict 変換ユーティリティ
# ================================================================

def build_column_meta_dict(state: dict) -> dict:
    """
    state["column_meta"] から ColumnMeta オブジェクト辞書を構築して返す。

    Returns:
        dict[str, ColumnMeta]
    """
    from backend.pipeline.column_selector import ColumnMeta
    raw = state.get("column_meta", {})
    result = {}
    for col, d in raw.items():
        if isinstance(d, dict):
            result[col] = ColumnMeta.from_dict(d)
        elif isinstance(d, ColumnMeta):
            result[col] = d
    return result


def extract_monotonic_from_column_meta(state: dict) -> dict[str, int]:
    """
    state["column_meta"] から単調性制約辞書（{列名: 1/-1}）を取得する。
    既存の state["monotonic_constraints"] とマージする（後者が優先）。

    Returns:
        dict[str, int] - monotonic != 0 の列のみ含む
    """
    raw = state.get("column_meta", {})
    result: dict[str, int] = {}
    for col, d in raw.items():
        v = int(d.get("monotonic", 0) if isinstance(d, dict) else d.monotonic)
        if v != 0:
            result[col] = v

    # 既存の直接指定があればマージ（直接指定を優先）
    existing = state.get("monotonic_constraints", {})
    for col, v in existing.items():
        if v != 0:
            result[col] = v

    return result
