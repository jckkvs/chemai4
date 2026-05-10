"""
frontend_nicegui/components/flow_stepper.py

解析フロー進捗ステッパー: 「データ→EDA→パイプライン→順解析→逆解析」の
5段階フローをステッパーUIで可視化し、現在のステップと次アクションを提示する。

Implements: F-FLOW01〜FLOW05
"""
from __future__ import annotations

from typing import Any

from nicegui import ui


# ステップ定義
FLOW_STEPS = [
    {
        "key": "data",
        "label": "📂 データ読込",
        "icon": "upload_file",
        "check": lambda s: s.get("df") is not None,
        "next_action": "「📂 解析設定」タブでCSVを読み込んでください",
        "done_label": "データ読込済み",
    },
    {
        "key": "eda",
        "label": "🔍 EDA・特徴量選択",
        "icon": "analytics",
        "check": lambda s: bool(s.get("descriptor_sets")),
        "next_action": "「🧪 SMILES記述子」タブで特徴量セットを作成してください",
        "done_label": "特徴量セット作成済み",
    },
    {
        "key": "pipeline",
        "label": "⚙️ パイプライン設定",
        "icon": "settings",
        "check": lambda s: bool(s.get("pipeline_cfg")),
        "next_action": "「⚙️ パイプライン」タブでモデル・前処理を設定してください",
        "done_label": "パイプライン設定済み",
    },
    {
        "key": "analysis",
        "label": "🚀 順解析",
        "icon": "play_arrow",
        "check": lambda s: s.get("automl_result") is not None or bool(s.get("automl_results")),
        "next_action": "画面上部の「🚀 解析開始」ボタンを押してください",
        "done_label": "順解析完了",
    },
    {
        "key": "inverse",
        "label": "🔮 逆解析",
        "icon": "find_replace",
        "check": lambda s: bool(s.get("_inv", {}).get("results")),
        "next_action": "「🔮 逆解析」タブで目標物性と制約を設定してください",
        "done_label": "逆解析完了",
    },
]


def get_current_step(state: dict[str, Any]) -> int:
    """現在の進捗ステップ番号を返す (0-indexed)。

    完了済みのステップ数を返す。全ステップ完了時はlen(FLOW_STEPS)を返す。
    """
    for i, step in enumerate(FLOW_STEPS):
        if not step["check"](state):
            return i
    return len(FLOW_STEPS)


def render_flow_stepper(
    state: dict[str, Any],
    compact: bool = True,
) -> None:
    """解析フロー進捗ステッパーを描画する。

    Args:
        state: アプリケーション状態dict
        compact: True=コンパクト横並び, False=通常ステッパー

    Implements: F-FLOW01
    """
    current = get_current_step(state)

    if compact:
        _render_compact_stepper(state, current)
    else:
        _render_full_stepper(state, current)


def _render_compact_stepper(state: dict[str, Any], current: int) -> None:
    """コンパクト版: 横並びのステップインジケーター。"""
    with ui.row().classes("items-center q-gutter-none full-width justify-center"):
        for i, step in enumerate(FLOW_STEPS):
            is_done = step["check"](state)
            is_current = (i == current)

            # ステップカード
            if is_done:
                bg = "rgba(74,222,128,0.15)"
                border = "rgba(74,222,128,0.4)"
                text_cls = "text-green"
                icon_color = "green"
            elif is_current:
                bg = "rgba(0,212,255,0.15)"
                border = "rgba(0,212,255,0.5)"
                text_cls = "text-cyan text-bold"
                icon_color = "cyan"
            else:
                bg = "rgba(255,255,255,0.03)"
                border = "rgba(255,255,255,0.1)"
                text_cls = "text-grey-6"
                icon_color = "grey-6"

            with ui.card().classes("q-pa-xs text-center").style(
                f"min-width: 85px; max-width: 120px; background: {bg}; "
                f"border: 1px solid {border}; border-radius: 8px; cursor: default;"
            ):
                if is_done:
                    ui.icon("check_circle", color=icon_color, size="xs")
                elif is_current:
                    ui.icon(step["icon"], color=icon_color, size="xs")
                else:
                    ui.icon(step["icon"], color=icon_color, size="xs")

                ui.label(step["label"]).classes(
                    f"text-caption {text_cls}"
                ).style("font-size: 0.7rem; line-height: 1.1;")

            # 矢印（最後以外）
            if i < len(FLOW_STEPS) - 1:
                arrow_color = "green" if is_done else "grey-7"
                ui.icon("arrow_forward", color=arrow_color, size="xs").classes(
                    "q-mx-none"
                )

    # 現在のステップのアクションガイド
    if current < len(FLOW_STEPS):
        step = FLOW_STEPS[current]
        with ui.row().classes("items-center q-gutter-sm justify-center q-mt-xs"):
            ui.icon("lightbulb", color="amber", size="xs")
            ui.label(f"次のステップ: {step['next_action']}").classes(
                "text-caption text-amber"
            ).style("font-size: 0.75rem;")
    else:
        with ui.row().classes("items-center q-gutter-sm justify-center q-mt-xs"):
            ui.icon("celebration", color="green", size="xs")
            ui.label("✅ 全ステップ完了！").classes(
                "text-caption text-green"
            ).style("font-size: 0.75rem;")


def _render_full_stepper(state: dict[str, Any], current: int) -> None:
    """フル版: NiceGUI stepper コンポーネント。"""
    with ui.stepper().classes("full-width").props("vertical") as stepper:
        for i, step in enumerate(FLOW_STEPS):
            is_done = step["check"](state)
            icon = "check" if is_done else step["icon"]

            with ui.step(step["label"], icon=icon):
                if is_done:
                    ui.label(f"✅ {step['done_label']}").classes("text-green text-caption")
                elif i == current:
                    ui.label(f"👉 {step['next_action']}").classes("text-cyan text-caption")
                else:
                    ui.label("待機中").classes("text-grey text-caption")
