"""
Morgan フィンガープリント オプションUIコンポーネント。

数え上げ系（カウント）モード、ビット出現順並べ替え等の設定を提供する。
"""
from __future__ import annotations

from typing import Any

from nicegui import ui


def render_morgan_options(state: dict[str, Any]) -> None:
    """
    Morgan フィンガープリントのオプション設定UIを描画する。

    Args:
        state: 共有ステート辞書（morgan_count, morgan_radius, morgan_bits,
               morgan_order_by_appearance を格納）
    """
    with ui.card().classes("full-width q-pa-sm glass-card"):
        ui.label("🔬 Morgan フィンガープリント オプション").classes(
            "text-body1 text-bold q-mb-xs"
        )

        # ── 数え上げ系（カウント）モード ──
        with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
            cb_morgan_count = ui.checkbox(
                "数え上げ系（カウント）を使用",
                value=state.get("morgan_count", False),
            )
            cb_morgan_count.tooltip(
                "Morganフィンガープリントをカウントベース（部分構造の出現回数）で計算する。"
                "チェックすると、各ビットは0/1ではなく出現回数を持つ。"
            )

        # ── 半径設定 ──
        with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
            ui.label("半径:").classes("text-caption")
            inp_radius = ui.input(
                value=str(state.get("morgan_radius", 2)),
            )
            inp_radius.props("type=number min=1 max=4 step=1").classes("w-20")
            inp_radius.tooltip(
                "Morganフィンガープリントの半径（1-4）。"
                "半径2が標準的（ECFP4）。"
                "大きいほどより広い範囲の部分構造を捉える。"
            )

        # ── ビット数設定 ──
        with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
            ui.label("ビット数:").classes("text-caption")
            inp_bits = ui.input(
                value=str(state.get("morgan_bits", 2048)),
            )
            inp_bits.props("type=number min=512 max=4096 step=512").classes("w-24")
            inp_bits.tooltip(
                "Morganフィンガープリントのビット数（512-4096）。"
                "多いほど衝突（異なる構造が同じビットにマップされる）が減るが、"
                "特徴量が増える。"
            )

        # ── 出現順並べ替え ──
        with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
            cb_order = ui.checkbox(
                "ビットをデータセット出現順に並べる",
                value=state.get("morgan_order_by_appearance", False),
            )
            cb_order.tooltip(
                "データセットで最初に出現した順にMorganビットを並べ替える。"
                "部分構造の出現順を保持し、自然言語説明を付与する。"
            )

        # ── 自然言語説明の表示 ──
        with ui.expansion("💡 自然言語説明について", icon="info").classes(
            "full-width q-mt-xs"
        ):
            ui.label(
                "Morganフィンガープリントの各ビットには、"
                "半径と部分構造に関する説明が付与される。"
                "数え上げ系（カウント）モードでは、"
                "各ビットは「半径Xの部分構造の出現回数」を表す。"
                "出現順並べ替えを有効にすると、"
                "データセットで最初に出現した順にビットが並ぶ。"
            ).classes("text-caption text-grey-6")

        # ── 値の保存 ──
        def _save() -> None:
            try:
                radius = int(inp_radius.value) if inp_radius.value else 2
                bits = int(inp_bits.value) if inp_bits.value else 2048
                state["morgan_count"] = cb_morgan_count.value
                state["morgan_radius"] = max(1, min(4, radius))
                state["morgan_bits"] = max(512, min(4096, bits))
                state["morgan_order_by_appearance"] = cb_order.value
            except (ValueError, TypeError):
                ui.notify(
                    "⚠️ 数値の変換に失敗しました。デフォルト値を使用します。",
                    type="warning",
                )

        # ── 変更時の保存 ──
        cb_morgan_count.on_value_change(_save)
        inp_radius.on_value_change(_save)
        inp_bits.on_value_change(_save)
        cb_order.on_value_change(_save)

        # 初期保存
        _save()
