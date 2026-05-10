"""
frontend_nicegui/components/advanced_settings/charge_spin_config.py

高度計算設定パネル（電荷・スピン・溶媒・pH）。
NiceGUI の ui.expansion で折りたたみ可能。

既存 UI コンポーネント (eda_panel.py, bayesian_opt_ui.py 等) には一切影響しない。
main.py 側で呼び出す場合は create_charge_spin_panel() を import するだけ。
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from nicegui import ui

logger = logging.getLogger(__name__)


def create_charge_spin_panel(
    on_change_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    電荷・スピン・溶媒設定パネルを描画する。

    Args:
        on_change_callback: 設定変更時に呼ばれるコールバック。
            引数は ``{"charge": int|None, "spin_multiplicity": int,
            "solvent": str|None, "ph": float|None}`` の辞書。

    Returns:
        UI ウィジェットの参照辞書（テスト・外部制御用）。
    """
    widgets: dict[str, Any] = {}

    with ui.expansion(
        "⚙️ 高度計算設定（電荷・スピン・環境）", icon="settings"
    ).props("dense").classes("w-full"):

        # ── 電荷設定 ──
        with ui.card().classes("w-full q-mb-sm"):
            ui.label("🔋 形式電荷設定").classes("text-lg font-bold")

            charge_input = ui.number(
                "電荷値", min=-10, max=10, step=1, value=0
            ).classes("w-32")

            charge_mode = ui.radio(
                options={"auto": "SMILESから自動推定", "manual": "手動指定"},
                value="auto",
            ).props("inline")

            def _toggle_charge(e):  # noqa: ANN001
                charge_input.set_visibility(e.value == "manual")

            charge_mode.on("update:model-value", _toggle_charge)
            charge_input.set_visibility(False)

            widgets["charge_mode"] = charge_mode
            widgets["charge_input"] = charge_input

        # ── スピン多重度 ──
        with ui.card().classes("w-full q-mb-sm"):
            ui.label("🌀 スピン多重度").classes("text-lg font-bold")
            ui.markdown(
                "`2S+1` : 1 = 閉殻（ほとんどの有機分子）, "
                "2 = ラジカル, 3 = 三重項（O₂, カルベン等）"
            )
            spin_select = ui.select(
                options={
                    1: "1（閉殻 / 単重項）",
                    2: "2（二重項 / ラジカル）",
                    3: "3（三重項）",
                    4: "4（四重項）",
                    5: "5（五重項）",
                },
                value=1,
            ).classes("w-48")
            widgets["spin_select"] = spin_select

        # ── 溶媒・pH ──
        with ui.card().classes("w-full q-mb-sm"):
            ui.label("🧪 溶媒・環境条件").classes("text-lg font-bold")

            solvent_select = ui.select(
                options={
                    "none": "気相（真空）",
                    "water": "水",
                    "methanol": "メタノール",
                    "ethanol": "エタノール",
                    "dmso": "DMSO",
                    "acetonitrile": "アセトニトリル",
                    "toluene": "トルエン",
                    "thf": "THF",
                    "hexane": "ヘキサン",
                },
                value="none",
                label="溶媒モデル（xTB ALPB）",
            ).classes("w-48")
            widgets["solvent_select"] = solvent_select

            ph_input = ui.number(
                "pH", min=0.0, max=14.0, step=0.1, value=7.4
            ).classes("w-32")
            widgets["ph_input"] = ph_input

            # 気相のときはpH非表示
            def _toggle_ph(e):  # noqa: ANN001
                ph_input.set_visibility(e.value not in ("none",))

            solvent_select.on("update:model-value", _toggle_ph)
            ph_input.set_visibility(False)

        # ── 適用ボタン ──
        def _apply_settings():
            config = {
                "charge": (
                    int(charge_input.value)
                    if charge_mode.value == "manual"
                    else None
                ),
                "spin_multiplicity": int(spin_select.value),
                "solvent": (
                    solvent_select.value
                    if solvent_select.value != "none"
                    else None
                ),
                "ph": (
                    float(ph_input.value)
                    if ph_input.visible
                    else None
                ),
            }
            logger.info("高度計算設定を適用: %s", config)
            if on_change_callback:
                on_change_callback(config)
            ui.notify("⚙️ 高度設定を適用しました", type="positive")

        ui.button(
            "✅ 設定を適用", on_click=_apply_settings
        ).props("icon=check color=primary")

    return widgets
