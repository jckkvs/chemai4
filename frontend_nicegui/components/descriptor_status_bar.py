"""
frontend_nicegui/components/descriptor_status_bar.py

選択中の記述子を常時表示するフローティングステータスバー。
タブパネルの「上部」に sticky で固定し、登録済みセットを目立つカード形式で一覧表示する。

Implements: F-2-6 | 記述子セット常時表示パネル
設計:
  - メインタブの直上（top: 0）に sticky 固定
  - 折りたたみ時: セット名バッジ列（登録各セット1個）+ 記述子数 + 操作ボタン
  - 展開時: 登録済みセットごとのカード（セット名・記述子数・チップ群・切替ボタン）
  - SMILES列未設定 or 計算未完了時は非表示
"""
from __future__ import annotations

import logging
from typing import Any

from nicegui import ui

logger = logging.getLogger(__name__)

_CHIP_COLORS: dict[str, str] = {
    "rdkit": "green",
    "molai": "purple",
    "xtb": "orange",
    "mordred": "blue",
    "cosmo": "deep-purple",
    "group_contrib": "teal",
    "descriptastorus": "lime",
    "molfeat": "pink",
    "mol2vec": "indigo",
    "chemprop": "red",
    "uma": "amber",
    "padel": "light-blue",
    "unipka": "cyan",
    "morgan": "blue-grey",
    "maccs": "light-green",
    "avalon": "brown",
}

_SET_COLORS: list[str] = [
    "#00d4ff", "#7b2ff7", "#4ade80", "#fb923c",
    "#f472b6", "#a3e635", "#38bdf8", "#facc15",
]


def _guess_group(name: str) -> str:
    nl = name.lower()
    if nl.startswith("molai_") or nl.startswith("cnn_pca_"):
        return "molai"
    if nl.startswith("xtb_") or name in (
        "HomoEnergy", "LumoEnergy", "HomoLumoGap",
        "DipoleMoment", "Polarizability",
    ):
        return "xtb"
    if nl.startswith("mordred_") or nl.startswith("mrd_"):
        return "mordred"
    if nl.startswith("joback_"):
        return "group_contrib"
    if nl.startswith("ds_"):
        return "descriptastorus"
    if nl.startswith("molfeat_"):
        return "molfeat"
    if nl.startswith("mol2vec_"):
        return "mol2vec"
    if nl.startswith("chemprop_"):
        return "chemprop"
    if nl.startswith("uma_"):
        return "uma"
    if nl.startswith("padel_"):
        return "padel"
    if nl.startswith("pka") or name == "pKa_pred":
        return "unipka"
    if nl.startswith("mu_") or nl.startswith("ln_gamma") or nl.startswith("cosmo_"):
        return "cosmo"
    if nl.startswith("morgan"):
        return "morgan"
    if nl.startswith("maccs"):
        return "maccs"
    if nl.startswith("avalon"):
        return "avalon"
    if nl.startswith("fr_"):
        return "rdkit"
    return "rdkit"


def render_descriptor_status_bar(state: dict[str, Any]) -> None:
    """
    記述子セット常時表示バーを描画する。

    タブパネルの直上に配置。precalc_done=True の場合のみ内容を描画する。
    """
    bar_container = ui.column().classes("full-width q-mb-sm")

    def _rebuild_bar():
        bar_container.clear()

        if not state.get("precalc_done") or state.get("precalc_df") is None:
            return

        precalc_df = state["precalc_df"]
        total_available = precalc_df.shape[1]

        sets: dict = state.get("descriptor_sets", {})
        current_set = state.get("current_set_name", "デフォルト")
        active: list = state.get("active_descriptors", state.get("selected_descriptors", []))
        n_use = len(active)
        n_samples = len(state["df"]) if state.get("df") is not None else 0

        ratio_warn = n_samples > 0 and n_use > n_samples
        _expanded: dict = {"value": state.get("_desc_bar_expanded", False)}

        with bar_container:
            with ui.card().classes("full-width q-pa-none").style(
                "background: rgba(8, 12, 28, 0.97);"
                "border: 1px solid rgba(0, 212, 255, 0.45);"
                "border-radius: 10px;"
                "box-shadow: 0 4px 24px rgba(0, 212, 255, 0.15);"
                "backdrop-filter: blur(14px);"
                "overflow: visible;"
            ):
                # ════════════════════════════════════════
                # ── 常時表示ヘッダー行 ──
                # ════════════════════════════════════════
                with ui.row().classes("items-center full-width q-px-lg q-py-sm").style(
                    "min-height: 54px; gap: 12px;"
                ):
                    # グロウアイコン
                    ui.html(
                        '<span style="font-size:20px; color:#00d4ff; '
                        'filter: drop-shadow(0 0 6px #00d4ff);">⚗️</span>'
                    )

                    ui.label("記述子セット").style(
                        "color: #00d4ff; font-weight: 700; font-size: 15px; "
                        "letter-spacing: 0.08em; text-shadow: 0 0 8px rgba(0,212,255,0.5);"
                    )

                    # フォーカスカウント
                    count_color = "#f87171" if ratio_warn else "#4ade80"
                    ui.html(
                        f'<span style="font-size: 20px; font-weight: 900; color: {count_color}; '
                        f'text-shadow: 0 0 10px {count_color}80;">{n_use}</span>'
                        f'<span style="font-size: 12px; color: #9ca3af; margin-left:2px;">/ {total_available} 記述子</span>'
                    )

                    if ratio_warn:
                        ui.html(
                            '<span style="color:#fbbf24; font-size:13px;">⚠️ 過学習リスク</span>'
                        ).tooltip(f"記述子数({n_use}) > サンプル数({n_samples})")

                    ui.space()

                    # セット切替バッジ一覧（登録済みセットを1個ずつカラーバッジで表示）
                    for i, sn in enumerate(sets.keys()):
                        is_cur = (sn == current_set)
                        color = _SET_COLORS[i % len(_SET_COLORS)]
                        s_descs = sets[sn].get("descriptors")
                        s_count = len(s_descs) if s_descs else total_available
                        border = "3px" if is_cur else "1px"
                        opacity = "1.0" if is_cur else "0.55"

                        def _switch(name=sn):
                            state["current_set_name"] = name
                            if sets[name].get("descriptors"):
                                state["active_descriptors"] = list(sets[name]["descriptors"])
                                state["selected_descriptors"] = list(sets[name]["descriptors"])
                            ui.notify(f"🔄 「{name}」に切替", type="info", timeout=1800)
                            _rebuild_bar()
                            ref = state.get("_refresh_tabs")
                            if ref:
                                try:
                                    ref()
                                except Exception:
                                    pass

                        ui.button(
                            f"{'✦ ' if is_cur else ''}{sn}  {s_count}",
                            on_click=_switch,
                        ).style(
                            f"background: transparent; border: {border} solid {color};"
                            f"color: {color}; border-radius: 20px; font-size: 12px;"
                            f"font-weight: {'800' if is_cur else '500'};"
                            f"padding: 4px 14px; opacity: {opacity};"
                            f"box-shadow: {'0 0 10px ' + color + '66' if is_cur else 'none'};"
                            "transition: all 0.2s;"
                        ).props("flat dense no-caps")

                    ui.separator().props("vertical").style("height:32px; opacity:0.3;")

                    def _all_on():
                        all_cols = list(precalc_df.columns)
                        state["active_descriptors"] = all_cols
                        state["selected_descriptors"] = all_cols
                        ui.notify(f"✅ 全{len(all_cols)}記述子をON", type="positive", timeout=2000)
                        _rebuild_bar()

                    def _all_off():
                        state["active_descriptors"] = []
                        state["selected_descriptors"] = []
                        ui.notify("⬜ 全記述子をOFF", type="info", timeout=2000)
                        _rebuild_bar()

                    ui.button("全ON", on_click=_all_on).props(
                        "flat dense size=sm no-caps"
                    ).style("color:#00d4ff;")
                    ui.button("全OFF", on_click=_all_off).props(
                        "flat dense size=sm no-caps"
                    ).style("color:#9ca3af;")

                    def _toggle():
                        state["_desc_bar_expanded"] = not state.get("_desc_bar_expanded", False)
                        _rebuild_bar()

                    ui.button(
                        icon="expand_more" if not _expanded["value"] else "expand_less",
                        on_click=_toggle,
                    ).props("flat dense round size=sm").style("color:#6b7280;")

                # ════════════════════════════════════════
                # ── 展開部分: セットごとのカード群 ──
                # ════════════════════════════════════════
                if state.get("_desc_bar_expanded"):
                    ui.separator().style("border-color: rgba(0,212,255,0.2); margin: 0 16px;")
                    with ui.row().classes("full-width q-px-lg q-py-md").style(
                        "flex-wrap: wrap; gap: 16px; align-items: flex-start;"
                    ):
                        for i, (sn, sdata) in enumerate(sets.items()):
                            color = _SET_COLORS[i % len(_SET_COLORS)]
                            s_descs: list = sdata.get("descriptors") or active
                            s_n = len(s_descs)
                            is_cur = (sn == current_set)

                            with ui.card().style(
                                f"background: rgba(255,255,255,0.03);"
                                f"border: {'2px' if is_cur else '1px'} solid {color}{'99' if not is_cur else ''};"
                                f"border-radius: 10px;"
                                f"padding: 14px 16px;"
                                f"min-width: 260px; max-width: 420px; flex: 1;"
                                f"box-shadow: {'0 0 16px ' + color + '44' if is_cur else 'none'};"
                            ):
                                # セットヘッダー
                                with ui.row().classes("items-center q-mb-sm").style("gap:8px;"):
                                    ui.html(
                                        f'<span style="display:inline-block; width:10px; height:10px;'
                                        f'border-radius:50%; background:{color}; '
                                        f'box-shadow: 0 0 6px {color};"></span>'
                                    )
                                    ui.label(sn).style(
                                        f"font-weight: 800; font-size: 14px; color: {color};"
                                    )
                                    ui.html(
                                        f'<span style="font-size:12px; color:#9ca3af; margin-left:4px;">'
                                        f'{s_n} 記述子</span>'
                                    )
                                    ui.space()
                                    if is_cur:
                                        ui.html(
                                            '<span style="font-size:11px; color:#4ade80; '
                                            'border:1px solid #4ade8066; border-radius:20px; '
                                            'padding: 1px 8px;">使用中</span>'
                                        )
                                    else:
                                        def _set_switch(name=sn):
                                            state["current_set_name"] = name
                                            if sets[name].get("descriptors"):
                                                state["active_descriptors"] = list(sets[name]["descriptors"])
                                                state["selected_descriptors"] = list(sets[name]["descriptors"])
                                            ui.notify(f"🔄 「{name}」に切替", type="info", timeout=1800)
                                            _rebuild_bar()

                                        ui.button("切替", on_click=_set_switch).props(
                                            "flat dense size=xs no-caps"
                                        ).style(f"color:{color}; border: 1px solid {color}66; border-radius:20px;")

                                # チップ一覧（最大30個）
                                with ui.element("div").style(
                                    "display:flex; flex-wrap:wrap; gap:4px; max-height: 120px; overflow-y: auto;"
                                ):
                                    display_n = min(30, len(s_descs))
                                    for desc_name in s_descs[:display_n]:
                                        group = _guess_group(desc_name)
                                        chip_color = _CHIP_COLORS.get(group, "grey")
                                        ui.chip(desc_name, color=chip_color).props(
                                            "dense size=xs outline"
                                        ).style("font-size:10px; max-width:160px; overflow:hidden;")

                                    if len(s_descs) > 30:
                                        ui.html(
                                            f'<span style="font-size:11px; color:#6b7280; padding:2px 6px;">'
                                            f'... 他 {len(s_descs) - 30} 個</span>'
                                        )

                    # フッター行
                    with ui.row().classes("items-center q-px-lg q-pb-sm").style("gap:12px;"):
                        def _open_detail():
                            switch_fn = state.get("_switch_to_data_smiles")
                            if switch_fn:
                                switch_fn()
                            else:
                                ui.notify(
                                    "📂 解析設定 → ⚗️ SMILES特徴量 タブで詳細選択できます",
                                    type="info",
                                )

                        ui.button("🔬 記述子詳細設定を開く", on_click=_open_detail).props(
                            "outline size=sm no-caps"
                        ).style("border-color:#00d4ff; color:#00d4ff; border-radius:8px;")

                        ui.space()

                        if n_samples > 0:
                            ratio = n_use / n_samples
                            color = "#4ade80" if ratio < 0.5 else "#fbbf24" if ratio < 1.0 else "#f87171"
                            ui.html(
                                f'<span style="font-size:12px; color:{color};">'
                                f'記述子/サンプル比: {ratio:.2f}</span>'
                            )

    _rebuild_bar()
    state["_refresh_descriptor_bar"] = _rebuild_bar
