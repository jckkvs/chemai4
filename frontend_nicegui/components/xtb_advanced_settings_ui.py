"""
frontend_nicegui/components/xtb_advanced_settings_ui.py

xTB量子化学計算の詳細設定UIパネル — スタンドアロンコンポーネント。

- GFN-xTB手法の選択（GFN0/GFN1/GFN2/GFN-FF）
- 計算タイプ切り替え（単点/最適化/振動数/MD）
- 溶媒和モデル（ALPB/GBSA）と溶媒選択
- 分子電荷・スピン多重度の設定
- CPU並列数・タイムアウト設定
- 設定のexport/importサポート
- advanced_config.py のdataclassと完全連携

既存UIへの影響: なし（完全新規コンポーネント）
呼び出し方: render_xtb_advanced_settings(state) を適切な場所に追加
"""
from __future__ import annotations

import json
import logging
from typing import Any

from nicegui import ui

logger = logging.getLogger(__name__)

# ── GFN手法の選択肢 ──
_GFN_METHODS = {
    "gfn2": "GFN2-xTB（標準 / 推奨）",
    "gfn1": "GFN1-xTB（高速 / 低精度）",
    "gfn0": "GFN0-xTB（超高速 / 概算）",
    "gfnff": "GFN-FF（分子力場 / 最高速）",
}

# ── 計算タイプ ──
_CALC_TYPES = {
    "sp":   "⚡ 単点計算 (sp) — 最速",
    "opt":  "🚀 構造最適化 (opt) — 推奨",
    "freq": "🌊 振動数計算 (freq) — 熱力学量",
    "md":   "🎲 分子動力学 (md) — アンサンブル",
}

# ── 溶媒（ALPB/GBSA対応）──
_SOLVENTS = [
    "none", "water", "methanol", "ethanol",
    "acetone", "dmso", "thf", "chloroform",
    "ch2cl2", "benzene", "toluene", "ether",
    "acetonitrile", "hexane", "cyclohexane",
]

# ── 溶媒和モデル ──
_SOLVENT_MODELS = {
    "alpb": "ALPB（推奨 / 解析的）",
    "gbsa": "GBSA（高速 / 近似的）",
}


def render_xtb_advanced_settings(state: dict[str, Any]) -> None:
    """
    xTB量子化学計算の詳細設定パネルを描画する。

    設定は state["xtb_advanced_config"] に保存される。
    """
    # ── デフォルト設定の初期化 ──
    if "xtb_advanced_config" not in state:
        state["xtb_advanced_config"] = _get_default_config()

    cfg = state["xtb_advanced_config"]

    # ── ヘッダー ──
    with ui.card().classes("w-full").style(
        "background: rgba(123, 47, 247, 0.06); "
        "border: 1px solid rgba(123, 47, 247, 0.2); "
        "border-radius: 16px; padding: 20px;"
    ):
        with ui.row().classes("items-center gap-3 w-full"):
            ui.icon("bolt").classes("text-2xl").style("color: #a78bfa;")
            with ui.column().classes("gap-0"):
                ui.label("⚡ xTB 計算詳細設定").classes("text-lg font-bold").style(
                    "color: #e0e0f0;"
                )
                ui.label(
                    "GFN-xTB量子化学計算のパラメータをカスタマイズ"
                ).classes("text-sm").style("color: #a0a0c0;")

            # 現在の設定サマリーバッジ
            with ui.row().classes("gap-2 items-center ml-auto"):
                method = cfg.get("method", "gfn2")
                calc = cfg.get("calc_type", "opt")
                solvent = cfg.get("solvent", "none")
                ui.badge(method.upper(), color="purple")
                ui.badge(_CALC_TYPES.get(calc, calc)[:4], color="blue")
                if solvent != "none":
                    ui.badge(f"溶媒: {solvent}", color="cyan")

    ui.separator().classes("q-my-md")

    # ════════════════════════════════════════════════════
    # セクション1: 基本設定
    # ════════════════════════════════════════════════════
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 20px;"
    ):
        ui.label("🔧 基本設定").classes("text-md font-bold q-mb-md").style(
            "color: #e0e0f0;"
        )

        with ui.row().classes("gap-6 items-start q-mt-sm flex-wrap"):
            # GFN手法選択
            with ui.column().classes("gap-1"):
                ui.label("GFN手法").classes("text-sm").style("color: #a0a0c0;")
                method_select = ui.select(
                    options=_GFN_METHODS,
                    value=cfg.get("method", "gfn2"),
                    label="GFN-xTB手法",
                ).classes("w-64")

                def _on_method(e):
                    cfg["method"] = e.value
                    _update_summary()
                method_select.on_value_change(_on_method)

                # 手法の説明
                method_desc_labels = {
                    "gfn2": "🟢 最も精度が高い半経験的量子化学法。HOMO/LUMO、双極子、分極率の計算に推奨。",
                    "gfn1": "🟡 GFN2より高速。精度はやや低いが大規模系に適する。",
                    "gfn0": "🟠 非自己無撞着 (non-SCF)。超高速だが精度は低い。スクリーニング用途に。",
                    "gfnff": "⚡ 分子力場ベース。構造最適化のみ。電子的性質は計算不可。",
                }

                method_hint = ui.label(
                    method_desc_labels.get(cfg.get("method", "gfn2"), "")
                ).classes("text-xs q-mt-xs").style("color: #a0a0c0; max-width: 280px;")

                def _update_method_hint(e):
                    method_hint.text = method_desc_labels.get(e.value, "")
                method_select.on_value_change(_update_method_hint)

            # 計算タイプ
            with ui.column().classes("gap-1"):
                ui.label("計算タイプ").classes("text-sm").style("color: #a0a0c0;")
                calc_select = ui.select(
                    options=_CALC_TYPES,
                    value=cfg.get("calc_type", "opt"),
                    label="計算タイプ",
                ).classes("w-64")

                def _on_calc(e):
                    cfg["calc_type"] = e.value
                    # freq/mdの場合は警告
                    if e.value == "freq":
                        ui.notify("振動数計算は高精度ですが非常に時間がかかります", type="warning")
                    elif e.value == "md":
                        ui.notify("分子動力学はMD設定が必要です", type="info")
                    _update_summary()
                calc_select.on_value_change(_on_calc)

            # 精度レベル
            with ui.column().classes("gap-1"):
                ui.label("計算精度 (accuracy)").classes("text-sm").style(
                    "color: #a0a0c0;"
                )
                acc_input = ui.number(
                    label="accuracy",
                    value=cfg.get("accuracy", 1.0),
                    min=0.001, max=10.0, step=0.1,
                    format="%.3f",
                ).classes("w-32")
                ui.label(
                    "1.0=標準, 0.1=高精度, 10=低精度"
                ).classes("text-xs").style("color: #a0a0c0;")

                def _on_acc(e):
                    cfg["accuracy"] = float(e.value)
                acc_input.on_value_change(_on_acc)

    ui.separator().classes("q-my-md")

    # ════════════════════════════════════════════════════
    # セクション2: 溶媒和設定
    # ════════════════════════════════════════════════════
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 20px;"
    ):
        with ui.row().classes("items-center gap-2 q-mb-md"):
            ui.label("🌊 溶媒和モデル").classes("text-md font-bold").style(
                "color: #e0e0f0;"
            )
            ui.chip("推奨: ALPB + water", icon="water_drop").props(
                "outline color=cyan size=sm"
            )

        with ui.row().classes("gap-6 items-start flex-wrap"):
            # 溶媒選択
            with ui.column().classes("gap-1"):
                ui.label("溶媒").classes("text-sm").style("color: #a0a0c0;")
                solvent_select = ui.select(
                    options={s: s if s != "none" else "なし（気相計算）" for s in _SOLVENTS},
                    value=cfg.get("solvent", "none"),
                    label="溶媒",
                ).classes("w-48")

                def _on_solvent(e):
                    cfg["solvent"] = e.value
                    solvent_model_row.set_visibility(e.value != "none")
                    _update_summary()
                solvent_select.on_value_change(_on_solvent)

            # 溶媒和モデル
            with ui.column().classes("gap-1") as solvent_model_row:
                ui.label("溶媒和モデル").classes("text-sm").style("color: #a0a0c0;")
                model_select = ui.select(
                    options=_SOLVENT_MODELS,
                    value=cfg.get("solvent_model", "alpb"),
                    label="溶媒和モデル",
                ).classes("w-48")

                def _on_model(e):
                    cfg["solvent_model"] = e.value
                model_select.on_value_change(_on_model)

            # 溶媒なしの場合は非表示
            solvent_model_row.set_visibility(cfg.get("solvent", "none") != "none")

    ui.separator().classes("q-my-md")

    # ════════════════════════════════════════════════════
    # セクション3: 電荷・スピン設定
    # ════════════════════════════════════════════════════
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 20px;"
    ):
        ui.label("⚛️ 電荷・スピン多重度").classes("text-md font-bold q-mb-md").style(
            "color: #e0e0f0;"
        )

        # デフォルトモード / カスタムモード
        charge_mode = ui.toggle(
            {
                "auto": "自動（SMILESから推定）",
                "custom":  "手動設定",
            },
            value=cfg.get("charge_mode", "auto"),
        ).props("no-caps dense")

        custom_row = ui.row().classes("gap-6 items-end q-mt-md flex-wrap")

        with custom_row:
            charge_input = ui.number(
                label="電荷 (charge)",
                value=cfg.get("charge", 0),
                min=-10, max=10, step=1,
            ).classes("w-36")

            spin_input = ui.number(
                label="スピン多重度 (UHF)",
                value=cfg.get("spin", 0),
                min=0, max=10, step=1,
            ).classes("w-36")

            ui.label(
                "spin=0: 閉殻系 (restricted)\n"
                "spin=1: ラジカル (unrestricted, 1 α-excess electron)"
            ).classes("text-xs").style("color: #a0a0c0; white-space: pre-line;")

            def _on_charge(e):
                cfg["charge"] = int(e.value)
            def _on_spin(e):
                cfg["spin"] = int(e.value)
            charge_input.on_value_change(_on_charge)
            spin_input.on_value_change(_on_spin)

        def _on_charge_mode(e):
            cfg["charge_mode"] = e.value
            custom_row.set_visibility(e.value == "custom")

        charge_mode.on_value_change(_on_charge_mode)
        custom_row.set_visibility(cfg.get("charge_mode", "auto") == "custom")

    ui.separator().classes("q-my-md")

    # ════════════════════════════════════════════════════
    # セクション4: パフォーマンス設定
    # ════════════════════════════════════════════════════
    with ui.expansion(
        "🖥️ パフォーマンス・並列化設定",
        icon="settings",
    ).classes("w-full").style(
        "background: rgba(255,255,255,0.02); border-radius: 12px;"
    ):
        with ui.row().classes("gap-6 items-end flex-wrap q-pa-md"):
            # OMP並列数
            with ui.column().classes("gap-1"):
                ui.label("OMP並列スレッド数").classes("text-sm").style(
                    "color: #a0a0c0;"
                )
                omp_input = ui.number(
                    label="max_cores",
                    value=cfg.get("max_cores", 4),
                    min=1, max=64, step=1,
                ).classes("w-32")

                def _on_omp(e):
                    cfg["max_cores"] = int(e.value)
                omp_input.on_value_change(_on_omp)

            # タイムアウト
            with ui.column().classes("gap-1"):
                ui.label("タイムアウト (秒/分子)").classes("text-sm").style(
                    "color: #a0a0c0;"
                )
                timeout_input = ui.number(
                    label="timeout_s",
                    value=cfg.get("timeout_s", 120),
                    min=10, max=3600, step=10,
                ).classes("w-36")

                def _on_timeout(e):
                    cfg["timeout_s"] = int(e.value)
                timeout_input.on_value_change(_on_timeout)

            # 最大SCF反復
            with ui.column().classes("gap-1"):
                ui.label("最大SCF反復数").classes("text-sm").style(
                    "color: #a0a0c0;"
                )
                scf_input = ui.number(
                    label="max_iter",
                    value=cfg.get("max_iter", 250),
                    min=50, max=5000, step=50,
                ).classes("w-36")

                def _on_scf(e):
                    cfg["max_iter"] = int(e.value)
                scf_input.on_value_change(_on_scf)

    ui.separator().classes("q-my-md")

    # ════════════════════════════════════════════════════
    # 設定エクスポート/インポート
    # ════════════════════════════════════════════════════
    with ui.row().classes("gap-4 items-center"):
        def _export_config():
            json_str = json.dumps(cfg, ensure_ascii=False, indent=2)
            ui.download(
                json_str.encode("utf-8"),
                "xtb_advanced_config.json",
            )
            ui.notify("📥 xTB設定をエクスポートしました", type="positive")

        def _reset_config():
            state["xtb_advanced_config"] = _get_default_config()
            ui.notify("🔄 デフォルト設定にリセットしました", type="info")

        ui.button(
            "📤 設定をエクスポート",
            on_click=_export_config,
            icon="download",
        ).props("outline color=cyan size=sm")

        ui.button(
            "🔄 デフォルトにリセット",
            on_click=_reset_config,
            icon="refresh",
        ).props("flat color=grey size=sm")

    # ── 設定サマリーラベル ──
    summary_label = ui.label("").classes("text-xs q-mt-sm").style(
        "color: #a0a0c0;"
    )

    def _update_summary():
        m = cfg.get("method", "gfn2")
        c = cfg.get("calc_type", "opt")
        s = cfg.get("solvent", "none")
        sm = cfg.get("solvent_model", "alpb")
        chg = cfg.get("charge", 0)
        spin = cfg.get("spin", 0)
        cores = cfg.get("max_cores", 4)
        timeout = cfg.get("timeout_s", 120)

        solvent_str = f"{sm.upper()}/{s}" if s != "none" else "気相"
        summary_label.text = (
            f"設定サマリー: {m.upper()} | {c} | 溶媒={solvent_str} | "
            f"電荷={chg:+d} | UHF={spin} | {cores}スレッド | タイムアウト={timeout}s"
        )

    _update_summary()


def _get_default_config() -> dict[str, Any]:
    """デフォルトxTB設定を返す。"""
    return {
        "method": "gfn2",
        "calc_type": "opt",
        "accuracy": 1.0,
        "solvent": "none",
        "solvent_model": "alpb",
        "charge": 0,
        "spin": 0,
        "charge_mode": "auto",
        "max_cores": 4,
        "timeout_s": 120,
        "max_iter": 250,
    }


def apply_xtb_config_to_adapter(state: dict[str, Any]) -> dict[str, Any]:
    """
    state["xtb_advanced_config"] をXTBAdapterのkwargs形式に変換する。

    Returns:
        XTBAdapter(**kwargs) に渡せる辞書。
    """
    cfg = state.get("xtb_advanced_config", _get_default_config())
    kwargs: dict[str, Any] = {
        "method": cfg.get("method", "gfn2"),
        "calc_type": cfg.get("calc_type", "opt"),
        "accuracy": cfg.get("accuracy", 1.0),
        "max_cores": cfg.get("max_cores", 4),
        "timeout": cfg.get("timeout_s", 120),
        "max_iter": cfg.get("max_iter", 250),
    }

    # 溶媒設定
    solvent = cfg.get("solvent", "none")
    if solvent != "none":
        kwargs["solvent"] = solvent
        kwargs["solvent_model"] = cfg.get("solvent_model", "alpb")

    # 電荷/スピン
    if cfg.get("charge_mode", "auto") == "custom":
        kwargs["charge"] = cfg.get("charge", 0)
        kwargs["uhf"] = cfg.get("spin", 0)

    return kwargs
