"""
frontend_nicegui/components/auto_params_ui.py

NiceGUI版パラメータ自動UIレンダラー。

ParamSpecリストからNiceGUIウィジェットを自動生成する。
estimatorやChemAdapterの引数設定UIをハードコーディングなしで実現。

Usage:
    from backend.ui.param_schema import introspect_params
    from frontend_nicegui.components.auto_params_ui import render_param_editor

    specs = introspect_params(RandomForestRegressor)
    values = render_param_editor(specs, title="RandomForest設定")
    # → ユーザーが設定した値の辞書 (reactive)
"""
from __future__ import annotations

from typing import Any

from nicegui import ui

from backend.ui.param_schema import ParamSpec, get_basic_specs, get_advanced_specs


def render_param_editor(
    specs: list[ParamSpec],
    title: str = "パラメータ設定",
    *,
    values: dict[str, Any] | None = None,
    show_advanced: bool = True,
    compact: bool = False,
) -> dict[str, Any]:
    """
    ParamSpecリストからNiceGUI UIを自動生成する。

    Args:
        specs: ParamSpecリスト（backend.ui.param_schemaで生成）
        title: セクションタイトル
        values: 初期値辞書（ParamSpecのdefaultで補完）
        show_advanced: 上級者パラメータを折りたたみで表示するか
        compact: コンパクトモード（余白を減らす）

    Returns:
        dict[str, Any]: ユーザーが設定した値のリアクティブ辞書
    """
    if values is None:
        values = {}

    # デフォルト値で初期化
    for spec in specs:
        if spec.name not in values:
            values[spec.name] = spec.default

    basic = get_basic_specs(specs)
    advanced = get_advanced_specs(specs)

    padding = "q-pa-sm" if compact else "q-pa-md"

    if not specs:
        ui.label("⚙️ 設定可能なパラメータはありません").classes("text-grey-5")
        return values

    # ── 基本パラメータ ──
    if basic:
        with ui.card().classes(f"glass-card {padding} full-width q-mb-sm"):
            ui.label(f"⚙️ {title}").classes("text-subtitle2 q-mb-sm")
            for spec in basic:
                _render_widget(spec, values)

    # ── 上級者パラメータ（折りたたみ）──
    if advanced and show_advanced:
        with ui.expansion(
            f"🔧 詳細設定 ({len(advanced)}項目)",
            icon="tune",
        ).classes("full-width q-mt-xs"):
            with ui.card().classes(f"glass-card {padding} full-width"):
                for spec in advanced:
                    _render_widget(spec, values)

    return values


def render_model_param_editor(
    model_cls: type,
    title: str | None = None,
    *,
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    sklearn estimatorクラスからパラメータUIを自動生成する便利関数。

    Args:
        model_cls: estimatorクラス（例: RandomForestRegressor）
        title: セクションタイトル（Noneでクラス名を使用）
        values: 初期値辞書

    Returns:
        dict[str, Any]: ユーザーが設定した値の辞書
    """
    from backend.ui.param_schema import introspect_params
    specs = introspect_params(model_cls)
    if title is None:
        title = model_cls.__name__
    return render_param_editor(specs, title=title, values=values)


def render_adapter_param_editor(
    adapter_cls: type,
    title: str | None = None,
    *,
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    ChemAdapterクラスからパラメータUIを自動生成する便利関数。

    Args:
        adapter_cls: アダプタクラス
        title: セクションタイトル
        values: 初期値辞書

    Returns:
        dict[str, Any]: ユーザーが設定した値の辞書
    """
    from backend.ui.param_schema import introspect_adapter_class
    specs = introspect_adapter_class(adapter_cls)
    if title is None:
        title = getattr(adapter_cls, "__name__", str(adapter_cls))
    return render_param_editor(specs, title=title, values=values, compact=True)


# ─────────────────────────────────────────────────────────────
# ウィジェット描画
# ─────────────────────────────────────────────────────────────

def _render_widget(spec: ParamSpec, values: dict[str, Any]) -> None:
    """ParamSpecに応じたNiceGUIウィジェットを描画する。"""

    label = spec.name.replace("_", " ").title()
    tooltip_text = spec.description or spec.type_hint_raw or ""

    with ui.row().classes("items-center q-gutter-xs full-width"):

        if spec.param_type == "bool":
            cb = ui.checkbox(
                label,
                value=values.get(spec.name, spec.default or False),
                on_change=lambda e, n=spec.name: values.update({n: e.value}),
            )
            if tooltip_text:
                cb.tooltip(tooltip_text)

        elif spec.param_type == "int":
            num = ui.number(
                label,
                value=values.get(spec.name, spec.default or 0),
                min=spec.min_val,
                max=spec.max_val,
                step=spec.step or 1,
                on_change=lambda e, n=spec.name: values.update({n: int(e.value) if e.value is not None else None}),
            ).classes("w-40")
            if tooltip_text:
                num.tooltip(tooltip_text)

        elif spec.param_type == "float":
            num = ui.number(
                label,
                value=values.get(spec.name, spec.default or 0.0),
                min=spec.min_val,
                max=spec.max_val,
                step=spec.step or 0.01,
                format="%.4g",
                on_change=lambda e, n=spec.name: values.update({n: float(e.value) if e.value is not None else None}),
            ).classes("w-40")
            if tooltip_text:
                num.tooltip(tooltip_text)

        elif spec.param_type == "select":
            choices = spec.choices or []
            # Noneを許容する場合は選択肢に追加
            if spec.nullable and None not in choices:
                choices = [None] + choices
            sel = ui.select(
                label=label,
                options=choices,
                value=values.get(spec.name, spec.default),
                on_change=lambda e, n=spec.name: values.update({n: e.value}),
            ).classes("w-48")
            if tooltip_text:
                sel.tooltip(tooltip_text)

        elif spec.param_type == "multiselect":
            choices = spec.choices or []
            current = values.get(spec.name, spec.default or [])
            sel = ui.select(
                label=label,
                options=choices,
                value=current,
                multiple=True,
                on_change=lambda e, n=spec.name: values.update({n: list(e.value)}),
            ).classes("w-64")
            if tooltip_text:
                sel.tooltip(tooltip_text)

        elif spec.param_type in ("str", "text", "union"):
            current_val = values.get(spec.name, spec.default)
            inp = ui.input(
                label=label,
                value=str(current_val) if current_val is not None else "",
                on_change=lambda e, n=spec.name: values.update({n: e.value}),
            ).classes("w-48")
            if tooltip_text:
                inp.tooltip(tooltip_text)
            if spec.nullable:
                ui.label("(空欄=None)").classes("text-caption text-grey-6")

        else:
            # フォールバック
            inp = ui.input(
                label=f"{label} ({spec.param_type})",
                value=str(values.get(spec.name, spec.default) or ""),
                on_change=lambda e, n=spec.name: values.update({n: e.value}),
            ).classes("w-48")
            if tooltip_text:
                inp.tooltip(tooltip_text)
