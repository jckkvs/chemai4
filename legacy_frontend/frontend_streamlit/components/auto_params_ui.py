"""
frontend_streamlit/components/auto_params_ui.py

ParamSpecリストからStreamlitウィジェットを自動生成するレンダラー。

Usage:
    from backend.ui.param_schema import introspect_params
    from frontend_streamlit.components.auto_params_ui import render_param_editor

    specs = introspect_params(RandomForestRegressor)
    params = render_param_editor(specs, title="RandomForest パラメータ")
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from backend.ui.param_schema import (
    ParamSpec,
    get_basic_specs,
    get_advanced_specs,
    apply_params,
)


def render_param_editor(
    specs: list[ParamSpec],
    title: str = "パラメータ設定",
    *,
    values: dict[str, Any] | None = None,
    key_prefix: str = "",
    show_advanced: bool = True,
) -> dict[str, Any]:
    """
    ParamSpecリストからStreamlitウィジェットを自動生成する。

    Args:
        specs: ParamSpecリスト
        title: セクションタイトル
        values: 初期値辞書（Noneならデフォルト値を使用）
        key_prefix: ウィジェットキーの接頭辞（重複回避用）
        show_advanced: 上級者パラメータを表示するか

    Returns:
        dict[str, Any]: ユーザーが設定した値の辞書
    """
    if not specs:
        st.caption("⚙️ 設定可能なパラメータはありません")
        return {}

    if values is None:
        values = {}

    result: dict[str, Any] = {}

    basic = get_basic_specs(specs)
    advanced = get_advanced_specs(specs)

    # ── 基本パラメータ ──
    if basic:
        st.markdown(f"**⚙️ {title}**")
        cols = st.columns(2)
        for i, spec in enumerate(basic):
            with cols[i % 2]:
                val = _render_widget(spec, values, key_prefix)
                result[spec.name] = val

    # ── 上級者パラメータ（折りたたみ）──
    if advanced and show_advanced:
        with st.expander(f"🔧 詳細設定 ({len(advanced)}項目)", expanded=False):
            cols = st.columns(2)
            for i, spec in enumerate(advanced):
                with cols[i % 2]:
                    val = _render_widget(spec, values, key_prefix)
                    result[spec.name] = val

    return result


def _render_widget(
    spec: ParamSpec,
    values: dict[str, Any],
    prefix: str,
) -> Any:
    """ParamSpec 1つ分のStreamlitウィジェットを描画し、値を返す。"""

    key = f"{prefix}{spec.name}"
    label = spec.name.replace("_", " ").title()
    help_text = spec.description or None
    current = values.get(spec.name, spec.default)

    if spec.param_type == "bool":
        return st.checkbox(
            label,
            value=bool(current) if current is not None else False,
            key=key,
            help=help_text,
        )

    elif spec.param_type == "int":
        default_val = int(current) if current is not None else 0
        if spec.min_val is not None and spec.max_val is not None:
            # スライダー + 数値入力のハイブリッド
            return st.number_input(
                label,
                min_value=int(spec.min_val),
                max_value=int(spec.max_val),
                value=max(int(spec.min_val), min(default_val, int(spec.max_val))),
                step=int(spec.step or 1),
                key=key,
                help=help_text,
            )
        else:
            return st.number_input(
                label,
                value=default_val,
                step=int(spec.step or 1),
                key=key,
                help=help_text,
            )

    elif spec.param_type == "float":
        default_val = float(current) if current is not None else 0.0
        kwargs = {"key": key, "help": help_text, "step": spec.step or 0.01, "format": "%.4g"}
        if spec.min_val is not None:
            kwargs["min_value"] = spec.min_val
        if spec.max_val is not None:
            kwargs["max_value"] = spec.max_val
        if spec.min_val is not None and spec.max_val is not None:
            default_val = max(spec.min_val, min(default_val, spec.max_val))
        return st.number_input(label, value=default_val, **kwargs)

    elif spec.param_type == "select":
        choices = list(spec.choices) if spec.choices else []
        if spec.nullable:
            # Noneを選択肢に追加
            display_choices = ["(None)"] + [str(c) for c in choices]
            current_str = "(None)" if current is None else str(current)
            idx = display_choices.index(current_str) if current_str in display_choices else 0
            selected = st.selectbox(
                label, display_choices, index=idx, key=key, help=help_text,
            )
            return None if selected == "(None)" else selected
        else:
            str_choices = [str(c) for c in choices]
            current_str = str(current) if current is not None else (str_choices[0] if str_choices else "")
            idx = str_choices.index(current_str) if current_str in str_choices else 0
            return st.selectbox(label, str_choices, index=idx, key=key, help=help_text)

    elif spec.param_type == "multiselect":
        choices = list(spec.choices) if spec.choices else []
        current_list = current if isinstance(current, list) else []
        return st.multiselect(
            label, choices, default=current_list, key=key, help=help_text,
        )

    elif spec.param_type in ("str", "text", "union"):
        default_str = str(current) if current is not None else ""
        val = st.text_input(
            label, value=default_str, key=key, help=help_text,
        )
        if spec.nullable and val.strip() == "":
            return None
        return val

    else:
        # フォールバック
        default_str = str(current) if current is not None else ""
        return st.text_input(
            f"{label} ({spec.param_type})", value=default_str, key=key, help=help_text,
        )


# ─── 便利関数 ───


def render_model_param_editor(
    model_cls: type,
    title: str | None = None,
    *,
    values: dict[str, Any] | None = None,
    key_prefix: str = "",
) -> dict[str, Any]:
    """
    sklearn estimatorクラスからStreamlit UIを自動生成する。

    Args:
        model_cls: estimatorクラス
        title: セクションタイトル（Noneでクラス名を使用）
        values: 初期値辞書
        key_prefix: ウィジェットキーの接頭辞

    Returns:
        dict[str, Any]: ユーザーが設定した値の辞書
    """
    from backend.ui.param_schema import introspect_params
    specs = introspect_params(model_cls)
    if title is None:
        title = model_cls.__name__
    return render_param_editor(
        specs, title=title, values=values,
        key_prefix=key_prefix or f"model_{model_cls.__name__}_",
    )


def render_adapter_param_editor(
    adapter_cls: type,
    title: str | None = None,
    *,
    values: dict[str, Any] | None = None,
    key_prefix: str = "",
) -> dict[str, Any]:
    """
    ChemAdapterクラスからStreamlit UIを自動生成する。

    Args:
        adapter_cls: アダプタクラス
        title: セクションタイトル
        values: 初期値辞書
        key_prefix: ウィジェットキーの接頭辞

    Returns:
        dict[str, Any]: ユーザーが設定した値の辞書
    """
    from backend.ui.param_schema import introspect_adapter_class
    specs = introspect_adapter_class(adapter_cls)
    if title is None:
        title = getattr(adapter_cls, "__name__", str(adapter_cls))
    return render_param_editor(
        specs, title=title, values=values,
        key_prefix=key_prefix or f"adapter_{title}_",
    )
