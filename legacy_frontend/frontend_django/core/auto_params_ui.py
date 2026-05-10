"""
frontend_django/core/auto_params_ui.py

ParamSpecリストからDjango Formフィールド + HTMLテンプレートタグを
自動生成するレンダラー。

Usage (views.py):
    from backend.ui.param_schema import introspect_params
    from frontend_django.core.auto_params_ui import build_param_form

    FormClass = build_param_form(RandomForestRegressor, "RandomForest")
    form = FormClass(request.POST or None)
    if form.is_valid():
        params = form.get_params()

Usage (template):
    {% load auto_params_tags %}
    {% render_param_form form "RandomForest パラメータ" %}
"""
from __future__ import annotations

import json
from typing import Any

from django import forms
from django.utils.safestring import mark_safe

from backend.ui.param_schema import (
    ParamSpec,
    get_basic_specs,
    get_advanced_specs,
    introspect_params,
    introspect_adapter_class,
    apply_params,
)


# ─────────────────────────────────────────────────────────────
# 動的フォーム生成
# ─────────────────────────────────────────────────────────────


def build_param_form(
    cls: type,
    form_name: str = "ParamForm",
    *,
    values: dict[str, Any] | None = None,
) -> type[forms.Form]:
    """
    任意のPythonクラスからDjango Formクラスを動的に生成する。

    Args:
        cls: 解析対象のクラス（sklearn estimator, ChemAdapter等）
        form_name: 生成するFormクラス名
        values: 初期値辞書

    Returns:
        動的に生成されたDjango Formクラス
    """
    specs = introspect_params(cls)
    return _specs_to_form(specs, form_name, values)


def build_adapter_form(
    adapter_cls: type,
    form_name: str = "AdapterParamForm",
    *,
    values: dict[str, Any] | None = None,
) -> type[forms.Form]:
    """
    ChemAdapterクラスからDjango Formクラスを動的に生成する。
    """
    specs = introspect_adapter_class(adapter_cls)
    return _specs_to_form(specs, form_name, values)


def _specs_to_form(
    specs: list[ParamSpec],
    form_name: str,
    values: dict[str, Any] | None,
) -> type[forms.Form]:
    """ParamSpecリストからDjango Formクラスを生成する。"""

    fields: dict[str, forms.Field] = {}
    meta_specs: dict[str, ParamSpec] = {}

    for spec in specs:
        field = _spec_to_field(spec, values)
        if field is not None:
            fields[spec.name] = field
            meta_specs[spec.name] = spec

    # _specsアトリビュートを持つFormクラスを動的生成
    attrs = {
        **fields,
        "_param_specs": meta_specs,
        "get_params": _get_params_method,
        "get_basic_fields": _get_basic_fields_method,
        "get_advanced_fields": _get_advanced_fields_method,
    }
    return type(form_name, (forms.Form,), attrs)


def _spec_to_field(spec: ParamSpec, values: dict[str, Any] | None) -> forms.Field | None:
    """ParamSpecからDjango Formフィールドを生成する。"""

    initial = None
    if values and spec.name in values:
        initial = values[spec.name]
    elif spec.default is not None:
        initial = spec.default

    label = spec.name.replace("_", " ").title()
    help_text = spec.description or ""
    required = not spec.nullable

    widget_attrs = {"class": "form-control form-control-sm"}

    if spec.param_type == "bool":
        return forms.BooleanField(
            label=label,
            initial=bool(initial) if initial is not None else False,
            required=False,
            help_text=help_text,
            widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
        )

    elif spec.param_type == "int":
        kwargs: dict[str, Any] = {
            "label": label,
            "initial": int(initial) if initial is not None else 0,
            "required": required,
            "help_text": help_text,
            "widget": forms.NumberInput(attrs={**widget_attrs, "step": str(int(spec.step or 1))}),
        }
        if spec.min_val is not None:
            kwargs["min_value"] = int(spec.min_val)
        if spec.max_val is not None:
            kwargs["max_value"] = int(spec.max_val)
        return forms.IntegerField(**kwargs)

    elif spec.param_type == "float":
        kwargs = {
            "label": label,
            "initial": float(initial) if initial is not None else 0.0,
            "required": required,
            "help_text": help_text,
            "widget": forms.NumberInput(attrs={**widget_attrs, "step": str(spec.step or 0.01)}),
        }
        if spec.min_val is not None:
            kwargs["min_value"] = spec.min_val
        if spec.max_val is not None:
            kwargs["max_value"] = spec.max_val
        return forms.FloatField(**kwargs)

    elif spec.param_type == "select":
        choices_list = [(str(c), str(c)) for c in (spec.choices or [])]
        if spec.nullable:
            choices_list = [("", "(None)")] + choices_list
        return forms.ChoiceField(
            label=label,
            choices=choices_list,
            initial=str(initial) if initial is not None else "",
            required=required,
            help_text=help_text,
            widget=forms.Select(attrs=widget_attrs),
        )

    elif spec.param_type == "multiselect":
        choices_list = [(str(c), str(c)) for c in (spec.choices or [])]
        current = initial if isinstance(initial, list) else []
        return forms.MultipleChoiceField(
            label=label,
            choices=choices_list,
            initial=[str(c) for c in current],
            required=False,
            help_text=help_text,
            widget=forms.SelectMultiple(attrs={**widget_attrs, "size": min(5, len(choices_list))}),
        )

    elif spec.param_type in ("str", "text", "union"):
        return forms.CharField(
            label=label,
            initial=str(initial) if initial is not None else "",
            required=required,
            help_text=help_text + (" (空欄=None)" if spec.nullable else ""),
            widget=forms.TextInput(attrs=widget_attrs),
        )

    else:
        return forms.CharField(
            label=f"{label} ({spec.param_type})",
            initial=str(initial) if initial is not None else "",
            required=False,
            help_text=help_text,
            widget=forms.TextInput(attrs=widget_attrs),
        )


# ─────────────────────────────────────────────────────────────
# Formクラスのメソッド
# ─────────────────────────────────────────────────────────────


def _get_params_method(self) -> dict[str, Any]:
    """フォームの値からパラメータ辞書を生成する。"""
    specs = list(self._param_specs.values())
    cleaned = self.cleaned_data if hasattr(self, "cleaned_data") else {}
    return apply_params(specs, cleaned)


def _get_basic_fields_method(self):
    """basicグループのフィールド名リストを返す。"""
    return [
        name for name, spec in self._param_specs.items()
        if spec.group == "basic"
    ]


def _get_advanced_fields_method(self):
    """advancedグループのフィールド名リストを返す。"""
    return [
        name for name, spec in self._param_specs.items()
        if spec.group == "advanced"
    ]


# ─────────────────────────────────────────────────────────────
# HTML レンダリングヘルパー
# ─────────────────────────────────────────────────────────────


def render_param_form_html(
    form: forms.Form,
    title: str = "パラメータ設定",
    *,
    show_advanced: bool = True,
) -> str:
    """
    パラメータフォームをHTMLとしてレンダリングする。

    テンプレートタグが使えない場合の代替手段。
    """
    html_parts = []

    basic_fields = form.get_basic_fields() if hasattr(form, "get_basic_fields") else []
    advanced_fields = form.get_advanced_fields() if hasattr(form, "get_advanced_fields") else []

    # 基本パラメータ
    if basic_fields:
        html_parts.append(f'<div class="card mb-3"><div class="card-body">')
        html_parts.append(f'<h6 class="card-title">⚙️ {title}</h6>')
        html_parts.append('<div class="row g-2">')
        for name in basic_fields:
            if name in form.fields:
                html_parts.append(f'<div class="col-md-6">{_field_html(form, name)}</div>')
        html_parts.append('</div></div></div>')

    # 上級者パラメータ（アコーディオン）
    if advanced_fields and show_advanced:
        accordion_id = f"adv_{id(form)}"
        html_parts.append(f'''
        <div class="accordion mb-3" id="acc_{accordion_id}">
          <div class="accordion-item">
            <h2 class="accordion-header">
              <button class="accordion-button collapsed" type="button"
                      data-bs-toggle="collapse" data-bs-target="#collapse_{accordion_id}">
                🔧 詳細設定 ({len(advanced_fields)}項目)
              </button>
            </h2>
            <div id="collapse_{accordion_id}" class="accordion-collapse collapse">
              <div class="accordion-body">
                <div class="row g-2">
        ''')
        for name in advanced_fields:
            if name in form.fields:
                html_parts.append(f'<div class="col-md-6">{_field_html(form, name)}</div>')
        html_parts.append('</div></div></div></div></div>')

    return mark_safe("\n".join(html_parts))


def _field_html(form: forms.Form, name: str) -> str:
    """1つのフィールドをBootstrapスタイルのHTMLに変換する。"""
    bf = form[name]
    field = form.fields[name]
    errors = bf.errors

    error_html = ""
    if errors:
        error_html = f'<div class="invalid-feedback d-block">{"<br>".join(errors)}</div>'

    help_html = ""
    if field.help_text:
        help_html = f'<small class="form-text text-muted">{field.help_text}</small>'

    if isinstance(field.widget, forms.CheckboxInput):
        return f'''
        <div class="form-check mb-2">
            {bf} <label class="form-check-label" for="{bf.id_for_label}">{bf.label}</label>
            {help_html}{error_html}
        </div>'''

    return f'''
    <div class="mb-2">
        <label class="form-label small" for="{bf.id_for_label}">{bf.label}</label>
        {bf}{help_html}{error_html}
    </div>'''


# ─────────────────────────────────────────────────────────────
# JSON API用ヘルパー
# ─────────────────────────────────────────────────────────────


def specs_to_json(cls: type) -> str:
    """
    クラスのパラメータ仕様をJSON文字列に変換する。
    フロントエンドのJavaScript UIコンポーネント用。
    """
    specs = introspect_params(cls)
    return json.dumps([s.to_dict() for s in specs], ensure_ascii=False, indent=2)
