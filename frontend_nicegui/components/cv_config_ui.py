# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/cv_config_ui.py

交差検証（CV）設定UI — NiceGUI版
全20種のCV手法をカテゴリ別ラジオボタンで選択。
各CV手法のパラメータを動的に生成する。

⚠️ ドロップダウンリスト(ui.select単一選択)は使用禁止。
   ui.radio, ui.toggle, ui.slider, ui.number, ui.checkbox を使用。
"""
from __future__ import annotations

from typing import Any
from nicegui import ui

# ═══════════════════════════════════════════════════════════
# CV手法のスキーマ定義
# バックエンド cv_manager.py の _CV_REGISTRY と対応
# ═══════════════════════════════════════════════════════════

_CV_CATEGORIES = {
    "🤖 自動": {
        "icon": "🤖",
        "methods": {
            "auto": {
                "label": "auto（タスクに応じて自動選択）",
                "desc": "回帰→KFold / 分類→StratifiedKFold を自動判定",
                "params": {},
            },
        },
    },
    "📐 基本": {
        "icon": "📐",
        "methods": {
            "kfold": {
                "label": "K-Fold",
                "desc": "標準的なK分割交差検証",
                "params": {
                    "n_splits": {"type": "slider", "label": "分割数 (K)", "min": 2, "max": 20, "step": 1, "default": 5},
                    "shuffle": {"type": "checkbox", "label": "シャッフル", "default": True},
                },
            },
            "stratified_kfold": {
                "label": "Stratified K-Fold",
                "desc": "クラス比率を保持するK-Fold（分類向け）",
                "classification_only": True,
                "params": {
                    "n_splits": {"type": "slider", "label": "分割数 (K)", "min": 2, "max": 20, "step": 1, "default": 5},
                    "shuffle": {"type": "checkbox", "label": "シャッフル", "default": True},
                },
            },
        },
    },
    "🚪 Leave系": {
        "icon": "🚪",
        "methods": {
            "loo": {
                "label": "Leave-One-Out (LOO)",
                "desc": "1サンプルずつ除外（小データ向け、計算コスト高）",
                "params": {},
            },
            "lpo": {
                "label": "Leave-P-Out",
                "desc": "Pサンプルずつ除外（計算コスト：C(n,p) 通り）",
                "params": {
                    "p": {"type": "slider", "label": "除外サンプル数 (P)", "min": 1, "max": 5, "step": 1, "default": 2},
                },
            },
        },
    },
    "👥 グループ系": {
        "icon": "👥",
        "requires_groups": True,
        "methods": {
            "group_kfold": {
                "label": "Group K-Fold",
                "desc": "グループを考慮したK-Fold（同一グループは同じFoldに配置）",
                "params": {
                    "n_splits": {"type": "slider", "label": "分割数 (K)", "min": 2, "max": 20, "step": 1, "default": 5},
                },
            },
            "stratified_group_kfold": {
                "label": "Stratified Group K-Fold",
                "desc": "グループとクラス比率を両方保持",
                "classification_only": True,
                "params": {
                    "n_splits": {"type": "slider", "label": "分割数 (K)", "min": 2, "max": 20, "step": 1, "default": 5},
                    "shuffle": {"type": "checkbox", "label": "シャッフル", "default": True},
                },
            },
            "logo": {
                "label": "Leave-One-Group-Out (LOGO)",
                "desc": "1グループずつ除外して検証",
                "params": {},
            },
            "lpgo": {
                "label": "Leave-P-Groups-Out",
                "desc": "Pグループずつ除外して検証",
                "params": {
                    "n_groups": {"type": "slider", "label": "除外グループ数 (P)", "min": 1, "max": 10, "step": 1, "default": 2},
                },
            },
            "group_shuffle_split": {
                "label": "Group Shuffle Split",
                "desc": "グループ考慮のランダム分割",
                "params": {
                    "n_splits": {"type": "slider", "label": "繰返し回数", "min": 1, "max": 20, "step": 1, "default": 5},
                    "test_size": {"type": "slider_float", "label": "テスト比率", "min": 0.05, "max": 0.5, "step": 0.05, "default": 0.2},
                },
            },
        },
    },
    "🔁 繰返し": {
        "icon": "🔁",
        "methods": {
            "repeated_kfold": {
                "label": "Repeated K-Fold",
                "desc": "K-Foldを複数回繰り返す（安定性評価に有効）",
                "params": {
                    "n_splits": {"type": "slider", "label": "分割数 (K)", "min": 2, "max": 20, "step": 1, "default": 5},
                    "n_repeats": {"type": "slider", "label": "繰返し回数", "min": 2, "max": 20, "step": 1, "default": 3},
                },
            },
            "repeated_stratified_kfold": {
                "label": "Repeated Stratified K-Fold",
                "desc": "Stratified K-Foldを複数回繰り返す",
                "classification_only": True,
                "params": {
                    "n_splits": {"type": "slider", "label": "分割数 (K)", "min": 2, "max": 20, "step": 1, "default": 5},
                    "n_repeats": {"type": "slider", "label": "繰返し回数", "min": 2, "max": 20, "step": 1, "default": 3},
                },
            },
        },
    },
    "🎲 シャッフル": {
        "icon": "🎲",
        "methods": {
            "shuffle_split": {
                "label": "Shuffle Split",
                "desc": "ランダムにtrain/testを分割（Monte Carlo CV）",
                "params": {
                    "n_splits": {"type": "slider", "label": "繰返し回数", "min": 1, "max": 30, "step": 1, "default": 5},
                    "test_size": {"type": "slider_float", "label": "テスト比率", "min": 0.05, "max": 0.5, "step": 0.05, "default": 0.2},
                },
            },
            "stratified_shuffle_split": {
                "label": "Stratified Shuffle Split",
                "desc": "クラス比率保持のランダム分割",
                "classification_only": True,
                "params": {
                    "n_splits": {"type": "slider", "label": "繰返し回数", "min": 1, "max": 30, "step": 1, "default": 5},
                    "test_size": {"type": "slider_float", "label": "テスト比率", "min": 0.05, "max": 0.5, "step": 0.05, "default": 0.2},
                },
            },
        },
    },
    "📈 時系列": {
        "icon": "📈",
        "methods": {
            "timeseries": {
                "label": "Time Series Split",
                "desc": "時系列データ向け（未来→テスト、過去→学習）",
                "params": {
                    "n_splits": {"type": "slider", "label": "分割数", "min": 2, "max": 20, "step": 1, "default": 5},
                    "max_train_size": {"type": "number", "label": "最大学習サンプル数（0=制限なし）", "min": 0, "max": 100000, "step": 100, "default": 0},
                    "gap": {"type": "slider", "label": "ギャップ（学習-テスト間隔）", "min": 0, "max": 50, "step": 1, "default": 0},
                },
            },
            "walk_forward": {
                "label": "Walk-Forward Validation（WalkCV）",
                "desc": "拡張窓方式のウォークフォワード検証",
                "params": {
                    "n_splits": {"type": "slider", "label": "分割数", "min": 2, "max": 20, "step": 1, "default": 5},
                    "min_train_size": {"type": "number", "label": "最小学習サンプル数（0=自動）", "min": 0, "max": 100000, "step": 10, "default": 0},
                    "gap": {"type": "slider", "label": "ギャップ", "min": 0, "max": 50, "step": 1, "default": 0},
                },
            },
        },
    },
    "🧬 化学特化": {
        "icon": "🧬",
        "requires_smiles": True,
        "methods": {
            "scaffold": {
                "label": "Scaffold Split (Bemis-Murcko)",
                "desc": "分子骨格に基づく分割。新規骨格への汎化性能を評価。",
                "params": {
                    "n_splits": {"type": "slider", "label": "分割数 (K)", "min": 2, "max": 20, "step": 1, "default": 5},
                },
            },
        },
    },
}


def _render_dynamic_params(
    cv_key: str,
    method_schema: dict,
    state: dict,
) -> None:
    """選択されたCV手法に応じてパラメータUIを動的生成する。"""
    params = method_schema.get("params", {})
    if not params:
        ui.label("🎯 パラメータなし（そのまま実行可能）").classes(
            "text-caption text-grey"
        )
        return

    cv_params = state.setdefault("_cv_extra_params", {})

    for pkey, schema in params.items():
        ptype = schema["type"]
        label = schema["label"]
        default = schema["default"]
        current = cv_params.get(pkey, default)

        if ptype == "slider":
            with ui.row().classes("items-center q-gutter-sm full-width"):
                ui.label(label).classes("text-body2").style("min-width:180px;")
                sl = ui.slider(
                    min=schema["min"], max=schema["max"],
                    step=schema["step"], value=current,
                ).props("label-always").classes("col")

                val_label = ui.label(str(current)).classes("text-body2 text-bold").style("min-width:30px;")

                def _on_slider(e, key=pkey, vlbl=val_label):
                    cv_params[key] = int(e.value)
                    vlbl.set_text(str(int(e.value)))
                    if key == "n_splits":
                        state["cv_folds"] = int(e.value)

                sl.on("update:model-value", _on_slider)

        elif ptype == "slider_float":
            with ui.row().classes("items-center q-gutter-sm full-width"):
                ui.label(label).classes("text-body2").style("min-width:180px;")
                sl = ui.slider(
                    min=schema["min"], max=schema["max"],
                    step=schema["step"], value=current,
                ).props("label-always").classes("col")

                val_label = ui.label(f"{current:.2f}").classes("text-body2 text-bold").style("min-width:50px;")

                def _on_float_slider(e, key=pkey, vlbl=val_label):
                    cv_params[key] = round(float(e.value), 2)
                    vlbl.set_text(f"{float(e.value):.2f}")

                sl.on("update:model-value", _on_float_slider)

        elif ptype == "number":
            with ui.row().classes("items-center q-gutter-sm full-width"):
                ui.label(label).classes("text-body2").style("min-width:180px;")
                ui.number(
                    value=current,
                    min=schema["min"], max=schema["max"],
                    step=schema["step"],
                    on_change=lambda e, key=pkey: cv_params.update({key: int(e.value) if e.value else 0}),
                ).props("outlined dense").classes("col-3")

        elif ptype == "checkbox":
            ui.checkbox(
                label, value=current,
                on_change=lambda e, key=pkey: cv_params.update({key: e.value}),
            )


def _render_cv_dialog_content(state: dict) -> None:
    """CV設定ダイアログの本体コンテンツ。
    
    全CV手法（LOO/LOGO/KFold/Stratified/WalkCV等）を常に表示。
    カテゴリトグル → ラジオボタンで選択する。
    """
    task_type = state.get("task_type", "regression")
    has_groups = bool(state.get("group_col"))
    current_cv = state.get("cv_key", "auto")

    # ── 現在のカテゴリを特定 ──
    current_cat = list(_CV_CATEGORIES.keys())[0]
    for cname, cinfo in _CV_CATEGORIES.items():
        if current_cv in cinfo["methods"]:
            current_cat = cname
            break

    # ── カテゴリトグル ──
    ui.label("① CVカテゴリを選択").classes("text-subtitle2 text-bold q-mb-xs")
    cat_names = list(_CV_CATEGORIES.keys())
    cat_toggle = ui.toggle(
        {name: name for name in cat_names},
        value=current_cat,
    ).props("no-caps dense rounded").classes("q-mb-md flex-wrap")

    method_container = ui.column().classes("full-width")

    def _rebuild_methods(category_name: str):
        method_container.clear()
        cat_info = _CV_CATEGORIES[category_name]
        methods = cat_info["methods"]
        is_group_cat = cat_info.get("requires_groups", False)

        with method_container:
            if is_group_cat and not has_groups:
                ui.label(
                    "⚠️ グループ列が未設定です。「列の役割」タブでグループ列を指定してください。\n"
                    "　  グループ列なしでも選択はできますが、解析時に自動でKFoldに切り替わります。"
                ).classes("text-warning text-caption q-mb-sm")
                
            has_smiles = bool(state.get("smiles_cols") or state.get("mix_smiles_cols") or state.get("smiles_col") or state.get("smiles_column"))
            is_smiles_cat = cat_info.get("requires_smiles", False)
            if is_smiles_cat and not has_smiles:
                ui.label(
                    "⚠️ SMILES列が見つかりません。SMILES列を含むデータを読み込むか、役割を設定してください。\n"
                    "　  SMILES列なしでも選択はできますが、解析時に自動でKFoldに切り替わります。"
                ).classes("text-warning text-caption q-mb-sm")

            available = {}
            for mkey, minfo in methods.items():
                if minfo.get("classification_only") and task_type != "classification":
                    continue
                available[mkey] = f"{minfo['label']}"

            if not available:
                ui.label("このカテゴリで利用可能なCV手法はありません（分類タスクのみ対応）。").classes("text-caption text-grey")
                return

            sel_key = current_cv if current_cv in available else list(available.keys())[0]

            ui.label("② CV手法を選択").classes("text-subtitle2 text-bold q-mb-xs")

            method_radio = ui.radio(
                available, value=sel_key,
            ).props("dense").classes("full-width q-mb-sm")

            # 手法の説明表示
            desc_label = ui.label(
                methods.get(sel_key, {}).get("desc", "")
            ).classes("text-caption text-cyan q-mb-sm q-pl-sm")

            param_container = ui.column().classes("full-width q-pl-lg").style(
                "border-left:3px solid rgba(0,212,255,0.3);margin-left:12px;padding:8px;"
            )

            def _on_method_change(e):
                new_key = e.value
                state["cv_key"] = new_key
                state["_cv_extra_params"] = {}
                minfo = methods.get(new_key, {})
                desc_label.set_text(minfo.get("desc", ""))
                param_container.clear()
                with param_container:
                    _render_dynamic_params(new_key, minfo, state)

            method_radio.on_value_change(_on_method_change)

            state["cv_key"] = sel_key
            minfo = methods.get(sel_key, {})
            with param_container:
                _render_dynamic_params(sel_key, minfo, state)

    cat_toggle.on_value_change(lambda e: _rebuild_methods(e.value))
    _rebuild_methods(current_cat)

    # ── タイムアウト ──
    ui.separator().classes("q-my-sm")
    ui.label("⏱️ タイムアウト設定").classes("text-subtitle2 text-bold q-mb-xs")
    with ui.row().classes("items-center q-gutter-sm"):
        timeout_slider = ui.slider(
            min=30, max=3600, step=30,
            value=state.get("timeout", 300),
        ).props("label-always").classes("col-5")
        timeout_label = ui.label(f"{state.get('timeout', 300)}秒").classes("text-body2 text-bold")

        def _on_timeout(e):
            state["timeout"] = int(e.value)
            timeout_label.set_text(f"{int(e.value)}秒")

        timeout_slider.on("update:model-value", _on_timeout)


def _get_cv_display_name(cv_key: str) -> str:
    """cv_key に対応する表示名を返す。"""
    if cv_key == "auto":
        return "🤖 auto"
    for _cname, cinfo in _CV_CATEGORIES.items():
        if cv_key in cinfo["methods"]:
            return cinfo["methods"][cv_key]["label"]
    return cv_key


def render_cv_config(state: dict) -> None:
    """交差検証設定UIをインラインでレンダリングする。

    ダイアログではなく、expansion（折りたたみ）内に直接配置。
    サマリーは常時表示。
    """
    cv_key = state.get("cv_key", "auto")
    cv_name = _get_cv_display_name(cv_key)
    folds = state.get("cv_folds", state.get("_cv_extra_params", {}).get("n_splits", 5))
    timeout = state.get("timeout", 300)

    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(123,47,247,0.3); border-radius: 10px;"
        "background: rgba(30,10,50,0.25);"
    ):
        # ── サマリーヘッダー（常時表示） ──
        with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
            ui.icon("loop", color="purple").classes("text-h6")
            ui.label("交差検証（CV）").classes("text-subtitle1 text-bold")
            ui.badge(cv_name, color="cyan" if cv_key == "auto" else "teal").props("dense")

        with ui.row().classes("q-gutter-md text-caption text-grey q-mb-sm"):
            ui.label(f"方法: {cv_name}")
            if cv_key not in ("auto", "loo", "logo", "leave_one_out"):
                ui.label(f"分割数: {folds}")
            ui.label(f"タイムアウト: {timeout}秒")
            extra_params = state.get("_cv_extra_params", {})
            if extra_params.get("shuffle") is True:
                ui.label("シャッフル: ON")
            elif extra_params.get("shuffle") is False:
                ui.label("シャッフル: OFF")

        # ── 展開で詳細設定（インライン） ──
        with ui.expansion(
            "⚙️ CV設定を展開", icon="settings",
        ).classes("full-width").props("dense"):
            _render_cv_dialog_content(state)

