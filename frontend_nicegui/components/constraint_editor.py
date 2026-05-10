"""
frontend_nicegui/components/constraint_editor.py

制約設定エディタ: 逆解析で使用する制約をGUIで設定する。

既存の backend/optim/constraints.py の6制約クラスを直接利用:
    - RangeConstraint: 範囲制約
    - SumConstraint: 合計制約（組成和=100等）
    - InequalityConstraint: 線形不等式
    - AtLeastNConstraint: 最低N個以上
    - CustomConstraint: Python式

Implements: F-CONS-UI01〜UI04
"""
from __future__ import annotations

from typing import Any

from nicegui import ui


# ═══════════════════════════════════════════════════════════
# プリセットテンプレート
# ═══════════════════════════════════════════════════════════

CONSTRAINT_PRESETS = [
    {
        "label": "🧪 組成和 = 100%",
        "icon": "pie_chart",
        "description": "選択した変数の合計が100になる制約（wt%組成系）",
        "type": "sum",
        "default_target": 100.0,
        "default_tolerance": 0.01,
    },
    {
        "label": "📏 範囲制約",
        "icon": "straighten",
        "description": "特定変数の上限・下限を設定",
        "type": "range",
    },
    {
        "label": "⚖️ 線形不等式",
        "icon": "balance",
        "description": "例: 2*A + B ≤ 50",
        "type": "inequality",
    },
    {
        "label": "✅ 最低N個以上",
        "icon": "checklist",
        "description": "選択変数のうち少なくともN個が閾値を超える",
        "type": "at_least_n",
    },
    {
        "label": "🔧 カスタム式",
        "icon": "code",
        "description": "Python式による自由な制約（例: A * B <= 50）",
        "type": "custom",
    },
]


# ═══════════════════════════════════════════════════════════
# メインコンポーネント
# ═══════════════════════════════════════════════════════════

def render_constraint_editor(state: dict[str, Any]) -> None:
    """制約設定エディタを描画する。

    state["_inv"]["constraints_list"] に Constraint オブジェクトのリストを格納。
    """
    inv = state.setdefault("_inv", {})
    constraints_list: list[dict] = inv.setdefault("constraints_list", [])

    feature_columns = _get_feature_columns(state)

    ui.label("🔒 制約設定").classes("text-h6 text-bold q-mb-sm")
    ui.label(
        "逆解析の探索範囲を制約で絞り込みます。"
        "プリセットテンプレートから追加するか、カスタム式を入力してください。"
    ).classes("text-caption text-grey-5 q-mb-md")

    # ── プリセットボタン群 ──
    ui.label("📌 テンプレートから追加").classes("text-subtitle2 q-mb-xs")
    with ui.row().classes("q-gutter-sm q-mb-md flex-wrap"):
        for preset in CONSTRAINT_PRESETS:
            def _add_preset(p=preset):
                _add_constraint_from_preset(p, constraints_list, feature_columns)
                _refresh_list()

            ui.button(
                preset["label"],
                on_click=_add_preset,
                icon=preset["icon"],
            ).props("outline size=sm no-caps").tooltip(preset["description"])

    # ── 設定済み制約一覧 ──
    list_container = ui.column().classes("full-width")
    
    def _refresh_list():
        list_container.clear()
        with list_container:
            if not constraints_list:
                ui.label("制約なし（全範囲を探索します）").classes(
                    "text-caption text-grey-5"
                )
                return

            for idx, c in enumerate(constraints_list):
                _render_constraint_card(idx, c, constraints_list, _refresh_list)

    _refresh_list()

    # ── サマリー ──
    ui.separator().classes("q-my-md")
    n_active = sum(1 for c in constraints_list if c.get("active", True))
    with ui.row().classes("items-center q-gutter-sm"):
        ui.icon("info", color="cyan", size="xs")
        ui.label(f"有効な制約: {n_active}件").classes("text-caption text-cyan")
        if n_active > 0:
            ui.button(
                "🗑️ 全削除",
                on_click=lambda: (constraints_list.clear(), _refresh_list()),
            ).props("flat color=red size=sm no-caps")


def _render_constraint_card(
    idx: int,
    constraint: dict,
    constraints_list: list[dict],
    on_change,
) -> None:
    """個別制約のカード表示。"""
    is_active = constraint.get("active", True)
    c_type = constraint.get("type", "unknown")
    
    type_labels = {
        "sum": "🧪 組成和",
        "range": "📏 範囲",
        "inequality": "⚖️ 不等式",
        "at_least_n": "✅ 最低N個",
        "custom": "🔧 カスタム",
    }
    type_label = type_labels.get(c_type, c_type)

    bg = "rgba(0,212,255,0.08)" if is_active else "rgba(255,255,255,0.03)"
    border = "rgba(0,212,255,0.2)" if is_active else "rgba(255,255,255,0.1)"

    with ui.card().classes("full-width q-pa-xs q-mb-xs").style(
        f"background: {bg}; border: 1px solid {border}; border-radius: 8px;"
    ):
        with ui.row().classes("items-center full-width justify-between"):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.checkbox(
                    "",
                    value=is_active,
                    on_change=lambda e, i=idx: (
                        constraints_list[i].update({"active": e.value}),
                        on_change(),
                    ),
                ).props("dense")
                ui.badge(type_label, color="grey").props("dense outline")
                ui.label(
                    constraint.get("description", "制約")
                ).classes("text-caption")

            ui.button(
                icon="delete",
                on_click=lambda i=idx: (
                    constraints_list.pop(i),
                    on_change(),
                ),
            ).props("flat round size=xs color=red")

        # 詳細パラメータ表示
        params = constraint.get("params", {})
        if params:
            with ui.row().classes("q-gutter-sm text-caption text-grey-5 q-mt-xs"):
                for k, v in params.items():
                    ui.label(f"{k}: {v}")


def _add_constraint_from_preset(
    preset: dict,
    constraints_list: list[dict],
    feature_columns: list[str],
) -> None:
    """プリセットから制約を追加する。"""
    c_type = preset["type"]

    if c_type == "sum":
        constraints_list.append({
            "type": "sum",
            "active": True,
            "description": f"合計 = {preset.get('default_target', 100.0)}",
            "params": {
                "columns": feature_columns[:5] if feature_columns else [],
                "target": preset.get("default_target", 100.0),
                "tolerance": preset.get("default_tolerance", 0.01),
            },
        })
    elif c_type == "range":
        col = feature_columns[0] if feature_columns else "X"
        constraints_list.append({
            "type": "range",
            "active": True,
            "description": f"{col}: 0.0 ≤ x ≤ 100.0",
            "params": {"column": col, "lo": 0.0, "hi": 100.0},
        })
    elif c_type == "inequality":
        constraints_list.append({
            "type": "inequality",
            "active": True,
            "description": "線形不等式（編集してください）",
            "params": {"expression": "", "rhs": 0.0, "operator": "le"},
        })
    elif c_type == "at_least_n":
        constraints_list.append({
            "type": "at_least_n",
            "active": True,
            "description": f"少なくとも1変数 > 0",
            "params": {
                "columns": feature_columns[:3] if feature_columns else [],
                "min_count": 1,
                "threshold": 0.0,
            },
        })
    elif c_type == "custom":
        constraints_list.append({
            "type": "custom",
            "active": True,
            "description": "カスタム式（編集してください）",
            "params": {"expression": ""},
        })

    ui.notify(f"✅ {preset['label']} を追加しました", type="positive", timeout=2000)


def _get_feature_columns(state: dict) -> list[str]:
    """利用可能な特徴量列名を取得する。"""
    # 解析結果から取得
    ar = state.get("automl_result")
    if ar is not None:
        proc_X = getattr(ar, "processed_X", None)
        if proc_X is not None and hasattr(proc_X, "columns"):
            return list(proc_X.columns)

    # 設定から取得
    cfg = state.get("pipeline_cfg", {})
    features = cfg.get("features", [])
    if features:
        return features

    # DFから取得
    df = state.get("df")
    target = state.get("target_col", "")
    if df is not None and hasattr(df, "columns"):
        return [c for c in df.columns if c != target]

    return []


def build_constraints_from_state(state: dict) -> list:
    """state["_inv"]["constraints_list"] から
    backend/optim/constraints.py の Constraint オブジェクトを構築する。

    Returns:
        list[Constraint]
    """
    from backend.optim.constraints import (
        RangeConstraint,
        SumConstraint,
        InequalityConstraint,
        AtLeastNConstraint,
        CustomConstraint,
    )

    inv = state.get("_inv", {})
    raw_list = inv.get("constraints_list", [])
    result = []

    for c in raw_list:
        if not c.get("active", True):
            continue

        c_type = c.get("type")
        params = c.get("params", {})

        try:
            if c_type == "sum":
                result.append(SumConstraint(
                    columns=params.get("columns", []),
                    target=params.get("target", 100.0),
                    tolerance=params.get("tolerance", 0.01),
                ))
            elif c_type == "range":
                result.append(RangeConstraint(
                    column=params.get("column", ""),
                    lo=params.get("lo"),
                    hi=params.get("hi"),
                ))
            elif c_type == "inequality":
                expr = params.get("expression", "")
                if expr:
                    # 簡易パース: "2*A + B" → {A: 2.0, B: 1.0}
                    result.append(InequalityConstraint(
                        coefficients=params.get("coefficients", {}),
                        rhs=params.get("rhs", 0.0),
                        operator=params.get("operator", "le"),
                    ))
            elif c_type == "at_least_n":
                result.append(AtLeastNConstraint(
                    columns=params.get("columns", []),
                    min_count=params.get("min_count", 1),
                    threshold=params.get("threshold", 0.0),
                ))
            elif c_type == "custom":
                expr = params.get("expression", "")
                if expr:
                    result.append(CustomConstraint(expression=expr))
        except Exception:
            continue

    return result
