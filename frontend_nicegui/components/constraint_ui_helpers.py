"""
frontend_nicegui/components/constraint_ui_helpers.py

逆解析の制約設定UIで使うヘルパー群。

- UI状態 → backend Constraint オブジェクトへの変換
- 制約の競合検出・充足率計算
- テンプレート（JSON）の保存/読込
- エラーメッセージの日本語変換

既存バックエンド (constraints.py, inverse_optimizer.py, search_space.py) は
一切変更しない。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.optim.constraints import (
    Constraint,
    RangeConstraint,
    SumConstraint,
    InequalityConstraint,
    AtLeastNConstraint,
    CustomConstraint,
    apply_constraints,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 定数
# ═══════════════════════════════════════════════════════════
CONSTRAINT_TYPES = [
    {
        "key": "range",
        "label": "📏 範囲制約",
        "desc": "個別変数の上限・下限を設定",
        "icon": "straighten",
        "color": "green",
        "simple": True,
    },
    {
        "key": "sum",
        "label": "➕ 合計制約",
        "desc": "変数グループの合計値を固定（例: 配合比率=100）",
        "icon": "functions",
        "color": "blue",
        "simple": True,
    },
    {
        "key": "ratio",
        "label": "⚖️ 比率制約",
        "desc": "変数間の比率関係（例: A ≥ 2×B）",
        "icon": "balance",
        "color": "purple",
        "simple": False,
    },
    {
        "key": "exclusion",
        "label": "🚫 排他制約",
        "desc": "変数グループから最大/最小N個のみ使用",
        "icon": "block",
        "color": "orange",
        "simple": False,
    },
    {
        "key": "conditional",
        "label": "🔀 条件付き制約",
        "desc": "IF-THEN形式の条件分岐制約",
        "icon": "call_split",
        "color": "teal",
        "simple": False,
    },
    {
        "key": "formula",
        "label": "📐 数式制約",
        "desc": "Python式による任意の数式制約",
        "icon": "calculate",
        "color": "red",
        "simple": False,
    },
]

SUM_PRESETS = [
    {"label": "配合比率 (合計 100)", "target": 100.0, "tolerance": 0.1},
    {"label": "予算配分 (合計 1)", "target": 1.0, "tolerance": 0.001},
    {"label": "確率 (合計 1)", "target": 1.0, "tolerance": 1e-6},
]

RATIO_OPERATORS = {
    "ge": {"label": "以上 (≥)", "symbol": "≥"},
    "le": {"label": "以下 (≤)", "symbol": "≤"},
    "gt": {"label": "超過 (>)", "symbol": ">"},
    "lt": {"label": "未満 (<)", "symbol": "<"},
}

ERROR_MESSAGES: dict[str, dict[str, str]] = {
    "ConstraintConflictError": {
        "message": "制約同士が矛盾しています",
        "action": "矛盾している制約を確認し、どちらかを緩和してください",
    },
    "InfeasibleRegionError": {
        "message": "制約を満たす解が存在しません",
        "action": "制約を緩和するか、変数の範囲を広げてください",
    },
    "VariableNotFoundError": {
        "message": "指定された変数が見つかりません",
        "action": "変数名が正しいか確認してください",
    },
    "ExpressionSyntaxError": {
        "message": "数式の形式が正しくありません",
        "action": "赤く表示されている箇所を修正してください",
    },
    "NumericalInstabilityError": {
        "message": "計算が安定しません",
        "action": "数値の範囲を小さくするか、制約を簡略化してください",
    },
}


# ═══════════════════════════════════════════════════════════
# UI制約 → バックエンド Constraint 変換
# ═══════════════════════════════════════════════════════════
def ui_constraints_to_backend(
    constraint_items: list[dict[str, Any]],
) -> list[Constraint]:
    """UI制約リストをバックエンドのConstraintオブジェクトに変換。

    Args:
        constraint_items: UIで設定された制約のリスト。各要素は
            {"type": "range"/"sum"/..., ...タイプ固有のフィールド}

    Returns:
        backend Constraint オブジェクトのリスト
    """
    result: list[Constraint] = []

    for item in constraint_items:
        ctype = item.get("type", "")
        try:
            if ctype == "range":
                result.append(_convert_range(item))
            elif ctype == "sum":
                result.append(_convert_sum(item))
            elif ctype == "ratio":
                result.append(_convert_ratio(item))
            elif ctype == "exclusion":
                result.append(_convert_exclusion(item))
            elif ctype == "conditional":
                result.append(_convert_conditional(item))
            elif ctype == "formula":
                result.append(_convert_formula(item))
        except Exception as e:
            logger.warning(f"制約変換エラー ({ctype}): {e}")

    return result


def _convert_range(item: dict) -> RangeConstraint:
    return RangeConstraint(
        column=item["column"],
        lo=item.get("lo"),
        hi=item.get("hi"),
    )


def _convert_sum(item: dict) -> SumConstraint:
    return SumConstraint(
        columns=item["columns"],
        target=item.get("target", 100.0),
        tolerance=item.get("tolerance", 1e-6),
    )


def _convert_ratio(item: dict) -> InequalityConstraint:
    """比率制約: target_var ≥ ratio * base_var → target_var - ratio*base_var ≥ 0."""
    target_var = item["target_var"]
    base_var = item["base_var"]
    ratio = item.get("ratio", 1.0)
    operator = item.get("operator", "ge")

    coefficients = {target_var: 1.0, base_var: -ratio}
    return InequalityConstraint(
        coefficients=coefficients,
        rhs=0.0,
        operator=operator,
    )


def _convert_exclusion(item: dict) -> AtLeastNConstraint:
    columns = item["columns"]
    mode = item.get("mode", "exactly_one")
    threshold = item.get("threshold", 0.01)

    if mode == "exactly_one":
        min_count = 1
    elif mode == "at_most_one":
        min_count = 0
    else:  # at_least_one
        min_count = 1

    return AtLeastNConstraint(
        columns=columns,
        min_count=min_count,
        threshold=threshold,
        label=item.get("group_name", ""),
    )


def _convert_conditional(item: dict) -> CustomConstraint:
    """条件付き制約をPython式に変換。"""
    if_var = item["if_var"]
    if_op = item.get("if_op", ">")
    if_val = item.get("if_val", 0)
    then_var = item["then_var"]
    then_op = item.get("then_op", ">=")
    then_val = item.get("then_val", 0)

    expr = f"({if_var} {if_op} {if_val}) == False or ({then_var} {then_op} {then_val})"
    return CustomConstraint(expression=expr)


def _convert_formula(item: dict) -> CustomConstraint:
    expression = item.get("expression", "True")
    operator = item.get("operator", "<=")
    rhs = item.get("rhs", 0)

    full_expr = f"({expression}) {operator} {rhs}"
    return CustomConstraint(expression=full_expr)


# ═══════════════════════════════════════════════════════════
# 制約検証
# ═══════════════════════════════════════════════════════════
def validate_constraints(
    constraint_items: list[dict[str, Any]],
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """制約の妥当性を検証。

    Returns:
        {
            "conflicts": [...],        # 競合する制約ペア
            "satisfaction": {...},      # 各制約の充足率
            "overall_rate": float,      # 全制約充足行の割合
            "n_satisfied": int,
            "n_total": int,
        }
    """
    result: dict[str, Any] = {
        "conflicts": [],
        "satisfaction": {},
        "overall_rate": 1.0,
        "n_satisfied": 0,
        "n_total": 0,
    }

    # 競合検出（範囲制約の矛盾チェック）
    range_constraints = [c for c in constraint_items if c.get("type") == "range"]
    for i, a in enumerate(range_constraints):
        for b in range_constraints[i + 1:]:
            if a.get("column") == b.get("column"):
                # 同一変数に複数の範囲制約 → 交差チェック
                a_lo = a.get("lo", -np.inf) if a.get("lo") is not None else -np.inf
                a_hi = a.get("hi", np.inf) if a.get("hi") is not None else np.inf
                b_lo = b.get("lo", -np.inf) if b.get("lo") is not None else -np.inf
                b_hi = b.get("hi", np.inf) if b.get("hi") is not None else np.inf
                if a_lo > b_hi or b_lo > a_hi:
                    result["conflicts"].append({
                        "a": a.get("column"),
                        "detail": f"{a.get('column')}: [{a_lo}, {a_hi}] と [{b_lo}, {b_hi}] は共存不可",
                    })

    # データ充足率
    if df is not None and len(df) > 0:
        backend_constraints = ui_constraints_to_backend(constraint_items)
        if backend_constraints:
            try:
                _, stats = apply_constraints(df, backend_constraints)
                result["n_total"] = stats["before"]
                result["n_satisfied"] = stats["after"]
                result["overall_rate"] = (
                    stats["after"] / stats["before"] if stats["before"] > 0 else 0.0
                )
                for detail in stats.get("details", []):
                    result["satisfaction"][detail["constraint"]] = detail["removed"]
            except Exception as e:
                logger.warning(f"制約充足率計算エラー: {e}")

    return result


# ═══════════════════════════════════════════════════════════
# 自然言語プレビュー
# ═══════════════════════════════════════════════════════════
def describe_constraint(item: dict[str, Any]) -> str:
    """制約を人間が読める日本語文で返す。"""
    ctype = item.get("type", "")

    if ctype == "range":
        col = item.get("column", "?")
        lo = item.get("lo")
        hi = item.get("hi")
        if lo is not None and hi is not None:
            return f"{col} は {lo} 以上 {hi} 以下"
        elif lo is not None:
            return f"{col} は {lo} 以上"
        elif hi is not None:
            return f"{col} は {hi} 以下"
        return f"{col}: 制約なし"

    elif ctype == "sum":
        cols = item.get("columns", [])
        target = item.get("target", 100)
        tol = item.get("tolerance", 0)
        cols_str = " + ".join(cols[:5])
        if len(cols) > 5:
            cols_str += f" + ... ({len(cols)}変数)"
        if tol > 0.001:
            return f"{cols_str} = {target} (±{tol})"
        return f"{cols_str} = {target}"

    elif ctype == "ratio":
        target = item.get("target_var", "?")
        base = item.get("base_var", "?")
        ratio = item.get("ratio", 1.0)
        op_sym = RATIO_OPERATORS.get(item.get("operator", "ge"), {}).get("symbol", "≥")
        return f"{target} {op_sym} {ratio} × {base}"

    elif ctype == "exclusion":
        cols = item.get("columns", [])
        mode = item.get("mode", "exactly_one")
        name = item.get("group_name", "グループ")
        mode_text = {
            "exactly_one": "ちょうど1つのみ使用",
            "at_most_one": "最大1つまで使用",
            "at_least_one": "最小1つは使用",
        }.get(mode, mode)
        return f"[{name}] {', '.join(cols[:3])}... → {mode_text}"

    elif ctype == "conditional":
        if_var = item.get("if_var", "?")
        if_op = item.get("if_op", ">")
        if_val = item.get("if_val", 0)
        then_var = item.get("then_var", "?")
        then_op = item.get("then_op", ">=")
        then_val = item.get("then_val", 0)
        return f"IF {if_var} {if_op} {if_val} THEN {then_var} {then_op} {then_val}"

    elif ctype == "formula":
        expr = item.get("expression", "?")
        op = item.get("operator", "<=")
        rhs = item.get("rhs", 0)
        return f"{expr} {op} {rhs}"

    return str(item)


# ═══════════════════════════════════════════════════════════
# テンプレート IO
# ═══════════════════════════════════════════════════════════
TEMPLATE_DIR = Path.home() / ".chemai" / "constraint_templates"


def save_template(
    name: str,
    constraint_items: list[dict[str, Any]],
    tags: list[str] | None = None,
) -> Path:
    """制約テンプレートをJSONとして保存。"""
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = TEMPLATE_DIR / f"{safe_name}.json"

    data = {
        "name": name,
        "tags": tags or [],
        "constraints": constraint_items,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"制約テンプレート保存: {path}")
    return path


def load_template(path: Path) -> dict[str, Any]:
    """制約テンプレートを読み込み。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def list_templates() -> list[dict[str, Any]]:
    """保存済みテンプレート一覧を返す。"""
    if not TEMPLATE_DIR.exists():
        return []
    templates = []
    for p in TEMPLATE_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            templates.append({
                "name": data.get("name", p.stem),
                "tags": data.get("tags", []),
                "n_constraints": len(data.get("constraints", [])),
                "path": str(p),
            })
        except Exception:
            pass
    return templates


def delete_template(path: Path) -> None:
    """テンプレートを削除。"""
    if path.exists():
        path.unlink()


# ═══════════════════════════════════════════════════════════
# エラーメッセージ変換
# ═══════════════════════════════════════════════════════════
def friendly_error(error: Exception) -> dict[str, str]:
    """技術的エラーをユーザーフレンドリーなメッセージに変換。"""
    error_type = type(error).__name__
    if error_type in ERROR_MESSAGES:
        return ERROR_MESSAGES[error_type]

    error_str = str(error)
    # 部分一致で検出
    for key, msg in ERROR_MESSAGES.items():
        if key.lower() in error_str.lower():
            return msg

    return {
        "message": f"予期しないエラーが発生しました: {error_str}",
        "action": "設定を確認してもう一度お試しください。問題が続く場合は開発者にお問い合わせください。",
    }


# ═══════════════════════════════════════════════════════════
# データ統計ヘルパー
# ═══════════════════════════════════════════════════════════
def get_column_stats(df: pd.DataFrame, col: str) -> dict[str, float]:
    """列の基本統計量を返す。"""
    s = df[col].dropna()
    if len(s) == 0:
        return {"min": 0, "max": 1, "q1": 0, "median": 0.5, "q3": 1, "mean": 0.5, "std": 0}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "q1": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q3": float(s.quantile(0.75)),
        "mean": float(s.mean()),
        "std": float(s.std()),
    }
"""Complexity: 7, Description: 制約UI→バックエンド変換、競合検出、充足率計算、テンプレートJSON IO、エラー日本語変換、統計ヘルパーを提供。バックエンドは一切変更しない。"""
