# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/settings_checker.py

設定整合性チェッカー — リアルタイム設定矛盾検出パネル

設定タブの先頭に常設し、以下の矛盾を検知して警告を表示:
- グループ系CV選択 → グループ列未設定
- 分類専用CV選択 → 回帰タスク
- 時系列CV + シャッフル前処理
- モデル未選択
- データ未読込で解析開始しようとしている
- 記述子未選択
"""
from __future__ import annotations

from nicegui import ui
from typing import Any


def _check_conflicts(state: dict) -> list[dict]:
    """
    stateを検査し矛盾リストを返す。

    Returns:
        list of { "level": "error"|"warning"|"ok", "title": str, "msg": str }
    """
    issues = []

    df = state.get("df")
    target_col = state.get("target_col", "")
    task_type = state.get("task_type", "regression")
    cv_key = state.get("cv_key", "auto")
    group_col = state.get("group_col", "")
    selected_models = state.get("selected_models", [])
    selected_descs = state.get("selected_descriptors", [])
    smiles_col = state.get("smiles_col", "")
    precalc_done = state.get("precalc_done", False)

    # ── データ関連 ──
    if df is None:
        issues.append({
            "level": "error",
            "title": "データ未読込",
            "msg": "データを読み込んでください（解析設定タブ）",
        })
    elif not target_col:
        issues.append({
            "level": "error",
            "title": "目的変数未設定",
            "msg": "「列の役割」タブで目的変数を指定してください",
        })

    # ── SMILES/記述子関連 ──
    if smiles_col and df is not None and not precalc_done:
        issues.append({
            "level": "warning",
            "title": "SMILES記述子未計算",
            "msg": "計算中またはSMILES特徴量が未計算です。解析前に完了を確認してください",
        })

    if smiles_col and precalc_done and not selected_descs:
        issues.append({
            "level": "warning",
            "title": "記述子未選択",
            "msg": "SMILES特徴量が計算済みですが記述子が1つも選択されていません。「SMILES特徴量」タブで選択してください",
        })

    # ── CV×グループ関連 ──
    GROUP_REQUIRING_CVS = {
        "group_kfold", "stratified_group_kfold", "logo", "lpgo", "group_shuffle_split"
    }
    if cv_key in GROUP_REQUIRING_CVS and not group_col:
        issues.append({
            "level": "error",
            "title": f"CV矛盾: {cv_key} にはグループ列が必要",
            "msg": "「列の役割」タブでグループ列を指定してください。未設定の場合は解析時にKFoldへ自動フォールバックします",
        })

    # ── CV×タスクタイプ関連 ──
    CLASSIFICATION_ONLY_CVS = {
        "stratified_kfold", "stratified_group_kfold",
        "repeated_stratified_kfold", "stratified_shuffle_split"
    }
    if cv_key in CLASSIFICATION_ONLY_CVS and task_type == "regression":
        issues.append({
            "level": "warning",
            "title": f"CV×タスク矛盾: {cv_key} は分類向けです",
            "msg": "現在のタスクは「回帰」ですが分類専用CVが選択されています。KFoldへ自動切替します",
        })

    # ── モデル未選択 ──
    if df is not None and not selected_models:
        issues.append({
            "level": "error",
            "title": "モデル未選択",
            "msg": "推定器が1つも選択されていません。「パイプライン設定」→「推定器」タブで選択してください",
        })

    # ── 組み合わせ数が過大 ──
    n_imp = max(1, len(state.get("_pg_num_imputers", ["mean"])))
    n_scl = max(1, len(state.get("_pg_num_scalers", ["standard"])))
    n_ci = max(1, len(state.get("_pg_cat_imputers", ["most_frequent"])))
    n_le = max(1, len(state.get("_pg_low_encoders", ["onehot"])))
    n_bi = max(1, len(state.get("_pg_bin_imputers", ["most_frequent"])))
    n_eng = max(1, len(state.get("_pg_engineer", ["none"])))
    n_sel = max(1, len(state.get("_pg_selectors", ["none"])))
    n_est = max(1, len(selected_models)) if selected_models else 1
    n_total = n_imp * n_scl * n_ci * n_le * n_bi * n_eng * n_sel * n_est
    if n_total > 500:
        issues.append({
            "level": "warning",
            "title": f"パイプライン組み合わせ数が多い ({n_total:,}通り)",
            "msg": "処理時間が非常に長くなる可能性があります。各ステップの選択数を減らすか、タイムアウトを延長してください",
        })

    # ── 全て問題なし ──
    if not issues:
        issues.append({
            "level": "ok",
            "title": "設定に矛盾はありません ✅",
            "msg": f"タスク: {task_type} / CV: {cv_key} / モデル: {len(selected_models)}個",
        })

    return issues


def render_settings_checker(state: dict) -> None:
    """
    設定整合性チェッカーパネルを描画する。
    設定タブの先頭に配置して常時確認できるようにする。
    """
    issues = _check_conflicts(state)

    # 全体の深刻度
    has_error = any(i["level"] == "error" for i in issues)
    has_warning = any(i["level"] == "warning" for i in issues)
    all_ok = all(i["level"] == "ok" for i in issues)

    if all_ok:
        border_color = "rgba(0,200,100,0.4)"
        header_color = "green"
        header_icon = "check_circle"
    elif has_error:
        border_color = "rgba(244,67,54,0.4)"
        header_color = "red"
        header_icon = "error"
    else:
        border_color = "rgba(255,193,7,0.4)"
        header_color = "amber"
        header_icon = "warning"

    with ui.card().classes("full-width q-pa-sm q-mb-md").style(
        f"border: 1px solid {border_color}; border-radius: 10px;"
        "background: rgba(0,15,30,0.4);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
            ui.icon(header_icon, color=header_color).classes("text-h5")
            ui.label("設定整合性チェッカー").classes("text-subtitle2 text-bold")
            # 再チェックボタン
            ui.button(
                "再チェック",
                on_click=lambda: _rerender_checker(state, checker_container),
            ).props("flat dense size=xs no-caps color=cyan")

        checker_container = ui.column().classes("full-width")
        _fill_checker(issues, checker_container)


def _fill_checker(issues: list[dict], container) -> None:
    """チェッカーコンテンツを描画する。"""
    container.clear()
    with container:
        for issue in issues:
            level = issue["level"]
            title = issue["title"]
            msg = issue["msg"]

            if level == "ok":
                icon, color = "check_circle", "green"
            elif level == "error":
                icon, color = "error", "red"
            else:
                icon, color = "warning", "amber"

            with ui.row().classes("items-start q-gutter-xs q-mb-xs full-width"):
                ui.icon(icon, color=color).classes("text-body1").style("margin-top:2px;")
                with ui.column().classes("q-gutter-none"):
                    ui.label(title).classes(f"text-body2 text-bold text-{color}")
                    ui.label(msg).classes("text-caption text-grey").style("font-size:0.78rem;")


def _rerender_checker(state: dict, container) -> None:
    """再チェックして描画を更新する。"""
    issues = _check_conflicts(state)
    _fill_checker(issues, container)
