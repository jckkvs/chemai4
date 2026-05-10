"""
frontend_nicegui/components/descriptor_help_page.py

推奨記述子データベースのヘルプページ。
全目的変数・推奨記述子・ライブラリ・意味・論文を一覧表示する。
"""
from __future__ import annotations

from nicegui import ui


def render_descriptor_help() -> None:
    """推奨記述子の全データを一覧表示するヘルプページ。"""
    try:
        from backend.chem.recommender import (
            get_all_target_recommendations,
            get_target_categories,
            get_targets_by_category,
        )
    except ImportError:
        ui.label("recommender.py を読み込めません").classes("text-warning")
        return

    all_recs = get_all_target_recommendations()

    # ── ヘッダー ──
    with ui.row().classes("items-center q-gutter-sm q-mb-md"):
        ui.icon("menu_book", color="cyan").classes("text-h4")
        ui.label("推奨記述子データベース").classes("text-h5")
        ui.badge(f"{len(all_recs)}種の目的変数", color="cyan").props("outline")

    ui.label(
        "有機化学・ポリマー系の各目的変数に対して、事前知識・論文に基づき設計された"
        "推奨説明変数の一覧です。マウスオーバーで詳細を確認できます。"
    ).classes("text-body2 text-grey q-mb-md")

    # ── 初心者向けクイックスタート ──
    with ui.card().classes("full-width q-pa-md q-mb-md").style(
        "border: 2px solid rgba(251,191,36,0.5); border-radius: 12px;"
        "background: linear-gradient(135deg, rgba(50,40,0,0.3), rgba(30,25,0,0.2));"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("school", color="amber").classes("text-h5")
            ui.label("🔰 初心者ガイド: 目的変数の列を選ぶ").classes("text-h6 text-amber")

        ui.label(
            "初めてこのツールを使う方へ。SMILES記述子を活用した機械学習は"
            "以下の3ステップで行います。"
        ).classes("text-body2 text-grey q-mb-sm")

        steps = [
            (
                "1️⃣",
                "目的変数（予測したい値）の列を選ぶ",
                "データの「列の役割」タブで、予測したい数値列（例: 沸点、LogP、活性値）を"
                "「🎯 目的変数」に設定します。これだけで残りは自動判定されます。",
            ),
            (
                "2️⃣",
                "推奨記述子を適用する",
                "「SMILES記述子」タブの「目的変数で選ぶ」から、予測対象に合った"
                "推奨記述子セットをワンクリックで適用します。どの記述子を使うか"
                "悩む必要はありません。",
            ),
            (
                "3️⃣",
                "解析を開始する",
                "記述子が選択されたら解析開始ボタンを押すだけ。AutoMLが最適な"
                "モデルを自動で探索・交差検証します。",
            ),
        ]

        for emoji, title, desc in steps:
            with ui.row().classes("items-start q-gutter-sm q-mb-xs"):
                ui.label(emoji).classes("text-h6").style("min-width: 35px;")
                with ui.column().classes("q-gutter-none"):
                    ui.label(title).classes("text-body1 text-bold")
                    ui.label(desc).classes("text-caption text-grey")

        ui.label(
            "💡 ヒント: 目的変数がわからない場合は、データ読み込み後に"
            "「列の役割」タブで各列の統計を確認し、数値列の中から1つ選んでください。"
        ).classes("text-caption text-amber q-mt-sm")

    # ── カテゴリ別に展開 ──
    categories = get_target_categories()
    for cat_name in categories:
        cat_recs = get_targets_by_category(cat_name)

        with ui.expansion(
            f"📂 {cat_name} ({len(cat_recs)}種)", icon="folder",
        ).classes("full-width q-mb-sm").props("default-opened"):

            for rec in cat_recs:
                with ui.card().classes("full-width q-pa-md q-mb-sm").style(
                    "border: 1px solid rgba(0,188,212,0.2); border-radius: 10px;"
                    "background: rgba(0,20,40,0.2);"
                ):
                    # 目的変数名 + サマリー
                    with ui.row().classes("items-center q-gutter-sm"):
                        ui.icon("science", color="cyan")
                        ui.label(rec.target_name).classes("text-subtitle1 text-bold")
                        ui.badge(f"{len(rec.descriptors)}記述子", color="teal").props(
                            "outline"
                        )

                    ui.label(rec.summary).classes("text-caption text-grey q-mt-xs q-mb-sm")

                    # 記述子テーブル
                    rows = [
                        {
                            "name": d.name,
                            "library": d.library,
                            "meaning": d.meaning,
                            "category": d.category,
                            "source": d.source,
                        }
                        for d in rec.descriptors
                    ]
                    columns = [
                        {"name": "name", "label": "記述子名", "field": "name", "sortable": True},
                        {"name": "library", "label": "ライブラリ", "field": "library", "sortable": True},
                        {"name": "meaning", "label": "物理化学的意味", "field": "meaning"},
                        {"name": "category", "label": "分類", "field": "category", "sortable": True},
                        {"name": "source", "label": "論文・出典", "field": "source"},
                    ]
                    ui.table(
                        columns=columns,
                        rows=rows,
                        row_key="name",
                    ).classes("full-width").props("dense flat bordered")

    # ── 全記述子フラットテーブル ──
    ui.separator().classes("q-my-md")
    with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
        ui.icon("table_view", color="amber").classes("text-h5")
        ui.label("全記述子フラット一覧").classes("text-h6")

    # 全記述子を重複なく集計
    seen = set()
    all_rows = []
    for rec in all_recs:
        for d in rec.descriptors:
            key = f"{d.name}_{d.library}"
            if key not in seen:
                seen.add(key)
                # この記述子が関連する目的変数リスト
                related_targets = [
                    r.target_name for r in all_recs
                    if any(dd.name == d.name for dd in r.descriptors)
                ]
                all_rows.append({
                    "name": d.name,
                    "library": d.library,
                    "meaning": d.meaning,
                    "category": d.category,
                    "source": d.source,
                    "targets": ", ".join(related_targets[:3]) + (
                        f" +{len(related_targets) - 3}" if len(related_targets) > 3 else ""
                    ),
                })

    ui.badge(f"{len(all_rows)}種類の記述子", color="cyan").props("outline")

    flat_columns = [
        {"name": "name", "label": "記述子名", "field": "name", "sortable": True},
        {"name": "library", "label": "ライブラリ", "field": "library", "sortable": True},
        {"name": "meaning", "label": "物理化学的意味", "field": "meaning"},
        {"name": "category", "label": "分類", "field": "category", "sortable": True},
        {"name": "source", "label": "論文・出典", "field": "source"},
        {"name": "targets", "label": "関連目的変数", "field": "targets"},
    ]
    ui.table(
        columns=flat_columns,
        rows=all_rows,
        row_key="name",
        pagination={"rowsPerPage": 50, "sortBy": "library"},
    ).classes("full-width").props("dense flat bordered")
