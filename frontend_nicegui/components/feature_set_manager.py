# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/feature_set_manager.py

特徴量セット × パイプライン 実行マトリクス UI。

ユーザーが複数の「特徴量セット」を定義し、各セットに
 - 通常パイプライン（標準前処理）
 - JL-RP高次元パイプライン（ランダム射影で次元削減）
を個別に割り当てて解析できる。

state["feature_sets"] の構造:
    [
        {
            "id": "fs_0",
            "name": "RDKit基本",
            "descriptors": ["MolWt", ...],  # 記述子名リスト
            "pipeline": "normal",           # "normal" | "highdim"
            "rp_eps": 0.1,                  # JL-RPの歪み許容誤差
        },
        ...
    ]
"""
from __future__ import annotations

import uuid
from typing import Any

from nicegui import ui

try:
    from sklearn.random_projection import johnson_lindenstrauss_min_dim
    _JL_AVAILABLE = True
except ImportError:
    _JL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_OPTIONS = {
    "normal":  "🔷 通常パイプライン",
    "highdim": "✨ JL-RP高次元パイプライン",
}

_CARD_STYLE = "border:1px solid rgba(99,102,241,0.25); border-radius:10px; background:rgba(0,15,30,0.3);"
_HIGHDIM_CARD_STYLE = "border:1px solid rgba(168,85,247,0.4); border-radius:10px; background:rgba(20,0,40,0.35);"


# ─────────────────────────────────────────────────────────────────────────────
# JL 自動判定ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def _jl_preview(n_features: int, n_samples: int, eps: float) -> dict:
    """JL条件判定と削減量の計算。"""
    if not _JL_AVAILABLE or n_samples <= 0 or n_features <= 0:
        return {"should_apply": False, "jl_min": 0, "reduction_pct": 0}
    try:
        jl_min = int(johnson_lindenstrauss_min_dim(n_samples, eps=eps))
        should = n_features > jl_min
        pct = (1 - jl_min / n_features) * 100 if should and n_features > 0 else 0
        return {"should_apply": should, "jl_min": jl_min, "reduction_pct": pct}
    except Exception:
        return {"should_apply": False, "jl_min": 0, "reduction_pct": 0}


# ─────────────────────────────────────────────────────────────────────────────
# 特徴量セット 1件の描画
# ─────────────────────────────────────────────────────────────────────────────

def _render_feature_set_row(
    parent: ui.column,
    fs: dict,
    state: dict,
    on_delete,
    on_refresh,
) -> None:
    """特徴量セット1件のカード行を描画する。"""
    n_features = len(fs.get("descriptors", []))
    n_samples = len(state.get("df")) if state.get("df") is not None else 0
    pipeline = fs.get("pipeline", "normal")
    rp_eps = fs.get("rp_eps", 0.1)

    # JL判定プレビュー
    jl = _jl_preview(n_features, n_samples, rp_eps)

    card_style = _HIGHDIM_CARD_STYLE if pipeline == "highdim" else _CARD_STYLE

    with parent:
        with ui.card().classes("full-width q-pa-sm q-mb-xs").style(card_style):
            with ui.row().classes("items-center full-width q-gutter-sm"):

                # ── インデックスバッジ
                ui.badge(fs.get("_index", "?"), color="indigo").props("rounded")

                # ── セット名（編集可能）
                ui.input(
                    value=fs.get("name", "セット"),
                    on_change=lambda e, f=fs: f.update({"name": e.value}),
                ).props("dense borderless").style("min-width:140px; max-width:180px;").classes(
                    "text-body2 text-bold"
                )

                # ── 記述子数バッジ
                ui.badge(
                    f"{n_features}個の記述子",
                    color="teal" if n_features > 0 else "grey",
                ).props("outline")

                # ── パイプライン選択
                ui.select(
                    _PIPELINE_OPTIONS,
                    value=pipeline,
                    label="パイプライン",
                    on_change=lambda e, f=fs, pr=parent: (
                        f.update({"pipeline": e.value}),
                        on_refresh(),
                    ),
                ).props("dense outlined").style("min-width:220px;")

                # ── JL判定表示（高次元のみ）
                if pipeline == "highdim":
                    if jl["should_apply"]:
                        ui.badge(
                            f"JL: {n_features}→{jl['jl_min']}次元 ({jl['reduction_pct']:.0f}%圧縮)",
                            color="purple",
                        ).props("outline")
                    elif n_features > 0:
                        ui.badge(
                            f"JL: 圧縮不要（{n_features}≤{jl['jl_min']}）",
                            color="grey",
                        ).props("outline").tooltip("n_features ≤ jl_min_dim のためパススルー")
                    else:
                        ui.badge("記述子未選択", color="grey").props("outline")

                ui.space()

                # ── 記述子を現在の選択に更新ボタン
                def _sync_desc(f=fs):
                    cur = state.get("selected_descriptors", [])
                    f["descriptors"] = list(cur)
                    ui.notify(
                        f"「{f['name']}」を現在の選択({len(cur)}個)に更新しました",
                        type="positive",
                    )
                    on_refresh()

                ui.button(icon="sync", on_click=_sync_desc).props(
                    "flat round dense color=cyan size=sm"
                ).tooltip("現在のSMILS特徴量選択でこのセットを更新")

                # ── 削除ボタン
                ui.button(
                    icon="delete", on_click=lambda _f=fs: on_delete(_f)
                ).props("flat round dense color=red size=sm")


# ─────────────────────────────────────────────────────────────────────────────
# メインエントリ
# ─────────────────────────────────────────────────────────────────────────────

def render_feature_set_manager(state: dict) -> None:
    """
    特徴量セット × パイプライン マトリクス管理UIを描画する。

    state["feature_sets"] を読み書きする。
    特徴量セットが空の場合は現在の selected_descriptors から1セット自動生成する。
    """
    # 初期化: feature_setsが未設定なら現在の選択から1セット生成
    if "feature_sets" not in state or not state["feature_sets"]:
        cur_descs = state.get("selected_descriptors", [])
        state["feature_sets"] = [
            {
                "id": str(uuid.uuid4())[:8],
                "name": "デフォルトセット",
                "descriptors": list(cur_descs),
                "pipeline": "normal",
                "rp_eps": 0.1,
            }
        ]

    # ── ヘッダー
    with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
        ui.icon("table_chart", color="indigo").classes("text-h5")
        ui.label("特徴量セット × パイプライン マトリクス").classes("text-subtitle1 text-bold")
        ui.badge(
            f"{len(state['feature_sets'])}セット",
            color="indigo",
        ).props("outline")

    ui.label(
        "各セットに記述子の組み合わせとパイプライン種別を設定します。"
        "解析開始時に全セットを順に実行し、結果を比較します。"
    ).classes("text-caption text-grey q-mb-sm")

    # ── マトリクス本体（再描画コンテナ）
    matrix_container = ui.column().classes("full-width")

    def _refresh():
        """セット一覧を再描画する。"""
        matrix_container.clear()
        fsets = state["feature_sets"]
        if not fsets:
            with matrix_container:
                ui.label("セットが1つもありません。「＋追加」で作成してください。").classes(
                    "text-caption text-grey"
                )
            return

        with matrix_container:
            # ヘッダー行ラベル
            with ui.row().classes("q-gutter-sm q-mb-xs").style("padding-left:8px;"):
                ui.label("#").classes("text-caption text-grey").style("width:30px;")
                ui.label("セット名").classes("text-caption text-grey").style("width:160px;")
                ui.label("記述子").classes("text-caption text-grey").style("width:100px;")
                ui.label("パイプライン").classes("text-caption text-grey").style("width:220px;")
                ui.label("JL判定").classes("text-caption text-grey")

            for i, fs in enumerate(fsets):
                fs["_index"] = i + 1
                _render_feature_set_row(
                    matrix_container,
                    fs,
                    state,
                    on_delete=lambda f: (_delete_set(state, f), _refresh()),
                    on_refresh=_refresh,
                )

        # ── 実行プレビュー
        _render_run_preview(matrix_container, state)

    # ── 操作ボタン行
    with ui.row().classes("q-gutter-sm q-mb-sm"):
        def _add_set():
            cur = state.get("selected_descriptors", [])
            n = len(state["feature_sets"]) + 1
            state["feature_sets"].append({
                "id": str(uuid.uuid4())[:8],
                "name": f"特徴量セット{n}",
                "descriptors": list(cur),
                "pipeline": "normal",
                "rp_eps": 0.1,
            })
            _refresh()
            ui.notify(f"セット{n}を追加（現在の選択 {len(cur)}個）", type="positive")

        def _add_highdim_set():
            cur = state.get("selected_descriptors", [])
            n = len(state["feature_sets"]) + 1
            state["feature_sets"].append({
                "id": str(uuid.uuid4())[:8],
                "name": f"高次元セット{n}",
                "descriptors": list(cur),
                "pipeline": "highdim",
                "rp_eps": 0.1,
            })
            _refresh()
            ui.notify(f"高次元セット{n}を追加（JL-RP適用）", type="positive")

        def _clear_all():
            state["feature_sets"] = []
            _refresh()

        ui.button("＋ 通常セット追加", on_click=_add_set).props(
            "unelevated dense no-caps color=indigo size=sm"
        )
        ui.button("✨ 高次元セット追加", on_click=_add_highdim_set).props(
            "unelevated dense no-caps color=purple size=sm"
        )
        ui.button("🔄 全セットクリア", on_click=_clear_all).props(
            "flat dense no-caps color=grey size=sm"
        )

    _refresh()


def _delete_set(state: dict, fs: dict) -> None:
    """セットを削除する。"""
    state["feature_sets"] = [f for f in state["feature_sets"] if f["id"] != fs["id"]]


def _render_run_preview(parent: ui.column, state: dict) -> None:
    """解析実行マトリクスのプレビューを描画する。"""
    fsets = state.get("feature_sets", [])
    if not fsets:
        return

    n_samples = len(state.get("df")) if state.get("df") is not None else 0

    with parent:
        ui.separator().classes("q-my-sm")
        with ui.card().classes("full-width q-pa-sm").style(
            "border:1px solid rgba(0,188,212,0.2); border-radius:8px; background:rgba(0,10,25,0.4);"
        ):
            ui.label("🚀 解析実行マトリクス（プレビュー）").classes("text-caption text-bold text-cyan q-mb-xs")
            ui.label(
                f"解析開始時に以下 {len(fsets)} セット × 選択モデル数 の組み合わせを順に評価します。"
            ).classes("text-caption text-grey q-mb-xs")

            for i, fs in enumerate(fsets):
                n_feat = len(fs.get("descriptors", []))
                pipe = fs.get("pipeline", "normal")
                jl = _jl_preview(n_feat, n_samples, fs.get("rp_eps", 0.1))

                if pipe == "highdim" and jl["should_apply"]:
                    pipe_label = f"✨ JL-RP ({n_feat}→{jl['jl_min']}次元)"
                    pipe_color = "purple"
                elif pipe == "highdim":
                    pipe_label = "✨ JL-RP (passthr.)"
                    pipe_color = "grey"
                else:
                    pipe_label = "🔷 通常"
                    pipe_color = "indigo"

                with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
                    ui.badge(f"#{i+1}", color="indigo").props("rounded")
                    ui.label(fs.get("name", f"セット{i+1}")).classes("text-body2").style("min-width:140px;")
                    ui.label(f"{n_feat}個").classes("text-caption text-grey").style("min-width:60px;")
                    ui.badge(pipe_label, color=pipe_color).props("outline")
