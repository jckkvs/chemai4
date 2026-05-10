"""
frontend_nicegui/components/pipeline_matrix.py

特徴量セット × パイプライン設定のマトリクス可視化コンポーネント。
"""
from __future__ import annotations

from typing import Any

from nicegui import ui


def render_pipeline_matrix(state: dict[str, Any]) -> None:
    """特徴量セット × パイプライン設定の状態をカラーコードで可視化する。

    各セットの状態:
        🟢 通常パイプライン（記述子数 ≤ サンプル数）
        🟡 RandomProjection 適用（高次元）
        🔴 未設定
        🔵 カスタム設定あり
    """
    feature_sets = state.get("descriptor_sets", {})
    pipeline_configs = state.get("pipeline_configs", {})
    n_samples = state.get("n_samples", 0)

    if not feature_sets:
        ui.label("特徴量セットが定義されていません").classes("text-grey-5")
        return

    ui.label("📋 特徴量セット × パイプライン 設定状況").classes(
        "text-subtitle1 text-bold q-mb-md"
    )

    for set_name, set_info in feature_sets.items():
        if isinstance(set_info, dict):
            desc_list = set_info.get("descriptors", [])
            n_desc = len(desc_list)
        elif isinstance(set_info, (list, set)):
            n_desc = len(set_info)
        else:
            n_desc = 0

        pipe_cfg = pipeline_configs.get(set_name, {})
        use_rp = pipe_cfg.get("use_random_projection", False)
        is_custom = bool(pipe_cfg.get("custom_steps"))

        # 状態判定
        ratio = n_desc / max(n_samples, 1) if n_samples > 0 else 0
        if not pipe_cfg:
            status = "未設定"
            color = "rgba(248,113,113,0.15)"
            border_color = "rgba(248,113,113,0.5)"
            icon = "🔴"
            recommendation = "パイプラインを設定してください"
        elif is_custom:
            status = "カスタム"
            color = "rgba(59,130,246,0.15)"
            border_color = "rgba(59,130,246,0.5)"
            icon = "🔵"
            recommendation = "カスタムパイプラインが設定済み"
        elif use_rp or ratio > 5.0:
            status = "RandomProjection"
            color = "rgba(250,204,21,0.15)"
            border_color = "rgba(250,204,21,0.5)"
            icon = "🟡"
            if not use_rp and ratio > 5.0:
                recommendation = (
                    f"⚠️ 記述子/サンプル比={ratio:.1f} → "
                    f"RandomProjection推奨"
                )
            else:
                recommendation = "高次元対応パイプライン適用中"
        else:
            status = "通常"
            color = "rgba(74,222,128,0.15)"
            border_color = "rgba(74,222,128,0.5)"
            icon = "🟢"
            recommendation = "標準パイプライン"

        with ui.card().classes("full-width q-pa-sm q-mb-xs").style(
            f"background: {color}; border: 1px solid {border_color}; "
            f"border-radius: 8px;"
        ):
            with ui.row().classes("items-center full-width justify-between"):
                with ui.row().classes("items-center q-gutter-sm"):
                    ui.label(icon).style("font-size: 1.2rem;")
                    ui.label(set_name).classes("text-subtitle2 text-bold")
                    ui.badge(status, color="grey").props("dense outline")

                with ui.row().classes("q-gutter-md text-caption text-grey"):
                    ui.label(f"📊 {n_desc}記述子")
                    if n_samples > 0:
                        ui.label(f"📏 比率: {ratio:.1f}")
                    else:
                        ui.label("📏 比率: —")

            if recommendation:
                ui.label(recommendation).classes(
                    "text-caption text-grey-5 q-mt-xs"
                )
