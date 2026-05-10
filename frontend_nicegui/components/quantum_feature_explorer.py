"""
frontend_nicegui/components/quantum_feature_explorer.py

量子化学特徴量貢献度エクスプローラー — スタンドアロンUIコンポーネント。

以下を一画面で提供する:
1. 量子特徴量カテゴリ別グループ表示（HOMO/LUMO/電荷/反応性/3D/アンサンブル）
2. 加重方法の確認と分布プロット（重量比 vs モル比 どちらで計算したか）
3. xTBML特徴量のPairplot（相関マトリクス）
4. 信頼度スコアとのクロス分析
5. 混合物特徴量の成分寄与度ウォーターフォール

既存UIへの影響: なし（完全新規コンポーネント）
使い方: render_quantum_feature_explorer(state) を新規タブパネルで呼び出す。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)


# ── 量子特徴量カテゴリ定義 ──
_QUANTUM_CATEGORIES = {
    "⚡ 軌道エネルギー": [
        "xtb_HomoEnergy", "xtb_LumoEnergy", "xtb_HomoLumoGap",
        "xtb_ml_HomoEnergy", "xtb_ml_LumoEnergy", "xtb_ml_Gap",
        "xtb_ml_TotalEnergy", "xtb_TotalEnergy",
    ],
    "🎯 反応性指標": [
        "xtb_ml_Hardness", "xtb_ml_Softness", "xtb_ml_Electrophilicity",
    ],
    "🔋 電荷・双極子": [
        "xtb_DipoleMoment", "xtb_Polarizability",
        "xtb_MullikenChargeMax", "xtb_MullikenChargeMin",
        "xtb_MullikenChargeMean", "xtb_MullikenChargeStd",
        "xtb_ml_Dipole",
    ],
    "🌀 3D幾何": [
        "xtb_ml_3D_MaxDistance", "xtb_ml_3D_Asphericity",
        "xtb_ml_3D_Eccentricity", "xtb_ml_3D_InertiaX",
        "xtb_ml_3D_InertiaY", "xtb_ml_3D_InertiaZ",
        "xtb_ml_3D_RadiusOfGyration",
    ],
    "🎲 アンサンブル統計": [
        "ens_energy_mean", "ens_energy_std", "ens_homo_mean",
        "ens_homo_std", "ens_gap_mean", "ens_gap_std",
        "ens_dipole_mean", "ens_dipole_std",
        "ens_boltzmann_entropy", "ens_n_conformers",
    ],
    "🔍 信頼度スコア": [
        "conf_convergence", "conf_electronic_stability",
        "conf_charge_consistency", "conf_descriptor_completeness",
        "conf_conformer_repr", "conf_overall",
    ],
}


def render_quantum_feature_explorer(state: dict[str, Any]) -> None:
    """量子化学特徴量貢献度エクスプローラーを描画する。"""

    # ── ヘッダー ──
    with ui.card().classes("w-full").style(
        "background: rgba(123, 47, 247, 0.08); "
        "border: 1px solid rgba(123, 47, 247, 0.25); "
        "border-radius: 16px; padding: 24px;"
    ):
        with ui.row().classes("items-center gap-4"):
            ui.icon("hub").classes("text-3xl").style("color: #a78bfa;")
            with ui.column().classes("gap-0"):
                ui.label("🔬 量子化学特徴量エクスプローラー").classes(
                    "text-xl font-bold"
                ).style("color: #e0e0f0;")
                ui.label(
                    "xTB/ML派生特徴量の分布・相関・貢献度を多角的に可視化"
                ).classes("text-sm").style("color: #a0a0c0;")

    # ── データソース確認 ──
    df_source = _get_feature_dataframe(state)

    if df_source is None or df_source.empty:
        ui.separator().classes("q-my-md")
        with ui.card().classes("w-full").style(
            "background: rgba(251, 191, 36, 0.08); "
            "border: 1px solid rgba(251, 191, 36, 0.2); "
            "border-radius: 12px; padding: 20px;"
        ):
            ui.icon("info").style("color: #fbbf24; font-size: 2rem;")
            ui.label("量子化学特徴量データが未計算です").classes("text-lg").style(
                "color: #fbbf24;"
            )
            ui.label(
                "「⚡ 計算管理」タブで計算を実行するか、"
                "「📂 解析設定」タブで記述子を計算してください。"
            ).classes("text-sm").style("color: #a0a0c0;")
        return

    ui.separator().classes("q-my-md")

    # ── サブタブ ──
    with ui.tabs().classes("full-width").props(
        "dense active-color=purple indicator-color=purple scrollable"
    ) as qtabs:
        t_cat   = ui.tab("cat",     label="📂 カテゴリ分類")
        t_dist  = ui.tab("dist",    label="📊 分布プロット")
        t_corr  = ui.tab("corr",    label="🔗 相関マトリクス")
        t_conf  = ui.tab("conf",    label="🔍 信頼度分析")
        t_mix   = ui.tab("mix",     label="🧪 混合物寄与度")

    with ui.tab_panels(qtabs, value=t_cat).classes("full-width bg-transparent"):

        # ════════════════════════════════════════════════════
        # カテゴリ分類タブ
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_cat):
            _render_category_overview(df_source)

        # ════════════════════════════════════════════════════
        # 分布プロット
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_dist):
            _render_distribution_plots(df_source)

        # ════════════════════════════════════════════════════
        # 相関マトリクス
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_corr):
            _render_correlation_matrix(df_source)

        # ════════════════════════════════════════════════════
        # 信頼度分析
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_conf):
            _render_confidence_analysis(df_source)

        # ════════════════════════════════════════════════════
        # 混合物寄与度
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_mix):
            _render_mixture_contribution(state)


# ────────────────────────────────────────────────────────────────
# データソース取得
# ────────────────────────────────────────────────────────────────

def _get_feature_dataframe(state: dict[str, Any]) -> pd.DataFrame | None:
    """stateから量子化学特徴量DataFrameを取得する。"""
    # 1. precalc_df: 記述子計算済み
    df = state.get("precalc_df")
    if df is not None and isinstance(df, pd.DataFrame):
        quantum_cols = [
            c for c in df.columns
            if any(c.startswith(p) for p in
                   ("xtb_", "ens_", "conf_", "xtb_ml_", "mix_"))
        ]
        if quantum_cols:
            return df[quantum_cols]

    # 2. 混合物計算結果
    mixture_result = state.get("_mixture_result")
    if mixture_result is not None:
        try:
            return mixture_result.to_dataframe()
        except Exception:
            pass

    return None


# ────────────────────────────────────────────────────────────────
# カテゴリ概要
# ────────────────────────────────────────────────────────────────

def _render_category_overview(df: pd.DataFrame) -> None:
    """特徴量をカテゴリ別に分類して表示する。"""
    import plotly.graph_objects as go

    ui.label("📂 量子特徴量カテゴリ別概要").classes("text-lg font-bold q-mb-sm").style(
        "color: #e0e0f0;"
    )
    ui.label(
        f"総量子特徴量数: {len(df.columns)}列 | サンプル数: {len(df)}件"
    ).classes("text-sm q-mb-md").style("color: #a0a0c0;")

    # カテゴリ別にマッチした列を集計
    category_hits: dict[str, list[str]] = {}
    unclassified: list[str] = []

    for col in df.columns:
        matched = False
        for cat_name, feat_list in _QUANTUM_CATEGORIES.items():
            if col in feat_list:
                category_hits.setdefault(cat_name, []).append(col)
                matched = True
                break
        if not matched:
            unclassified.append(col)

    if unclassified:
        category_hits["🔮 その他"] = unclassified

    # ドーナツチャート
    labels = list(category_hits.keys())
    values = [len(v) for v in category_hits.values()]

    if labels:
        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=[
                "#7b2ff7", "#00d4ff", "#4ade80", "#fbbf24",
                "#f472b6", "#60a5fa", "#a78bfa", "#34d399",
            ]),
            textinfo="label+value",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            title="特徴量カテゴリ分布",
            legend=dict(font=dict(size=11)),
        )
        ui.plotly(fig).classes("w-full")

    # カテゴリ別詳細展開
    ui.separator().classes("q-my-md")
    for cat_name, cols in category_hits.items():
        with ui.expansion(
            f"{cat_name} ({len(cols)}列)",
            icon="category",
        ).classes("w-full q-mb-sm").style(
            "background: rgba(255,255,255,0.02); border-radius: 8px;"
        ):
            # 統計サマリ
            if cols:
                sub_df = df[[c for c in cols if c in df.columns]]
                if not sub_df.empty:
                    stats = sub_df.describe().round(4).T
                    stats_rows = [
                        {
                            "feature": idx,
                            "mean": f"{row.get('mean', 0):.4g}",
                            "std": f"{row.get('std', 0):.4g}",
                            "min": f"{row.get('min', 0):.4g}",
                            "max": f"{row.get('max', 0):.4g}",
                            "null": str(sub_df[idx].isna().sum()),
                        }
                        for idx, row in stats.iterrows()
                        if idx in sub_df.columns
                    ]
                    stat_cols = [
                        {"name": k, "label": {"feature": "特徴量名", "mean": "平均", "std": "標準偏差",
                                               "min": "最小", "max": "最大", "null": "欠損"}[k],
                         "field": k, "sortable": True}
                        for k in ["feature", "mean", "std", "min", "max", "null"]
                    ]
                    ui.table(
                        columns=stat_cols,
                        rows=stats_rows,
                    ).classes("w-full").props("dense flat")


# ────────────────────────────────────────────────────────────────
# 分布プロット
# ────────────────────────────────────────────────────────────────

def _render_distribution_plots(df: pd.DataFrame) -> None:
    """選択した特徴量のヒストグラムと箱ひげ図を表示する。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ui.label("📊 特徴量分布プロット").classes("text-lg font-bold q-mb-sm").style(
        "color: #e0e0f0;"
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        ui.label("数値特徴量がありません").style("color: #a0a0c0;")
        return

    # 特徴量選択
    feat_select = ui.select(
        label="特徴量を選択（複数可）",
        options=numeric_cols,
        value=numeric_cols[:min(4, len(numeric_cols))],
        multiple=True,
    ).classes("w-full").props("use-chips outlined")

    plot_container = ui.column().classes("w-full q-mt-md")

    def _update_plot():
        selected = feat_select.value
        if not selected:
            return
        if isinstance(selected, str):
            selected = [selected]

        plot_container.clear()
        with plot_container:
            # ヒストグラム + KDE
            n_feats = len(selected[:8])
            cols_grid = min(2, n_feats)
            rows_grid = (n_feats + 1) // 2

            fig = make_subplots(
                rows=rows_grid, cols=cols_grid,
                subplot_titles=[s[:20] for s in selected[:8]],
            )
            colors = [
                "#7b2ff7", "#00d4ff", "#4ade80", "#fbbf24",
                "#f472b6", "#60a5fa", "#a78bfa", "#34d399",
            ]
            for i, feat in enumerate(selected[:8]):
                r, c = divmod(i, cols_grid)
                vals = df[feat].dropna().values
                fig.add_trace(
                    go.Histogram(
                        x=vals, name=feat,
                        marker_color=colors[i % len(colors)],
                        opacity=0.7, showlegend=False,
                        nbinsx=30,
                    ),
                    row=r + 1, col=c + 1,
                )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.1)",
                height=max(300, 220 * rows_grid),
                margin=dict(l=10, r=10, t=40, b=10),
                title="特徴量ヒストグラム",
            )
            ui.plotly(fig).classes("w-full")

            # 箱ひげ図
            if len(selected) > 1:
                fig2 = go.Figure()
                for i, feat in enumerate(selected[:8]):
                    vals = df[feat].dropna().values
                    fig2.add_trace(go.Box(
                        y=vals, name=feat[:20],
                        marker_color=colors[i % len(colors)],
                        boxmean="sd",
                    ))
                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=350,
                    margin=dict(l=10, r=10, t=30, b=80),
                    title="箱ひげ図（平均±標準偏差）",
                    xaxis_tickangle=-30,
                )
                ui.plotly(fig2).classes("w-full q-mt-md")

    feat_select.on_value_change(lambda _: _update_plot())
    _update_plot()


# ────────────────────────────────────────────────────────────────
# 相関マトリクス
# ────────────────────────────────────────────────────────────────

def _render_correlation_matrix(df: pd.DataFrame) -> None:
    """量子特徴量間の相関マトリクスをヒートマップで表示する。"""
    import plotly.graph_objects as go

    ui.label("🔗 量子特徴量間相関マトリクス").classes("text-lg font-bold q-mb-sm").style(
        "color: #e0e0f0;"
    )

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        ui.label("2列以上の数値特徴量が必要です").style("color: #a0a0c0;")
        return

    # カテゴリ選択
    category_options = {k: k for k in _QUANTUM_CATEGORIES.keys()}
    category_options["📋 全て"] = "all"

    cat_select = ui.select(
        label="カテゴリでフィルタ",
        options=list(category_options.keys()),
        value="📋 全て",
    ).classes("w-64")

    corr_container = ui.column().classes("w-full q-mt-md")

    def _update_corr():
        cat_val = cat_select.value
        if cat_val == "📋 全て":
            sub_df = numeric_df
        else:
            feat_list = _QUANTUM_CATEGORIES.get(cat_val, [])
            cols = [c for c in feat_list if c in numeric_df.columns]
            sub_df = numeric_df[cols] if cols else numeric_df

        if sub_df.shape[1] < 2:
            corr_container.clear()
            with corr_container:
                ui.label("選択カテゴリに2列以上の特徴量が必要です").style(
                    "color: #a0a0c0;"
                )
            return

        corr = sub_df.corr(numeric_only=True)
        n_feat = len(corr.columns)

        corr_container.clear()
        with corr_container:
            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=[c[:15] for c in corr.columns],
                y=[c[:15] for c in corr.index],
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                colorbar=dict(title="相関係数"),
                text=corr.values.round(2),
                texttemplate="%{text}",
                textfont=dict(size=max(7, min(12, 120 // n_feat))),
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=max(400, 25 * n_feat),
                margin=dict(l=10, r=10, t=40, b=60),
                title=f"相関マトリクス ({n_feat}×{n_feat})",
                xaxis_tickangle=-45,
            )
            ui.plotly(fig).classes("w-full")

            # 強相関ペア
            threshold = 0.8
            strong_pairs = []
            vals = corr.values
            cols = list(corr.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    r = vals[i, j]
                    if abs(r) >= threshold:
                        strong_pairs.append({
                            "特徴量A": cols[i],
                            "特徴量B": cols[j],
                            "相関係数": f"{r:.4f}",
                            "強さ": "🔴 強正相関" if r > 0 else "🔵 強負相関",
                        })
            if strong_pairs:
                ui.label(
                    f"⚠️ 強相関ペア (|r|≥{threshold}): {len(strong_pairs)}件"
                ).classes("text-sm q-mt-sm").style("color: #fbbf24;")
                with ui.expansion("強相関ペア一覧", icon="warning"):
                    corr_cols = [
                        {"name": k, "label": k, "field": k, "sortable": True}
                        for k in ["特徴量A", "特徴量B", "相関係数", "強さ"]
                    ]
                    ui.table(
                        columns=corr_cols, rows=strong_pairs,
                    ).classes("w-full").props("dense flat")

    cat_select.on_value_change(lambda _: _update_corr())
    _update_corr()


# ────────────────────────────────────────────────────────────────
# 信頼度分析
# ────────────────────────────────────────────────────────────────

def _render_confidence_analysis(df: pd.DataFrame) -> None:
    """信頼度スコアと他の特徴量との関係を分析する。"""
    import plotly.graph_objects as go

    ui.label("🔍 計算信頼度分析").classes("text-lg font-bold q-mb-sm").style(
        "color: #e0e0f0;"
    )

    conf_cols = [c for c in df.columns if c.startswith("conf_")]
    if not conf_cols:
        with ui.card().classes("w-full").style(
            "background: rgba(251, 191, 36, 0.05); border-radius: 12px; padding: 16px;"
        ):
            ui.label("ℹ️ 信頼度スコアが計算されていません").style("color: #fbbf24;")
            ui.label(
                "xTB計算時にuncertainty_estimatorが有効な場合に信頼度スコアが付与されます。"
            ).classes("text-sm").style("color: #a0a0c0;")
        return

    conf_df = df[conf_cols].dropna()
    if conf_df.empty:
        ui.label("信頼度データが空です").style("color: #a0a0c0;")
        return

    # 信頼度スコアバー
    means = conf_df.mean()
    colors_conf = [
        "#4ade80" if v >= 0.8 else "#fbbf24" if v >= 0.5 else "#f87171"
        for v in means.values
    ]
    fig = go.Figure(go.Bar(
        x=[c.replace("conf_", "") for c in means.index],
        y=means.values,
        marker_color=colors_conf,
        text=[f"{v:.3f}" for v in means.values],
        textposition="outside",
    ))
    fig.add_hline(y=0.8, line_dash="dash", line_color="#4ade80",
                  annotation_text="高信頼度閾値 (0.8)")
    fig.add_hline(y=0.5, line_dash="dash", line_color="#fbbf24",
                  annotation_text="中信頼度閾値 (0.5)")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 1.1]),
        title="計算信頼度スコア（平均値）",
        yaxis_title="信頼度 (0-1)",
    )
    ui.plotly(fig).classes("w-full")

    # 総合信頼度の分布
    if "conf_overall" in df.columns:
        ui.separator().classes("q-my-md")
        ui.label("📊 総合信頼度の分布").classes("text-md font-bold").style(
            "color: #e0e0f0;"
        )
        overall = df["conf_overall"].dropna()
        n_high = (overall >= 0.8).sum()
        n_mid = ((overall >= 0.5) & (overall < 0.8)).sum()
        n_low = (overall < 0.5).sum()

        with ui.row().classes("gap-4 q-mt-sm q-mb-md"):
            for count, label, color in [
                (n_high, "高信頼 (≥0.8)", "#4ade80"),
                (n_mid, "中信頼 (0.5-0.8)", "#fbbf24"),
                (n_low, "低信頼 (<0.5)", "#f87171"),
            ]:
                with ui.card().classes("q-pa-sm").style(
                    f"background: rgba(0,0,0,0.2); border-radius: 12px; min-width: 100px; "
                    f"border: 1px solid {color}40;"
                ):
                    ui.label(str(count)).classes("text-2xl font-bold").style(
                        f"color: {color};"
                    )
                    ui.label(label).classes("text-xs").style("color: #a0a0c0;")

        fig2 = go.Figure(go.Histogram(
            x=overall.values,
            nbinsx=20,
            marker_color="rgba(123, 47, 247, 0.7)",
            marker_line_color="rgba(123, 47, 247, 1)",
            marker_line_width=1,
        ))
        fig2.add_vline(x=0.8, line_dash="dash", line_color="#4ade80")
        fig2.add_vline(x=0.5, line_dash="dash", line_color="#fbbf24")
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="総合信頼度スコア",
            yaxis_title="分子数",
            title="総合信頼度の分布",
        )
        ui.plotly(fig2).classes("w-full")


# ────────────────────────────────────────────────────────────────
# 混合物成分寄与度
# ────────────────────────────────────────────────────────────────

def _render_mixture_contribution(state: dict[str, Any]) -> None:
    """混合物計算結果の成分寄与度をウォーターフォールで表示する。"""
    import plotly.graph_objects as go

    ui.label("🧪 混合物特徴量の成分寄与度").classes("text-lg font-bold q-mb-sm").style(
        "color: #e0e0f0;"
    )

    mixture_result = state.get("_mixture_result")
    if mixture_result is None:
        with ui.card().classes("w-full").style(
            "background: rgba(251, 191, 36, 0.05); border-radius: 12px; padding: 16px;"
        ):
            ui.label("ℹ️ 混合物計算が実行されていません").style("color: #fbbf24;")
            ui.label(
                "「🧪 混合物」タブで混合物特徴量を計算してください。"
            ).classes("text-sm").style("color: #a0a0c0;")
        return

    try:
        comp_features = mixture_result.component_features
        mix_features = mixture_result.mixture_features
        weighting_log = mixture_result.weighting_log
        conv_info = mixture_result.conversion_info

        if not comp_features or not mix_features:
            ui.label("成分特徴量データがありません").style("color: #a0a0c0;")
            return

        n_comp = len(comp_features)
        weight_fracs = conv_info.get("weight_fractions", [1 / n_comp] * n_comp)
        mole_fracs = conv_info.get("mole_fractions", [1 / n_comp] * n_comp)

        # 特徴量選択
        feat_options = [k.replace("mix_", "") for k in mix_features.keys()
                        if k.replace("mix_", "") in (comp_features[0] if comp_features else {})]
        if not feat_options:
            feat_options = list(mix_features.keys())[:20]

        ui.label(f"成分数: {n_comp} | 混合特徴量: {len(mix_features)}列").classes(
            "text-sm q-mb-sm"
        ).style("color: #a0a0c0;")

        feat_select = ui.select(
            label="可視化する特徴量を選択",
            options=feat_options[:50],
            value=feat_options[0] if feat_options else None,
        ).classes("w-64")

        chart_container = ui.column().classes("w-full q-mt-md")

        def _update_chart():
            sel = feat_select.value
            if not sel:
                return

            mix_key = f"mix_{sel}" if f"mix_{sel}" in mix_features else sel
            mix_val = mix_features.get(mix_key, 0.0)
            used_type = weighting_log.get(sel, "weight")

            # 各成分の値
            comp_vals = []
            for cf in comp_features:
                comp_vals.append(cf.get(sel, 0.0))

            # 使用した分率
            fracs = (
                mole_fracs if "mole" in used_type else weight_fracs
            )

            contributions = [v * f for v, f in zip(comp_vals, fracs)]

            chart_container.clear()
            with chart_container:
                # 加重方法バッジ
                color = "#00d4ff" if "mole" in used_type else "#4ade80"
                label_text = "mol比加重" if "mole" in used_type else "重量比加重"
                if "user" in used_type:
                    label_text += "（ユーザー上書き）"
                ui.badge(label_text).props(f"color={'cyan' if 'mole' in used_type else 'green'}")

                # ウォーターフォールチャート
                comp_labels = [
                    f"成分#{i+1}\n({comp_features[i].get('smiles','')[:8]})"
                    for i in range(len(comp_features))
                ]
                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    x=comp_labels + ["混合物合計"],
                    y=contributions + [mix_val],
                    measure=["relative"] * len(contributions) + ["total"],
                    text=[
                        f"寄与: {c:.4g}<br>分率: {f:.1%}"
                        for c, f in zip(contributions, fracs)
                    ] + [f"合計: {mix_val:.4g}"],
                    textposition="outside",
                    increasing=dict(marker=dict(color="rgba(74,222,128,0.75)")),
                    decreasing=dict(marker=dict(color="rgba(248,113,113,0.75)")),
                    totals=dict(marker=dict(color="rgba(123,47,247,0.85)")),
                    connector=dict(line=dict(color="rgba(255,255,255,0.2)")),
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=380,
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis_title=f"{sel}",
                    title=f"成分別寄与度: {sel}（{label_text}）",
                )
                ui.plotly(fig).classes("w-full")

                # 成分別テーブル
                tbl_rows = [
                    {
                        "成分": f"#{i+1}",
                        "特徴量値": f"{comp_vals[i]:.4g}",
                        "分率": f"{fracs[i]*100:.1f}%",
                        "寄与": f"{contributions[i]:.4g}",
                        "寄与率": f"{contributions[i]/mix_val*100:.1f}%" if abs(mix_val) > 1e-12 else "—",
                    }
                    for i in range(len(comp_features))
                ]
                tbl_rows.append({
                    "成分": "混合物",
                    "特徴量値": f"{mix_val:.4g}",
                    "分率": "—",
                    "寄与": f"{mix_val:.4g}",
                    "寄与率": "100%",
                })
                tbl_cols = [
                    {"name": k, "label": k, "field": k, "align": "center"}
                    for k in ["成分", "特徴量値", "分率", "寄与", "寄与率"]
                ]
                ui.table(columns=tbl_cols, rows=tbl_rows).classes(
                    "w-full q-mt-sm"
                ).props("dense flat bordered")

        if feat_select.value:
            _update_chart()
        feat_select.on_value_change(lambda _: _update_chart())

    except Exception as e:
        ui.label(f"❌ 寄与度分析エラー: {e}").style("color: #f87171;")
        logger.error("混合物寄与度分析エラー: %s", e, exc_info=True)
