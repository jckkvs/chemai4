"""
frontend_streamlit/pages/pipeline/inverse_analysis_page.py

MolAI逆解析・分子探索ページ。
MolAIのみで解析した場合に結果タブ内で表示される。
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any


def render_inverse_analysis(
    molai_adapter: Any,
    best_model: Any,
    training_smiles: list[str],
    target_col: str,
    df: pd.DataFrame,
):
    """逆解析UIを描画する。

    Args:
        molai_adapter: 学習済みMolAIAdapter
        best_model: 学習済みベストモデル
        training_smiles: 学習に使ったSMILESリスト
        target_col: 目的変数のカラム名
        df: 元データ
    """
    from backend.chem.molai_inverse import MolAIInverseAnalyzer

    st.markdown("### 🔬 MolAI 逆解析・分子探索")
    st.caption("潜在空間を探索し、目標物性を持つ新規分子候補を発見します。")

    # ── Analyzer初期化 ──
    if "inverse_analyzer" not in st.session_state:
        try:
            analyzer = MolAIInverseAnalyzer(molai_adapter)
            # PCA変換済みのPC座標を取得
            pc_df = st.session_state.get("precalc_smiles_df")
            if pc_df is not None:
                pc_cols = [c for c in pc_df.columns if c.startswith("molai_pc")]
                if pc_cols:
                    pc_vectors = pc_df[pc_cols].values.astype(np.float32)
                    analyzer.set_training_data(training_smiles, pc_vectors)
                    st.session_state["inverse_analyzer"] = analyzer
                    st.session_state["inverse_pc_cols"] = pc_cols
        except Exception as e:
            st.error(f"逆解析の初期化に失敗しました: {e}")
            return

    analyzer = st.session_state.get("inverse_analyzer")
    if analyzer is None:
        st.warning("MolAI解析が完了していません。先に解析を実行してください。")
        return

    pc_cols = st.session_state.get("inverse_pc_cols", [])
    n_pca = len(pc_cols)

    # 目的変数の現在の統計量
    _target_vals = df[target_col].dropna()
    _t_min, _t_max, _t_mean = float(_target_vals.min()), float(_target_vals.max()), float(_target_vals.mean())

    # ── 探索タブ ──
    tab_rand, tab_bayes, tab_interp, tab_degen = st.tabs([
        "🎲 ランダム探索",
        "🧠 ベイズ最適化",
        "🔗 分子間補間",
        "🗺️ 縮退マップ",
    ])

    # ──── ランダム探索 ────
    with tab_rand:
        _tc1, _tc2 = st.columns(2)
        with _tc1:
            _target_val = st.number_input(
                f"目標値 ({target_col})",
                value=_t_mean,
                min_value=_t_min - abs(_t_max - _t_min),
                max_value=_t_max + abs(_t_max - _t_min),
                key="inv_target_rand",
            )
        with _tc2:
            _n_cand = st.slider("候補数", 10, 200, 50, key="inv_n_rand")

        if st.button("🎲 ランダム探索を実行", key="btn_rand", type="primary", use_container_width=True):
            with st.spinner("潜在空間をランダムサンプリング中..."):
                try:
                    results = analyzer.explore_random(
                        model=best_model,
                        target_value=_target_val,
                        n_candidates=_n_cand,
                        n_samples=5000,
                    )
                    st.session_state["inv_results_rand"] = results
                except Exception as e:
                    st.error(f"探索エラー: {e}")

        _results = st.session_state.get("inv_results_rand")
        if _results:
            _render_candidates(_results, target_col, _target_val)

    # ──── ベイズ最適化 ────
    with tab_bayes:
        _tc1, _tc2 = st.columns(2)
        with _tc1:
            _target_val_bo = st.number_input(
                f"目標値 ({target_col})",
                value=_t_mean,
                key="inv_target_bo",
            )
        with _tc2:
            _n_iter = st.slider("探索反復数", 20, 300, 100, key="inv_n_iter_bo")

        if st.button("🧠 ベイズ最適化を実行", key="btn_bayes", type="primary", use_container_width=True):
            with st.spinner("ガウス過程回帰でPCA空間を最適化中..."):
                try:
                    results = analyzer.explore_bayesian(
                        model=best_model,
                        target_value=_target_val_bo,
                        n_candidates=50,
                        n_iterations=_n_iter,
                    )
                    st.session_state["inv_results_bo"] = results
                except Exception as e:
                    st.error(f"探索エラー: {e}")

        _results = st.session_state.get("inv_results_bo")
        if _results:
            _render_candidates(_results, target_col, _target_val_bo)

    # ──── 分子間補間 ────
    with tab_interp:
        st.caption("2つの分子の潜在ベクトルを線形補間し、中間構造を生成します。")
        _smi_options = training_smiles[:200] if len(training_smiles) > 200 else training_smiles
        _ic1, _ic2 = st.columns(2)
        with _ic1:
            _smi_a = st.selectbox("分子A", _smi_options, index=0, key="inv_smi_a")
        with _ic2:
            _smi_b = st.selectbox("分子B", _smi_options, index=min(1, len(_smi_options) - 1), key="inv_smi_b")

        _n_steps = st.slider("補間ステップ数", 5, 50, 20, key="inv_n_steps")

        if st.button("🔗 補間を実行", key="btn_interp", type="primary", use_container_width=True):
            with st.spinner("潜在空間を線形補間中..."):
                try:
                    results = analyzer.explore_interpolation(
                        smiles_a=_smi_a,
                        smiles_b=_smi_b,
                        n_steps=_n_steps,
                        model=best_model,
                    )
                    st.session_state["inv_results_interp"] = results
                except Exception as e:
                    st.error(f"補間エラー: {e}")

        _results = st.session_state.get("inv_results_interp")
        if _results:
            _render_interpolation(_results, _smi_a, _smi_b)

    # ──── 縮退マップ ────
    with tab_degen:
        st.caption("PCA空間の2軸をスキャンし、同じ分子にデコードされる領域を可視化します。")
        _dc1, _dc2, _dc3 = st.columns(3)
        with _dc1:
            _dim1 = st.selectbox("X軸", range(n_pca), index=0, format_func=lambda x: f"PC{x+1}", key="degen_dim1")
        with _dc2:
            _dim2 = st.selectbox("Y軸", range(n_pca), index=min(1, n_pca - 1), format_func=lambda x: f"PC{x+1}", key="degen_dim2")
        with _dc3:
            _n_grid = st.slider("グリッド解像度", 10, 50, 25, key="degen_grid")

        if st.button("🗺️ 縮退マップを計算", key="btn_degen", type="primary", use_container_width=True):
            with st.spinner("PCA空間をスキャン中..."):
                try:
                    dmap = analyzer.compute_degeneracy_map(
                        pc_dim1=_dim1,
                        pc_dim2=_dim2,
                        n_grid=_n_grid,
                    )
                    st.session_state["inv_degen_map"] = dmap
                    st.session_state["inv_degen_dims"] = (_dim1, _dim2)
                except Exception as e:
                    st.error(f"縮退マップエラー: {e}")

        _dmap = st.session_state.get("inv_degen_map")
        if _dmap:
            _render_degeneracy_map(_dmap, st.session_state.get("inv_degen_dims", (0, 1)))


# ── 表示ヘルパー ──────────────────────────────────────────────────────────

def _render_candidates(candidates: list, target_col: str, target_value: float):
    """候補分子のランキングテーブルを表示。"""
    if not candidates:
        st.info("有効な候補が見つかりませんでした。")
        return

    st.success(f"✅ {len(candidates)}件の候補を発見")

    rows = []
    for i, c in enumerate(candidates):
        rows.append({
            "順位": i + 1,
            "SMILES": c.smiles,
            "予測値": round(c.predicted_value, 4) if c.predicted_value is not None else None,
            "差分": round(abs(c.predicted_value - target_value), 4) if c.predicted_value is not None else None,
            "新規": "✅" if c.is_novel else "—",
            "類似度": round(c.similarity_to_nearest, 3),
        })

    _df = pd.DataFrame(rows)
    st.dataframe(_df, use_container_width=True, hide_index=True, height=min(500, 35 + len(rows) * 35))

    # CSV ダウンロード
    csv = _df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSVダウンロード", csv, "molai_candidates.csv", "text/csv", use_container_width=True)


def _render_interpolation(candidates: list, smi_a: str, smi_b: str):
    """補間結果の表示。"""
    if not candidates:
        st.info("有効な中間分子が見つかりませんでした。")
        return

    st.success(f"✅ {len(candidates)}件の中間構造を生成")

    rows = []
    for i, c in enumerate(candidates):
        rows.append({
            "ステップ": i,
            "SMILES": c.smiles,
            "予測値": round(c.predicted_value, 4) if c.predicted_value is not None else None,
            "新規": "✅" if c.is_novel else "—",
        })

    _df = pd.DataFrame(rows)
    st.dataframe(_df, use_container_width=True, hide_index=True)

    # ユニーク分子数
    unique = set(c.smiles for c in candidates)
    st.metric("ユニーク分子数", f"{len(unique)} / {len(candidates)}")


def _render_degeneracy_map(dmap, dims: tuple[int, int]):
    """縮退マップのヒートマップ表示。"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("plotlyが必要です: pip install plotly")
        return

    dim1, dim2 = dims
    n_unique = len(dmap.unique_smiles)

    st.metric("ユニーク分子数", n_unique)
    st.metric("スキャン点数", len(dmap.grid_points))

    # ラベルを2Dグリッドに整形
    n_grid = int(np.sqrt(len(dmap.labels)))
    if n_grid * n_grid == len(dmap.labels):
        z = dmap.labels.reshape(n_grid, n_grid)
        x = np.unique(dmap.grid_points[:, dim1])
        y = np.unique(dmap.grid_points[:, dim2])

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=np.round(x, 3),
            y=np.round(y, 3),
            colorscale="Turbo",
            colorbar=dict(title="分子ID"),
        ))
        fig.update_layout(
            title=f"縮退マップ: PC{dim1+1} × PC{dim2+1}（{n_unique}種類の分子）",
            xaxis_title=f"PC{dim1+1}",
            yaxis_title=f"PC{dim2+1}",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "同じ色の領域 = PCA空間で異なる座標だが同じ分子にデコードされる「縮退」領域。"
            "色の種類が多い = 探索空間の表現力が高い。"
        )
    else:
        st.warning("グリッドサイズが不正です。")
