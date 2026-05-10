"""
frontend_nicegui/components/eda_dim_panel.py
【再構築】次元削減・重要度表示パネル（NiceGUI用）
"""

from nicegui import ui, app
import pandas as pd
from typing import Optional
import logging

from backend.data.eda_core import (
    run_dim_reduction_with_importance,
    CombinedEDAResult,
    ReductionMethod,
    _convert_to_native
)

logger = logging.getLogger(__name__)

class DimReductionState:
    _cache: dict[str, dict[str, CombinedEDAResult]] = {}
    
    @classmethod
    def get(cls, df_hash: str) -> Optional[dict[str, CombinedEDAResult]]:
        return cls._cache.get(df_hash)
    
    @classmethod
    def set(cls, df_hash: str, results: dict[str, CombinedEDAResult]):
        cls._cache[df_hash] = results
    
    @classmethod
    def clear(cls, df_hash: Optional[str] = None):
        if df_hash:
            cls._cache.pop(df_hash, None)
        else:
            cls._cache.clear()

@ui.refreshable
def dim_reduction_panel(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    scale: bool = True
):
    # 安全なハッシュキーの生成（to_jsonは巨大データでクラッシュ・激重化の原因になるため避ける）
    df_hash = f"{id(df)}_{df.shape}_{scale}"
    
    cached = DimReductionState.get(df_hash)
    if cached:
        _render_multiple_results(cached)
        return
    
    with ui.row().classes('w-full items-center'):
        spinner = ui.spinner(size='lg').props('color=primary')
        label = ui.label('PCA と t-SNE を並行して計算中...').classes('ml-2')
    
    async def compute_and_render():
        try:
            plot_df = df.copy()
            if target_col and target_col in plot_df.columns:
                plot_df = plot_df.drop(columns=[target_col])
            
            numeric_df = plot_df.select_dtypes(include='number').dropna(axis=1, thresh=max(1, len(plot_df)*0.5))
            numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True).to_dict())
            
            if numeric_df.shape[1] < 2:
                ui.notify('数値特徴量が2列未満です', type='warning')
                error_result_dict = {
                    "error": CombinedEDAResult(warnings=["数値特徴量が2列未満です。次元削減をスキップしました。"])
                }
                DimReductionState.set(df_hash, error_result_dict)
                dim_reduction_panel.refresh()
                return
            
            # PCA と t-SNE の両方を計算
            results_dict = {}
            target_methods = ["pca"]
            
            # t-SNEの計算量（Barnes-Hut: O(D * N log N)）を考慮し、閾値を設定
            # ブラウザ上でストレスなく描画できる目安として N < 3000 または N*D < 500k とする
            N, D = numeric_df.shape
            if N <= 3000 and (N * D) <= 500000:
                target_methods.append("tsne")
            
            from nicegui import run
            for method in target_methods:
                results_dict[method] = await run.cpu_bound(
                    run_dim_reduction_with_importance,
                    numeric_df, method=method, scale=scale, top_n_importance=20
                )
            
            if "tsne" not in target_methods:
                results_dict["tsne"] = CombinedEDAResult(
                    warnings=[f"データサイズ (N={N}, 個数={N*D}) が大きいため、計算負荷(O(N log N))を考慮し t-SNE の自動計算をスキップしました。"]
                )
            
            DimReductionState.set(df_hash, results_dict)
            dim_reduction_panel.refresh()
            
        except Exception as e:
            logger.error(f"[EDA Panel] 計算エラー: {e}", exc_info=True)
            ui.notify(f'計算エラー: {str(e)}', type='negative')
            # 無限ループ防止：エラー状態をキャッシュに保存する
            error_result_dict = {
                "error": CombinedEDAResult(warnings=[f"内部エラー: {str(e)}"])
            }
            DimReductionState.set(df_hash, error_result_dict)
            dim_reduction_panel.refresh()
        finally:
            pass
    
    ui.timer(0.1, compute_and_render, once=True)

def _render_multiple_results(results_dict: dict[str, CombinedEDAResult]):
    if "error" in results_dict:
        ui.label(f'⚠️ {results_dict["error"].warnings[0]}').classes('text-red-600')
        return

    # PCAとt-SNEのプロットを横並びに表示
    with ui.row().classes('w-full q-col-gutter-md'):
        for method in ["pca", "tsne"]:
            if method not in results_dict:
                continue
                
            result = results_dict[method]
            
            if result.warnings:
                for w in result.warnings:
                    ui.notify(f"[{method.upper()}] {w}", type='warning', close_button=True)
            
            with ui.column().classes('col-12 col-md-6'):
                with ui.card().classes('w-full h-full'):
                    ui.label(f'🔍 {method.upper()} 可視化').classes('text-lg font-bold')
                    
                    if result.dim_reduction and result.dim_reduction.status == "success":
                        coords = result.dim_reduction.coordinates
                        x_vals = [coords[k][0] for k in coords]
                        y_vals = [coords[k][1] for k in coords]
                        labels = list(coords.keys())
                        
                        import plotly.graph_objects as go

                        fig = go.Figure(data=[go.Scatter(
                            x=x_vals, y=y_vals,
                            mode='markers',
                            marker=dict(size=6, color='steelblue', opacity=0.7),
                            text=labels,
                            hoverinfo='text'
                        )])
                        
                        axis_name = "PC" if method == "pca" else "t-SNE"
                        fig.update_layout(
                            xaxis_title=f'{axis_name}1',
                            yaxis_title=f'{axis_name}2',
                            height=350,
                            template='plotly_white',
                            margin=dict(l=40, r=20, t=20, b=40)
                        )
                        
                        if result.dim_reduction.explained_variance:
                            var_pct = [f"{v*100:.1f}%" for v in result.dim_reduction.explained_variance[:2]]
                            fig.update_layout(title=dict(text=f'寄与率: {", ".join(var_pct)}', font=dict(size=12)))
                        
                        ui.plotly(fig).classes('w-full')
                        
                    else:
                        msg = result.dim_reduction.error_message if result.dim_reduction else "計算未実行"
                        ui.label(f'⚠️ {msg}').classes('text-red-600')
    
    with ui.card().classes('w-full'):
        ui.label('📊 特徴量重要度（上位20）').classes('text-lg font-bold')
        
        if result.feature_importance and result.feature_importance.status == "success":
            importance = result.feature_importance.importance
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            names = [f for f, _ in top_features]
            scores = [s for _, s in top_features]
            
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Bar(
                y=names, x=scores,
                orientation='h',
                marker=dict(color=scores, colorscale='Blues'),
                hoverinfo='x+y'
            )])
            
            fig.update_layout(
                title=f'重要度（{result.feature_importance.metric}）',
                xaxis_title='重要度スコア（正規化）',
                height=400,
                template='plotly_white',
                margin=dict(l=100, r=20, t=40, b=20)
            )
            
            ui.plotly(fig).classes('w-full')
            
            with ui.expansion('📋 全特徴量の重要度一覧', icon='table_chart').classes('w-full'):
                ui.table(
                    columns=[
                        {'name': 'rank', 'label': '順位', 'field': 'rank', 'sortable': True},
                        {'name': 'feature', 'label': '特徴量', 'field': 'feature', 'sortable': True},
                        {'name': 'importance', 'label': '重要度', 'field': 'importance', 'sortable': True, 'format': lambda v: f'{v:.3f}'}
                    ],
                    rows=[
                        {'rank': i+1, 'feature': f, 'importance': s}
                        for i, (f, s) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                    ],
                    pagination=10
                ).classes('w-full')
        else:
            msg = result.feature_importance.error_message if result.feature_importance else "重要度計算未実行"
            ui.label(f'⚠️ {msg}').classes('text-red-600')
    
    with ui.row().classes('w-full justify-end'):
        ui.button('🔄 再計算', icon='refresh', 
                 on_click=lambda: DimReductionState.clear() or dim_reduction_panel.refresh()
                ).props('outline')
        ui.button('📥 CSV出力', icon='download',
                 on_click=lambda: _export_results(result, method)
                ).props('outline')

def _export_results(result: CombinedEDAResult, method: str):
    import io
    if not result.dim_reduction or result.dim_reduction.status != "success":
        ui.notify('エクスポート対象の結果がありません', type='warning')
        return
    
    coords_df = pd.DataFrame([
        {'sample_id': k, 'x': v[0], 'y': v[1]}
        for k, v in result.dim_reduction.coordinates.items()
    ])
    coords_df.to_csv('dim_reduction_coords.csv', index=False)
    
    if result.feature_importance and result.feature_importance.status == "success":
        imp_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in result.feature_importance.importance.items()
        ]).sort_values('importance', ascending=False)
        imp_df.to_csv('feature_importance.csv', index=False)
    
    ui.notify('✅ 結果をCSV出力しました', type='positive')

@ui.refreshable
def dim_reduction_settings(
    on_apply: callable,
    default_scale: bool = True
):
    with ui.card().classes('w-full'):
        ui.label('⚙️ 次元削減設定').classes('text-lg font-bold')
        
        with ui.row().classes('w-full items-end gap-4'):
            scale = ui.switch('標準化（StandardScaler）', value=default_scale).classes('mt-2')
            
            ui.button('▶ 適用', icon='play_arrow',
                     on_click=lambda: on_apply(
                         scale=scale.value
                     )
                    ).props('color=primary')
        
        with ui.expansion('ℹ️ 手法の違い', icon='info').classes('w-full'):
            ui.markdown('''
            - **PCA**: 線形変換。計算が高速・解釈性が高い。寄与率で情報損失を評価可能。
            - **t-SNE**: 非線形変換。局所構造の可視化に優れるが、計算コスト高・パラメータ依存。
            ''').classes('text-sm text-gray-600')
