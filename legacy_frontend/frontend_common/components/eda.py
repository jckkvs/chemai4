"""
frontend_common/components/eda.py

各フロントエンド（Django/Streamlit/NiceGUI）で共通して使用するEDA（探索的データ分析）
の可視化ロジックやデータフォーマット変換を提供するモジュール。
"""
from typing import Any, Dict
import pandas as pd

def format_summary_stats(summary_data: Dict[str, Any]) -> str:
    """
    EDAのサマリーデータを表示用HTMLまたはMarkdown文字列にフォーマットする。
    """
    n_rows = summary_data.get("n_rows", 0)
    n_cols = summary_data.get("n_cols", 0)
    n_num = summary_data.get("n_numeric", 0)
    null_rate = summary_data.get("total_null_rate", 0.0) * 100
    
    return f"""
    <div class="eda-summary">
        <ul>
            <li><strong>行数/列数:</strong> {n_rows} / {n_cols}</li>
            <li><strong>数値列:</strong> {n_num}列</li>
            <li><strong>全体の欠損率:</strong> {null_rate:.2f}%</li>
        </ul>
    </div>
    """

def get_plotly_code_for_correlation(corr_data: Dict[str, Any]) -> str:
    """
    相関行列のデータからPlotlyのJSコード等を生成するヘルパー。
    （実装予定）
    """
    pass
