"""
backend/data/eda_report.py

EDA（探索的データ分析）の結果を静的なHTMLレポートとして出力するモジュール。
ydata-profiling 等の大規模ライブラリに依存せず、軽量なレポートを自前で生成する。
"""
import os
import json
from typing import Any, Dict

def generate_eda_html_report(eda_results: Dict[str, Any], output_path: str) -> str:
    """
    EDA結果の辞書からシンプルなHTMLレポートを生成して保存する。

    Args:
        eda_results: backend.data.eda モジュールで計算した結果の辞書
        output_path: 保存先のHTMLファイルパス

    Returns:
        保存されたファイルのパス
    """
    summary = eda_results.get("summary", {})
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <title>EDA Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 2rem; background-color: #f9f9f9; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; }}
            h2 {{ color: #2980b9; margin-top: 2rem; }}
            .card {{ background: #fff; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>探索的データ分析 (EDA) レポート</h1>
        
        <div class="card">
            <h2>データサマリー</h2>
            <ul>
                <li>行数: {summary.get("n_rows", "N/A")}</li>
                <li>列数: {summary.get("n_cols", "N/A")}</li>
                <li>数値列数: {summary.get("n_numeric", "N/A")}</li>
                <li>カテゴリ列数: {summary.get("n_categorical", "N/A")}</li>
                <li>全体の欠損率: {summary.get("total_null_rate", 0) * 100:.2f}%</li>
                <li>重複行数: {summary.get("n_duplicates", "N/A")}</li>
                <li>メモリ使用量: {summary.get("memory_mb", "N/A")} MB</li>
            </ul>
        </div>
        
        <div class="card">
            <h2>外れ値検出</h2>
            <p>※ 詳細はアプリケーション上からご確認ください。</p>
            <pre>{json.dumps(eda_results.get('outliers', []), indent=2, ensure_ascii=False)}</pre>
        </div>
        
        <div class="card">
            <h2>多重共線性 (VIF)</h2>
            <pre>{json.dumps(eda_results.get('vif', {}), indent=2, ensure_ascii=False)}</pre>
        </div>
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return output_path
