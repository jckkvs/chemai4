"""
frontend_streamlit/pages/help_page.py

目的変数に対する推奨説明変数のヘルプビュー画面
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.chem.recommender import get_all_target_recommendations

def render_help_page() -> None:
    """推奨説明変数のヘルプ・データベース一覧ページ"""
    st.markdown("## 📚 ヘルプ / ガイド")

    guide_tab, rec_tab = st.tabs(["📖 使い方ガイド", "🔬 推奨変数データベース"])

    with guide_tab:
        st.markdown("""
### 🚀 クイックスタート（3ステップで解析完了）

| ステップ | 操作 | 説明 |
|:--------:|------|------|
| **① データ読込** | CSVファイルをドラッグ＆ドロップ | SMILES列があれば自動検出されます |
| **② 目的変数設定** | プルダウンで予測したい列を選択 | タスク（回帰/分類）も自動判定 |
| **③ 🚀 解析実行** | ボタンを1クリック | 複数モデルを自動比較し、最良モデルを提示 |

---

### 🔧 計算エンジンの選び方

| カテゴリ | おすすめ場面 | 速度 |
|----------|------------|------|
| 🧪 **基本記述子** (RDKit等) | まず最初に。ほぼ全てのケースで有効 | 🟢 高速 |
| 🔑 **フィンガープリント** | 構造活性相関 (SAR) 解析 | 🟢 高速 |
| 🤖 **学習型表現** | 大量データ+非線形パターン | 🟡 中 |
| ⚛️ **量子化学・物性** | エネルギー/溶媒和/pKa予測 | 🔴 重い |

> **💡 ヒント**: 初心者はRDKit（デフォルトON）だけで十分です。慣れたら追加エンジンを試してください。

---

### ⚠️ よくある質問

**Q: 解析が遅い**
→ XTBやCOSMO-RSなどの🔴エンジンをOFFにしてください。

**Q: 精度が出ない**
→ 記述子選択で目的変数に関連する記述子を絞り込むか、フィンガープリントを追加してください。

**Q: リーケージ警告が出た**
→ 類似度の高いデータが存在します。推奨されたGroupKFoldを使うことで、正しい評価ができます。
        """)

    with rec_tab:
        st.markdown("""
        各目的変数（予測したい物理化学的な性質）に対して、事前に有効性が知られている推奨説明変数（記述子）の一覧です。
        モデルの精度を高め、結果の解釈性を保つためには、やみくもに全変数を投入するのではなく、
        **目的にあった適切な変数を少数（8〜20個程度）選定することが最良のプラクティス**となります。
        """)

        st.markdown("---")

        recommendations = get_all_target_recommendations()

        for idx, rec in enumerate(recommendations):
            # 目的変数ごとに展開可能なExpander（デフォルトは閉じる）
            with st.expander(f"🔹 {rec.target_name}", expanded=False):
                st.markdown(f"**【事前知識】**\n{rec.summary}")
                
                # 記述子のリストをDataFrameに変換してリッチにテーブル表示
                desc_data = []
                for d in rec.descriptors:
                    desc_data.append({
                        "カテゴリ": d.library,
                        "推奨記述子名": d.name,
                        "物理的・化学的意味": d.meaning,
                        "根拠ソース・主な論文": d.source,
                    })
                df_desc = pd.DataFrame(desc_data)
                
                # Streamlitのdataframe表示でスタイリング
                st.dataframe(
                    df_desc,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "カテゴリ": st.column_config.TextColumn("ライブラリ", width="small"),
                        "推奨記述子名": st.column_config.TextColumn("記述子名", width="medium"),
                        "物理的・化学的意味": st.column_config.TextColumn("意味付け", width="large"),
                        "根拠ソース・主な論文": st.column_config.TextColumn("出典", width="medium"),
                    }
                )

if __name__ == "__main__":
    # 単独実行テスト用
    st.set_page_config(page_title="推奨変数ヘルプ", layout="wide")
    render_help_page()
