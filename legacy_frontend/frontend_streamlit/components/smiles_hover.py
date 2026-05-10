"""
frontend_streamlit/components/smiles_hover.py

SMILES 文字列のマウスオーバー時に 2D 構造を表示する共通コンポーネント。

提供する機能:
  1. smiles_to_svg_b64(smiles)          - SMILES → base64 エンコード SVG 文字列
  2. render_smiles_table(df, smiles_col) - ホバー付き HTML テーブルを st.components で描画
  3. add_smiles_hover_to_plotly(fig, smiles_list, idx=None) - Plotly 図に SMILES 構造ホバーを追加
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── SVG 生成 ──────────────────────────────────────────────────────────────────

def smiles_to_svg_b64(smiles: str, width: int = 200, height: int = 200) -> str:
    """SMILES → base64 エンコードされた SVG 文字列を返す。
    RDKit 未インストール時は空文字列を返す。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return ""

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().clearBackground = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"
    except Exception:
        return ""


def smiles_to_png_b64(smiles: str, width: int = 200, height: int = 200) -> str:
    """SMILES → base64 エンコードされた PNG 文字列を返す（SVG が使えない環境向けフォールバック）。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return ""

        img = Draw.MolToImage(mol, size=(width, height))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _best_smiles_img(smiles: str, width: int = 200, height: int = 200) -> str:
    """SVG → PNG の順で画像データ URI を返す。両方失敗時は空文字列。"""
    uri = smiles_to_svg_b64(smiles, width, height)
    if not uri:
        uri = smiles_to_png_b64(smiles, width, height)
    return uri


# ── HTML テーブルホバー ────────────────────────────────────────────────────────

def render_smiles_table(
    df: pd.DataFrame,
    smiles_col: str,
    max_rows: int = 100,
    img_size: int = 220,
    height: int = 500,
) -> None:
    """SMILES 列のセルにマウスオーバーすると 2D 構造がポップアップする HTML テーブルを描画する。

    Parameters
    ----------
    df         : 表示するデータフレーム
    smiles_col : SMILES 文字列が入っている列名
    max_rows   : 最大表示行数（過多のときに絞る）
    img_size   : ポップアップ画像サイズ (px)
    height     : コンポーネントの高さ (px)
    """
    if smiles_col not in df.columns:
        st.dataframe(df, use_container_width=True)
        return

    display_df = df.head(max_rows).reset_index(drop=True)

    # ── ヘッダー行 HTML ──
    header_cells = "".join(
        f"<th>{col}</th>" for col in display_df.columns
    )
    header_html = f"<tr>{header_cells}</tr>"

    # ── データ行 HTML ──
    rows_html = ""
    for _, row in display_df.iterrows():
        cells_html = ""
        for col in display_df.columns:
            val = row[col]
            val_str = str(val) if pd.notna(val) else ""

            if col == smiles_col and val_str:
                img_uri = _best_smiles_img(val_str, img_size, img_size)
                if img_uri:
                    # ホバー付きセル
                    cells_html += (
                        f'<td class="smiles-cell">'
                        f'  <span class="smiles-text">{val_str[:30]}{"…" if len(val_str) > 30 else ""}</span>'
                        f'  <div class="smiles-popup">'
                        f'    <img src="{img_uri}" width="{img_size}" height="{img_size}" />'
                        f'    <div class="smiles-raw">{val_str}</div>'
                        f'  </div>'
                        f'</td>'
                    )
                else:
                    cells_html += f"<td>{val_str}</td>"
            else:
                # 数値は右揃え
                align = ' style="text-align:right"' if isinstance(val, (int, float)) else ""
                cells_html += f"<td{align}>{val_str}</td>"

        rows_html += f"<tr>{cells_html}</tr>"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: 'Segoe UI', sans-serif;
    font-size: 12px;
    background: #0e1117;
    color: #e0e0f0;
    margin: 0;
    padding: 4px;
  }}
  .table-wrapper {{
    overflow: auto;
    max-height: {height - 20}px;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    min-width: 600px;
  }}
  th {{
    background: #1a1f2e;
    color: #7fb3f5;
    padding: 6px 10px;
    border: 1px solid #2a2f3e;
    position: sticky;
    top: 0;
    z-index: 2;
    white-space: nowrap;
  }}
  td {{
    padding: 5px 10px;
    border: 1px solid #1e2435;
    white-space: nowrap;
    max-width: 220px;
    overflow: hidden;
    text-overflow: ellipsis;
    vertical-align: middle;
  }}
  tr:nth-child(even) td {{ background: #12161f; }}
  tr:hover td {{ background: #1c2640; }}

  /* ── SMILES ホバー ── */
  .smiles-cell {{
    position: relative;
    cursor: pointer;
    color: #7fffd4;
    font-style: italic;
  }}
  .smiles-text {{
    border-bottom: 1px dashed #7fffd4;
  }}
  .smiles-popup {{
    display: none;
    position: fixed;           /* fixed でスクロールにも追従 */
    z-index: 9999;
    background: #1a1f2e;
    border: 2px solid #4c9be8;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    pointer-events: none;
  }}
  .smiles-popup img {{
    display: block;
    background: #ffffff;
    border-radius: 6px;
  }}
  .smiles-raw {{
    margin-top: 6px;
    font-size: 10px;
    color: #aaa;
    max-width: {img_size}px;
    word-break: break-all;
  }}
  .smiles-cell:hover .smiles-popup {{
    display: block;
  }}
</style>
</head>
<body>
<div class="table-wrapper">
  <table>
    <thead>{header_html}</thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
<script>
  // fixed ポップアップをマウス位置に追従させる
  document.querySelectorAll('.smiles-cell').forEach(cell => {{
    const popup = cell.querySelector('.smiles-popup');
    cell.addEventListener('mousemove', e => {{
      let x = e.clientX + 16;
      let y = e.clientY - 10;
      // 画面右端でめくれる対策
      if (x + {img_size + 40} > window.innerWidth) x = e.clientX - {img_size + 40};
      if (y + {img_size + 60} > window.innerHeight) y = e.clientY - {img_size + 60};
      popup.style.left = x + 'px';
      popup.style.top  = y + 'px';
    }});
  }});
</script>
</body>
</html>
"""
    components.html(html, height=height, scrolling=False)


# ── Plotly ホバー拡張 ─────────────────────────────────────────────────────────

def add_smiles_hover_to_plotly(
    fig,
    smiles_list: list[str],
    label_list: Optional[list[str]] = None,
    img_size: int = 180,
) -> None:
    """Plotly 散布図のホバーに SMILES 2D 画像を埋め込む。

    plotly の hovertemplate に base64 SVG を <img> タグとして埋め込み、
    ホバー時に 2D 構造を表示する。

    Parameters
    ----------
    fig         : plotly の Figure オブジェクト（in-place 更新）
    smiles_list : 各データ点に対応する SMILES 文字列のリスト
    label_list  : 各点に表示するラベル（例: compound ID）。None の場合は SMILES を使用
    img_size    : 画像サイズ (px)
    """
    import plotly.graph_objects as go

    imgs = []
    for smi in smiles_list:
        uri = _best_smiles_img(smi, img_size, img_size) if smi else ""
        imgs.append(uri)

    labels = label_list if label_list is not None else smiles_list

    for trace in fig.data:
        n = len(trace.x) if hasattr(trace, "x") and trace.x is not None else 0
        if n == 0:
            continue

        # customdata に [smiles, label, img_uri] をセット
        custom = []
        for i in range(n):
            smi = smiles_list[i] if i < len(smiles_list) else ""
            lbl = labels[i] if i < len(labels) else ""
            img = imgs[i] if i < len(imgs) else ""
            custom.append([smi, lbl, img])

        trace.customdata = custom

        # hovertemplate に <img> タグを埋め込む
        # plotly は hovertemplate 内の HTML を一部サポートする（<b>, <br>, <img> など）
        trace.hovertemplate = (
            "<b>%{customdata[1]}</b><br>"
            "SMILES: %{customdata[0]}<br>"
            "<img src='%{customdata[2]}' "
            f"width='{img_size}' height='{img_size}' "
            "style='background:white; border-radius:6px; margin-top:4px;'>"
            "<extra></extra>"
        )


# ── 便利ラッパー ───────────────────────────────────────────────────────────────

def auto_render(
    df: pd.DataFrame,
    smiles_col: Optional[str] = None,
    height: int = 450,
    max_rows: int = 100,
) -> None:
    """SMILES 列を自動検出して render_smiles_table / st.dataframe を切り替える。

    smiles_col が None のときは列名から自動検出を試みる。
    """
    if smiles_col is None:
        for col in df.columns:
            if col.lower() in ("smiles", "smi", "smiles_col"):
                smiles_col = col
                break

    if smiles_col and smiles_col in df.columns:
        render_smiles_table(df, smiles_col=smiles_col, max_rows=max_rows, height=height)
    else:
        st.dataframe(df.head(max_rows), use_container_width=True, height=height)
