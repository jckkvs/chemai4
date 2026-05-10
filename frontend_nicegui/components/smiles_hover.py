# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/smiles_hover.py

SMILES 文字列のホバー時に 2D 構造をポップアップ表示する共通コンポーネント。
NiceGUI版: HTML/JavaScript挿入でSVG構造画像を表示。

機能:
  1. render_smiles_table(df, smiles_col) - ホバー付きHTMLテーブル
  2. smiles_to_svg_b64(smiles)           - SMILES → base64 SVG
"""
from __future__ import annotations

import base64
from typing import Optional

import pandas as pd
from nicegui import ui


# ── SVG 生成 ──

def smiles_to_svg_b64(smiles: str, width: int = 200, height: int = 200) -> str:
    """SMILES → base64 エンコードされた SVG data URI。RDKit未インストール時は空文字列。"""
    try:
        from rdkit import Chem
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
    """SMILES → base64 PNG data URI（SVGフォールバック）。"""
    try:
        import io
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


def _best_img(smiles: str, w: int = 200, h: int = 200) -> str:
    uri = smiles_to_svg_b64(smiles, w, h)
    if not uri:
        uri = smiles_to_png_b64(smiles, w, h)
    return uri


# ── ホバー付きHTMLテーブル ──

def render_smiles_table(
    df: pd.DataFrame,
    smiles_col: str,
    max_rows: int = 100,
    img_size: int = 220,
    height: int = 500,
) -> None:
    """SMILES列にホバーで2D構造ポップアップするHTMLテーブルをNiceGUI内に描画。"""
    if smiles_col not in df.columns:
        ui.table(
            columns=[{"name": c, "label": c, "field": c} for c in df.columns],
            rows=df.head(max_rows).to_dict("records"),
        ).classes("full-width")
        return

    display_df = df.head(max_rows).reset_index(drop=True)

    # ヘッダー
    header_cells = "".join(f"<th>{c}</th>" for c in display_df.columns)

    # データ行
    rows_html = ""
    for _, row in display_df.iterrows():
        cells = ""
        for col in display_df.columns:
            val = row[col]
            val_str = str(val) if pd.notna(val) else ""
            if col == smiles_col and val_str:
                img_uri = _best_img(val_str, img_size, img_size)
                if img_uri:
                    truncated = val_str[:30] + ("…" if len(val_str) > 30 else "")
                    cells += (
                        f'<td class="smi-cell">'
                        f'<span class="smi-text">{truncated}</span>'
                        f'<div class="smi-popup">'
                        f'<img src="{img_uri}" width="{img_size}" height="{img_size}"/>'
                        f'<div class="smi-raw">{val_str}</div>'
                        f'</div></td>'
                    )
                else:
                    cells += f"<td>{val_str}</td>"
            else:
                align = ' style="text-align:right"' if isinstance(val, (int, float)) else ""
                cells += f"<td{align}>{val_str}</td>"
        rows_html += f"<tr>{cells}</tr>"

    html = f"""
<style>
.smi-table {{ border-collapse:collapse; width:100%; font-size:12px; color:#e0e0f0; }}
.smi-table th {{ background:#1a1f2e; color:#7fb3f5; padding:6px 10px; border:1px solid #2a2f3e;
  position:sticky; top:0; z-index:2; white-space:nowrap; }}
.smi-table td {{ padding:5px 10px; border:1px solid #1e2435; white-space:nowrap;
  max-width:220px; overflow:hidden; text-overflow:ellipsis; }}
.smi-table tr:nth-child(even) td {{ background:#12161f; }}
.smi-table tr:hover td {{ background:#1c2640; }}
.smi-cell {{ position:relative; cursor:pointer; color:#7fffd4; font-style:italic; }}
.smi-text {{ border-bottom:1px dashed #7fffd4; }}
.smi-popup {{ display:none; position:fixed; z-index:9999;
  background:#1a1f2e; border:2px solid #4c9be8; border-radius:10px;
  padding:10px; box-shadow:0 8px 32px rgba(0,0,0,0.6); pointer-events:none; }}
.smi-popup img {{ display:block; background:#fff; border-radius:6px; }}
.smi-raw {{ margin-top:6px; font-size:10px; color:#aaa; max-width:{img_size}px; word-break:break-all; }}
.smi-cell:hover .smi-popup {{ display:block; }}
</style>
<div style="overflow:auto; max-height:{height}px;">
<table class="smi-table">
<thead><tr>{header_cells}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
<script>
document.querySelectorAll('.smi-cell').forEach(cell => {{
  const popup = cell.querySelector('.smi-popup');
  cell.addEventListener('mousemove', e => {{
    let x = e.clientX + 16, y = e.clientY - 10;
    if (x + {img_size + 40} > window.innerWidth) x = e.clientX - {img_size + 40};
    if (y + {img_size + 60} > window.innerHeight) y = e.clientY - {img_size + 60};
    popup.style.left = x + 'px';
    popup.style.top = y + 'px';
  }});
}});
</script>
"""
    ui.html(html)


# ── Plotlyホバー拡張 ──

def add_smiles_hover_to_plotly(
    fig,
    smiles_list: list[str],
    label_list: Optional[list[str]] = None,
    img_size: int = 180,
) -> None:
    """Plotly散布図のホバーにSMILES 2D画像を埋め込む。"""
    imgs = [_best_img(smi, img_size, img_size) if smi else "" for smi in smiles_list]
    labels = label_list if label_list is not None else smiles_list

    for trace in fig.data:
        n = len(trace.x) if hasattr(trace, "x") and trace.x is not None else 0
        if n == 0:
            continue
        custom = []
        for i in range(n):
            smi = smiles_list[i] if i < len(smiles_list) else ""
            lbl = labels[i] if i < len(labels) else ""
            img = imgs[i] if i < len(imgs) else ""
            custom.append([smi, lbl, img])
        trace.customdata = custom
        trace.hovertemplate = (
            "<b>%{customdata[1]}</b><br>"
            "SMILES: %{customdata[0]}<br>"
            f"<img src='%{{customdata[2]}}' width='{img_size}' height='{img_size}' "
            "style='background:white; border-radius:6px; margin-top:4px;'>"
            "<extra></extra>"
        )
