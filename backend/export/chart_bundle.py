"""
backend/export/chart_bundle.py
解析で生成されたすべての Plotly/Matplotlib チャートを
ZIP ファイルとして一括エクスポートするエンジン。
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from .base import BaseExporter


class ChartBundleExporter(BaseExporter):
    """チャート画像群を ZIP に束ねるエクスポータ。

    state["chart_paths"] に格納された画像パスリストを読み込み、
    ひとつの ZIP ファイルにまとめる。
    """

    def export(self, result: dict[str, Any], filename: str) -> Path:
        """チャート画像を ZIP にまとめて output_dir へ書き出す。

        Parameters
        ----------
        result : dict
            任意キー: "chart_paths" (list of str|Path)
        filename : str
            拡張子なしのファイル名（例: "charts_bundle"）。

        Returns
        -------
        Path
            書き出した .zip ファイルの絶対パス。
        """
        out_path = self.output_dir / f"{filename}.zip"
        chart_paths: list[Path] = [
            Path(p) for p in result.get("chart_paths", [])
            if Path(p).exists()
        ]

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if not chart_paths:
                # チャートがない場合、README を同梱する
                zf.writestr(
                    "README.txt",
                    "ChemAI ML Studio: チャートバンドル\n\n"
                    "解析チャートが見つかりませんでした。\n"
                    "解析を実行してから再度エクスポートしてください。\n",
                )
            else:
                for cp in chart_paths:
                    zf.write(cp, arcname=cp.name)

            # メタデータ JSON を同梱
            import json
            meta = {
                "best_model_name": result.get("best_model_name", "N/A"),
                "metrics": result.get("metrics", {}),
                "chart_count": len(chart_paths),
                "chart_files": [cp.name for cp in chart_paths],
            }
            zf.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))

        return out_path.resolve()
