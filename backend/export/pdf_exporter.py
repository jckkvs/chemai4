"""
backend/export/pdf_exporter.py
ReportLab を用いた PDF レポート出力エンジン。

依存: reportlab>=4.0
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False

from .base import BaseExporter


class PDFExporter(BaseExporter):
    """解析結果を PDF に変換して保存するエクスポータ。

    Attributes
    ----------
    ACCENT : colors.HexColor
        レポートのアクセントカラー（ダークブルー系）。

    Notes
    -----
    日本語フォントが必要なため、実行環境に NotoSansJP-Regular.ttf が存在しない場合は
    標準 Helvetica にフォールバックする。
    """

    ACCENT = colors.HexColor("#1a3a5c") if _HAS_REPORTLAB else None
    LIGHT_BG = colors.HexColor("#f0f4f8") if _HAS_REPORTLAB else None

    def __init__(self, output_dir: str | Path = "exports") -> None:
        super().__init__(output_dir)
        self._register_japanese_font()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _register_japanese_font(self) -> None:
        """NotoSansJP が存在すれば ReportLab に登録する。"""
        candidate_paths = [
            Path(__file__).parent.parent.parent / "assets" / "fonts" / "NotoSansJP-Regular.ttf",
            Path("assets/fonts/NotoSansJP-Regular.ttf"),
        ]
        for p in candidate_paths:
            if p.exists():
                pdfmetrics.registerFont(TTFont("NotoSansJP", str(p)))
                self._font_name = "NotoSansJP"
                return
        self._font_name = "Helvetica"

    def _styles(self) -> dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()
        font = self._font_name
        return {
            "title": ParagraphStyle(
                "Title",
                parent=base["Title"],
                fontName=font,
                fontSize=22,
                textColor=self.ACCENT,
                alignment=TA_CENTER,
                spaceAfter=6,
            ),
            "section": ParagraphStyle(
                "Section",
                parent=base["Heading2"],
                fontName=font,
                fontSize=13,
                textColor=self.ACCENT,
                spaceBefore=12,
                spaceAfter=4,
            ),
            "body": ParagraphStyle(
                "Body",
                parent=base["BodyText"],
                fontName=font,
                fontSize=10,
                leading=16,
                alignment=TA_LEFT,
            ),
            "caption": ParagraphStyle(
                "Caption",
                parent=base["BodyText"],
                fontName=font,
                fontSize=8,
                textColor=colors.grey,
                alignment=TA_CENTER,
            ),
        }

    def _metrics_table(self, metrics: dict[str, Any], styles: dict) -> Table:
        """評価指標を左列ラベル・右列値の表として組み立てる。"""
        data = [["指標", "値"]] + [
            [k, f"{v:.4f}" if isinstance(v, float) else str(v)]
            for k, v in metrics.items()
        ]
        table = Table(data, colWidths=[80 * mm, 60 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), self.ACCENT),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), self._font_name),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                    ("BACKGROUND", (0, 1), (-1, -1), self.LIGHT_BG),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [self.LIGHT_BG, colors.white]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return table

    def _importance_table(self, importances: dict[str, float], styles: dict, top_n: int = 10) -> Table:
        """特徴量重要度を上位 top_n 件でテーブル化する。"""
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        data = [["特徴量名", "重要度スコア"]] + [
            [feat, f"{score:.4f}"]
            for feat, score in sorted_imp
        ]
        table = Table(data, colWidths=[110 * mm, 50 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), self.ACCENT),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), self._font_name),
                    ("FONTSIZE", (0, 0), (-1, 0), 11),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [self.LIGHT_BG, colors.white]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def export(self, result: dict[str, Any], filename: str) -> Path:
        """解析結果を PDF に変換して output_dir へ書き出す。

        Parameters
        ----------
        result : dict
            必須キー: "best_model_name", "metrics"
            任意キー: "feature_importances", "chart_paths" (リスト of Path), "ai_commentary"
        filename : str
            拡張子を含まないファイル名 (例: "analysis_report")。

        Returns
        -------
        Path
            書き出した PDF ファイルの絶対パス。

        Raises
        ------
        ImportError
            reportlab が未インストールの場合。
        """
        if not _HAS_REPORTLAB:
            raise ImportError(
                "PDF出力には reportlab パッケージが必要です。\n"
                "インストール: pip install reportlab>=4.0"
            )
        out_path = self.output_dir / f"{filename}.pdf"
        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
        )

        styles = self._styles()
        story = []

        # ── タイトルセクション ──
        story.append(Paragraph("ChemAI ML Studio: 解析レポート", styles["title"]))
        story.append(HRFlowable(width="100%", thickness=1.5, color=self.ACCENT))
        story.append(Spacer(1, 6 * mm))

        # ── 最良モデル ──
        model_name = result.get("best_model_name", "N/A")
        story.append(Paragraph("🏆 最良モデル", styles["section"]))
        story.append(Paragraph(f"採択モデル: <b>{model_name}</b>", styles["body"]))
        story.append(Spacer(1, 4 * mm))

        # ── 評価指標テーブル ──
        metrics = result.get("metrics", {})
        if metrics:
            story.append(Paragraph("📊 評価指標", styles["section"]))
            story.append(self._metrics_table(metrics, styles))
            story.append(Spacer(1, 4 * mm))

        # ── AI考察コメント ──
        ai_comment = result.get("ai_commentary", "")
        if ai_comment:
            story.append(Paragraph("🤖 AIアシスタントによる考察", styles["section"]))
            story.append(Paragraph(ai_comment, styles["body"]))
            story.append(Spacer(1, 4 * mm))

        # ── 特徴量重要度テーブル ──
        importances = result.get("feature_importances", {})
        if importances:
            story.append(Paragraph("🔑 特徴量重要度 (上位10件)", styles["section"]))
            story.append(self._importance_table(importances, styles))
            story.append(Spacer(1, 4 * mm))

        # ── 添付チャート画像 ──
        chart_paths: list[Path] = [
            Path(p) for p in result.get("chart_paths", [])
            if Path(p).exists()
        ]
        if chart_paths:
            story.append(Paragraph("📈 解析チャート", styles["section"]))
            for cp in chart_paths:
                try:
                    img = Image(str(cp), width=160 * mm, height=90 * mm, kind="proportional")
                    story.append(img)
                    story.append(Paragraph(cp.stem, styles["caption"]))
                    story.append(Spacer(1, 3 * mm))
                except Exception:
                    story.append(Paragraph(f"⚠️ 画像読み込み失敗: {cp.name}", styles["body"]))

        doc.build(story)
        return out_path.resolve()
