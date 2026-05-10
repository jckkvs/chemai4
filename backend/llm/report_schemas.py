"""
backend/llm/report_schemas.py

LLM生成レポートのデータクラス定義。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReportSection:
    """単一レポートセクション"""
    title: str               # セクションタイトル（日本語）
    content: str             # Markdown形式の内容
    section_type: str        # "summary" | "performance" | "features" | "data_quality" | "recommendations"


@dataclass
class ReportResult:
    """LLM生成レポート全体"""
    success: bool
    title: str = ""
    summary: str = ""
    sections: list[ReportSection] = field(default_factory=list)
    raw_response: str = ""
    tokens_used: int = 0
    error_message: str = ""
    generated_at: str = ""       # ISO形式の生成日時
    model_info: str = ""        # 使用したLLMモデル情報

    def to_markdown(self) -> str:
        """全文Markdownを生成する。"""
        lines: list[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        if self.summary:
            lines.append("## エグゼクティブサマリー")
            lines.append(self.summary)
            lines.append("")
        for sec in self.sections:
            lines.append(f"## {sec.title}")
            lines.append(sec.content)
            lines.append("")
        # フッター：メタデータ
        if self.generated_at or self.model_info:
            lines.append("---")
            if self.generated_at:
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(self.generated_at)
                    lines.append(f"*生成日時: {dt.strftime('%Y-%m-%d %H:%M:%S')}*")
                except Exception:
                    lines.append(f"*生成日時: {self.generated_at}*")
            if self.model_info:
                lines.append(f"*使用モデル: {self.model_info}*")
        return "\n".join(lines)

    def to_html(self) -> str:
        """全文HTMLを生成する。"""
        css = """
        <style>
        body { font-family: 'Noto Sans JP', 'Yu Gothic', sans-serif; margin: 2em; color: #333; }
        h1 { color: #0d47a1; border-bottom: 2px solid #0d47a1; padding-bottom: 0.3em; }
        h2 { color: #1565c0; margin-top: 1.5em; border-left: 4px solid #1565c0; padding-left: 0.5em; }
        pre { background: #f5f5f5; padding: 1em; border-radius: 4px; overflow-x: auto; }
        code { background: #f5f5f5; padding: 0.2em 0.4em; border-radius: 3px; }
        </style>
        """
        body_parts: list[str] = []
        body_parts.append(f"<h1>{self._esc(self.title)}</h1>")
        if self.summary:
            body_parts.append(f"<h2>エグゼクティブサマリー</h2>")
            body_parts.append(f"<p>{self._esc(self.summary)}</p>")
        for sec in self.sections:
            body_parts.append(f"<h2>{self._esc(sec.title)}</h2>")
            # 簡易Markdown→HTML変換（段落分け）
            html_content = self._md_to_html_simple(sec.content)
            body_parts.append(html_content)
        # フッター：メタデータ
        if self.generated_at or self.model_info:
            footer_parts = []
            if self.generated_at:
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(self.generated_at)
                    footer_parts.append(f"生成日時: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception:
                    footer_parts.append(f"生成日時: {self.generated_at}")
            if self.model_info:
                footer_parts.append(f"使用モデル: {self.model_info}")
            footer = " | ".join(footer_parts)
            body_parts.append(f"<hr><p style='font-size:0.8em; color:#888;'>{self._esc(footer)}</p>")
        return f"<!DOCTYPE html><html lang='ja'><head><meta charset='UTF-8'>{css}</head><body>{''.join(body_parts)}</body></html>"

    @staticmethod
    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def _md_to_html_simple(md: str) -> str:
        """簡易Markdown→HTML変換（段落分け・リスト・太字）"""
        lines = md.split("\n")
        html_lines: list[str] = []
        in_list = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                content = stripped[2:].strip()
                html_lines.append(f"<li>{ReportResult._esc(content)}</li>")
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                if stripped:
                    # 太字 **text**
                    import re
                    content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", stripped)
                    html_lines.append(f"<p>{ReportResult._esc(content)}</p>")
                else:
                    html_lines.append("<br>")
        if in_list:
            html_lines.append("</ul>")
        return "\n".join(html_lines)

    def to_json(self) -> str:
        """構造化JSONとして出力する。"""
        return json.dumps({
            "title": self.title,
            "summary": self.summary,
            "generated_at": self.generated_at,
            "model_info": self.model_info,
            "tokens_used": self.tokens_used,
            "sections": [
                {"title": s.title, "content": s.content, "section_type": s.section_type}
                for s in self.sections
            ],
        }, ensure_ascii=False, indent=2)
