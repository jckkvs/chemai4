"""
frontend_nicegui/components/llm_report_tab.py

LLMを使った解析レポート生成タブ。
「🤖 LLMレポート」サブタブ内に表示される。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from nicegui import ui

logger = logging.getLogger(__name__)

# レポート保持用キー
_STATE_KEY = "llm_report_result"


def render_llm_report_tab(state: dict[str, Any]) -> None:
    """LLMレポート生成タブを描画する。"""

    ar = state.get("automl_result") or list(
        (state.get("automl_results") or {}).values()
    )[0] if state.get("automl_results") else None

    if ar is None:
        with ui.card().classes("glass-card q-pa-xl full-width text-center"):
            ui.icon("smart_toy", color="grey-7", size="xl").classes("q-mb-md")
            ui.label("LLMレポート生成にはモデル学習が必要です").classes(
                "text-h6 text-grey-5"
            )
            ui.label(
                "「📂 解析設定」→「🚀 解析開始」でモデルを学習してください。"
            ).classes("text-grey-6 q-mt-sm")
        return

    # ── ヘッダー ──
    with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
        ui.icon("smart_toy", color="cyan", size="md")
        ui.label("LLM解析レポート生成").classes("text-h6")

    ui.label(
        "LLM（Qwen等）を使って、解析結果から日本語の包括的レポートを自動生成します。"
    ).classes("text-caption text-grey q-mb-md")

    # ── モデル選択 & プロバイダー表示 ──
    _render_provider_info(state)

    # ── スピナーコンテナ（生成中表示用） ──
    spinner_container = ui.column().classes("full-width items-center q-mt-md")
    spinner_container.set_visibility(False)

    # ── 生成 & ダウンロードボタン ──
    with ui.row().classes("q-gutter-md q-mb-md"):
        gen_btn = (
            ui.button("🤖 レポート生成", on_click=lambda: _on_generate(ar, state, spinner_container))
            .props("color=cyan unelevated no-caps icon=smart_toy")
            .classes("btn-primary")
        )
        regen_btn = (
            ui.button("🔄 再生成", on_click=lambda: _on_generate(ar, state, spinner_container))
            .props("flat color=cyan no-caps icon=refresh")
            .classes("full-width")
        )
        dl_md_btn = (
            ui.button("📝 Markdown", on_click=lambda: _on_download_md(state))
            .props("outline color=teal no-caps icon=description")
            .classes("full-width")
        )
        dl_html_btn = (
            ui.button("🌐 HTML", on_click=lambda: _on_download_html(state))
            .props("outline color=purple no-caps icon=html")
            .classes("full-width")
        )
        clear_btn = (
            ui.button("🗑️ クリア", on_click=lambda: _on_clear(state, report_container, dl_md_btn, dl_html_btn))
            .props("flat color=grey no-caps icon=delete")
            .classes("full-width")
        )

    # 初期状態はダウンロードボタンを無効化
    dl_md_btn.disable()
    dl_html_btn.disable()


    # ── レポート表示エリア ──
    report_container = ui.column().classes("full-width q-mt-md")

    # 既存のレポートがあれば表示
    existing = state.get(_STATE_KEY)
    if existing is not None:
        dl_md_btn.enable()
        dl_html_btn.enable()
        _render_report_content(report_container, existing)

    # ── コールバック登録（タブ切り替え時の再描画用） ──
    state["_refresh_llm_report"] = lambda: _rebuild(
        report_container, ar, state, dl_md_btn, dl_html_btn
    )


# ──────────────────────────────────────────────────────────
# 内部ヘルパー
# ──────────────────────────────────────────────────────────

def _render_provider_info(state: dict) -> None:
    """現在のLLMプロバイダー情報を表示する。"""
    try:
        from backend.llm import get_llm_provider
        from backend.llm.provider import StubLLMProvider

        # 設定からプロバイダー名を取得
        provider_name = state.get("llm_provider", "huggingface")
        provider = get_llm_provider(provider_name)
        available = provider.is_available

        with ui.row().classes("items-center q-gutter-xs q-mb-sm"):
            if available:
                ui.badge("LLM利用可能", color="positive").props("dense")
                ui.label(f"プロバイダー: {provider.name}").classes(
                    "text-caption text-grey-5"
                )
            else:
                ui.badge("LLM利用不可（フォールバック）", color="warning").props(
                    "dense"
                )
                ui.label(
                    "スタブモードで静的レポートを生成します"
                ).classes("text-caption text-grey-5")
    except Exception:
        ui.label("LLMプロバイダー情報を取得できません").classes(
            "text-caption text-grey-5"
        )


async def _on_generate(ar, state: dict, spinner_container=None) -> None:
    """レポート生成ボタン押下時の処理。"""
    from nicegui import run

    # プログレス表示
    ui.notify("⏳ LLMレポートを生成中...", type="info", timeout=600000)

    # スピナー表示
    if spinner_container is None:
        spinner_container = ui.column().classes("full-width items-center q-mt-md")
    spinner_container.clear()
    with spinner_container:
        ui.spinner("dots", size="lg", color="cyan")
        ui.label("LLMが解析レポートを生成中です...").classes("text-caption text-grey-5")

    try:
        from backend.llm import get_llm_provider, LLMReportGenerator

        provider_name = state.get("llm_provider", "huggingface")
        provider = get_llm_provider(provider_name)
        generator = LLMReportGenerator(provider)

        result = await run.io_bound(generator.generate_report, ar, state)

        # スピナー非表示
        spinner_container.clear()
        spinner_container.set_visibility(False)

        # 結果を状態に保存
        state[_STATE_KEY] = result

        if result.success:
            ui.notify("✅ LLMレポートを生成しました", type="positive")
        else:
            ui.notify(
                f"⚠️ 生成に失敗: {result.error_message}", type="warning"
            )

        # 再描画
        _refresh_ui(state, result)

    except Exception as e:
        logger.warning(f"[LLMReportTab] 生成失敗: {e}")
        spinner_container.clear()
        spinner_container.set_visibility(False)
        ui.notify(f"❌ エラー: {e}", type="negative")


def _refresh_ui(state: dict, result=None) -> None:
    """UIを再描画する。"""
    refresh_fn = state.get("_refresh_llm_report")
    if refresh_fn:
        refresh_fn()


def _rebuild(
    container, ar, state, dl_md_btn, dl_html_btn
) -> None:
    """レポート表示を再構築する。"""
    result = state.get(_STATE_KEY)
    container.clear()

    if result is None:
        dl_md_btn.disable()
        dl_html_btn.disable()
        return

    dl_md_btn.enable()
    dl_html_btn.enable()
    _render_report_content(container, result)


def _render_report_content(container, result) -> None:
    """レポート内容を表示する。"""
    container.clear()
    with container:
        if not result.success:
            ui.label(f"生成失敗: {result.error_message}").classes(
                "text-negative"
            )
            return

        # タイトル
        with ui.card().classes("glass-card q-pa-md full-width"):
            ui.label(result.title or "解析レポート").classes(
                "text-h6 text-bold q-mb-sm"
            )
            if result.summary:
                ui.markdown(result.summary).classes("text-body1")

        # セクションごとに折りたたみ表示
        for sec in result.sections:
            with ui.expansion(sec.title, icon="description").classes(
                "glass-card full-width q-mt-sm"
            ):
                ui.markdown(sec.content).classes("text-body2")

        # メタ情報
        if result.tokens_used > 0:
            ui.label(f"使用トークン数: {result.tokens_used}").classes(
                "text-caption text-grey-6 q-mt-sm"
            )


def _on_download_md(state: dict) -> None:
    """Markdownダウンロード。"""
    result = state.get(_STATE_KEY)
    if result is None or not result.success:
        ui.notify("レポートがありません", type="warning")
        return
    md = result.to_markdown()
    filename = f"llm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    ui.download(md.encode("utf-8"), filename)
    ui.notify("📝 Markdownレポートをダウンロード", type="positive")


def _on_download_html(state: dict) -> None:
    """HTMLダウンロード。"""
    result = state.get(_STATE_KEY)
    if result is None or not result.success:
        ui.notify("レポートがありません", type="warning")
        return
    html = result.to_html()
    filename = f"llm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    ui.download(html.encode("utf-8"), filename)
    ui.notify("🌐 HTMLレポートをダウンロード", type="positive")


def _on_clear(state: dict, container, dl_md_btn, dl_html_btn) -> None:
    """レポートをクリアする。"""
    state.pop(_STATE_KEY, None)
    container.clear()
    dl_md_btn.disable()
    dl_html_btn.disable()
    ui.notify("🗑️ レポートをクリアしました", type="info")
