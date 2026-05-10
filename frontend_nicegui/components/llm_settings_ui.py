"""
frontend_nicegui/components/llm_settings_ui.py

LLM設定画面 - ローカル・クラウド両対応の統合設定UI。
プロバイダー選択、APIキー設定、モデル選択、動作テストを提供。
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from nicegui import ui, run

logger = logging.getLogger(__name__)

# ── プロバイダーカタログ ─────────────────────────────────────────────────────
PROVIDER_CATALOG = [
    {
        "id": "gguf",
        "label": "ローカルLLM (GGUF)",
        "icon": "memory",
        "description": "llama-cpp-pythonでGGUF量子化モデルをローカル実行",
        "color": "teal",
    },
    {
        "id": "huggingface",
        "label": "ローカルLLM (HuggingFace)",
        "icon": "psychology",
        "description": "transformersでHuggingFaceモデルをローカル実行",
        "color": "orange",
    },
    {
        "id": "openai",
        "label": "ChatGPT (OpenAI)",
        "icon": "smart_toy",
        "description": "OpenAI GPT-4o/4o-mini等のAPIを使用",
        "color": "green",
    },
    {
        "id": "anthropic",
        "label": "Claude (Anthropic)",
        "icon": "auto_awesome",
        "description": "Anthropic Claude APIを使用",
        "color": "purple",
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "icon": "hub",
        "description": "多数のLLM（Claude、GPT、Gemini等）に単一APIでアクセス",
        "color": "blue",
    },
    {
        "id": "custom",
        "label": "カスタム (OpenAI互換)",
        "icon": "settings",
        "description": "OpenAI互換API（ローカルサーバー等）を指定",
        "color": "grey",
    },
]


def render_llm_settings_page() -> None:
    """LLM設定ページ全体を描画する。"""

    # ── プロバイダー別インポート ─────────────────────────────────────────
    _import_status = _try_import_providers()

    # 現在のアクティブプロバイダーを読み込む
    config_dir = Path(__file__).parent.parent.parent
    active_config_file = config_dir / ".llm_active_provider.json"
    if active_config_file.exists():
        try:
            active_provider = json.loads(active_config_file.read_text(encoding="utf-8")).get("provider", "gguf")
        except Exception:
            active_provider = "gguf"
    else:
        active_provider = "gguf"

    # ── ヘッダー ─────────────────────────────────────────────────────────
    with ui.row().classes("items-center q-gutter-md q-mb-md"):
        ui.icon("smart_toy", color="indigo").classes("text-h4")
        with ui.column().classes("q-gutter-none"):
            ui.label("LLM設定").classes("text-h5 q-mb-none")
            ui.label("ローカル・クラウドLLMの設定と管理").classes("text-subtitle2 text-grey-6")

    # ── アクティブプロバイダー表示 ─────────────────────────────────────────
    active_info = next((p for p in PROVIDER_CATALOG if p["id"] == active_provider), None)
    if active_info:
        with ui.card().classes("full-width q-pa-sm q-mb-md").style(
            f"border:2px solid var(--q-{active_info['color']}); border-radius:8px;"
        ):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon(active_info["icon"], color=active_info["color"]).classes("text-h6")
                ui.label(f"現在のプロバイダー: {active_info['label']}").classes("text-subtitle1 text-bold")
                ui.badge("アクティブ", color=active_info["color"]).props("outline")

    # ── プロバイダー選択カード ─────────────────────────────────────────
    ui.label("プロバイダーを選択").classes("text-subtitle1 q-mb-sm")

    with ui.grid(columns=3).classes("full-width q-gutter-md q-mb-lg"):
        for provider in PROVIDER_CATALOG:
            _build_provider_card(provider, active_provider, _import_status)

    # ── 設定詳細エリア ─────────────────────────────────────────────────
    ui.separator().classes("q-my-md")

    # 各プロバイダーの設定UIを格納するコンテナ
    settings_container = ui.column().classes("full-width")

    # 初期表示
    _render_provider_settings(settings_container, active_provider, _import_status)


def _build_provider_card(provider: dict, active_provider: str, import_status: dict) -> None:
    """プロバイダー選択カードを構築する。"""
    is_active = provider["id"] == active_provider
    is_available = import_status.get(provider["id"], True)

    card_style = (
        f"border:2px solid var(--q-{provider['color']}); border-radius:12px; cursor:pointer;"
        if is_active
        else "border:1px solid rgba(0,0,0,0.12); border-radius:12px; cursor:pointer;"
    )
    if not is_available:
        card_style += " opacity:0.5;"

    with ui.card().classes("q-pa-md").style(card_style) as card:
        # アクティブバッジ
        if is_active:
            ui.badge("アクティブ", color=provider["color"]).props("rounded").classes("q-mb-xs")

        # アイコン
        ui.icon(provider["icon"], color=provider["color"]).classes("text-h5 q-mb-xs")

        # ラベル
        ui.label(provider["label"]).classes("text-subtitle1 text-bold q-mb-xs")

        # 説明
        ui.label(provider["description"]).classes("text-caption text-grey-6 q-mb-sm").style(
            "min-height:2.5em;"
        )

        # 状態表示
        if not is_available:
            ui.badge("未インストール", color="grey").props("rounded")
        elif is_active:
            ui.badge("使用中", color=provider["color"]).props("rounded")
        else:
            ui.badge("利用可能", color="green").props("rounded outline")

        # 選択ボタン
        if not is_active:

            def _make_handler(pid=provider["id"]):
                return lambda: _switch_provider(pid)

            ui.button(
                "選択",
                on_click=_make_handler(provider["id"]),
                color=provider["color"],
            ).props("unelevated no-caps dense full-width").bind_visibility_from(
                lambda: is_available
            )


def _switch_provider(provider_id: str) -> None:
    """アクティブプロバイダーを切り替える。"""
    config_dir = Path(__file__).parent.parent.parent
    active_config_file = config_dir / ".llm_active_provider.json"
    active_config_file.write_text(
        json.dumps({"provider": provider_id}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ui.notify(f"プロバイダーを {provider_id} に切り替えました", type="positive")
    ui.navigate.reload()


def _render_provider_settings(container, provider_id: str, import_status: dict) -> None:
    """選択されたプロバイダーの設定UIを描画する。"""
    container.clear()

    with container:
        if provider_id == "gguf":
            _render_gguf_settings()
        elif provider_id == "huggingface":
            _render_hf_settings()
        elif provider_id == "openai":
            _render_openai_settings()
        elif provider_id == "anthropic":
            _render_anthropic_settings()
        elif provider_id == "openrouter":
            _render_openrouter_settings()
        elif provider_id == "custom":
            _render_custom_settings()
        else:
            ui.label("未知のプロバイダーです。").classes("text-negative")


# ── GGUF設定 ──────────────────────────────────────────────────────────────────
def _render_gguf_settings() -> None:
    """GGUFローカルLLM設定UI。"""
    try:
        from backend.llm.providers.gguf_provider import (
            GGUF_MODEL_CATALOG,
            load_gguf_config,
            save_gguf_config,
            is_model_downloaded as gguf_is_downloaded,
            download_model_async as gguf_download,
            get_model_info,
        )

        cfg = load_gguf_config()
    except ImportError:
        ui.label("llama-cpp-python がインストールされていません。").classes("text-negative")
        ui.label("pip install llama-cpp-python を実行してください。").classes("text-caption")
        return

    ui.label("GGUF量子化モデル設定").classes("text-h6 q-mb-sm")

    with ui.card().classes("full-width q-pa-md").style("border-radius:8px;"):
        ui.label("① モデル選択").classes("text-subtitle2 text-bold q-mb-sm")

        model_options = {m["id"]: f"{m['label']} ({m['size_gb']}GB)" for m in GGUF_MODEL_CATALOG}
        current_model = cfg.get("model_id", GGUF_MODEL_CATALOG[0]["id"])

        model_sel = ui.select(
            model_options,
            value=current_model,
            label="使用するモデル",
        ).props("dense outlined").classes("full-width q-mb-sm")

        # モデル説明
        desc_label = ui.label("").classes("text-caption text-grey-6 q-mb-sm")

        def _update_desc():
            info = get_model_info(model_sel.value)
            desc_label.set_text(info.get("description", ""))

        model_sel.on_value_change(_update_desc)
        _update_desc()

        ui.separator().classes("q-my-sm")

        # ダウンロード状態
        ui.label("② モデルのダウンロード").classes("text-subtitle2 text-bold q-mb-sm")

        status_row = ui.row().classes("items-center q-gutter-sm")
        with status_row:
            status_label = ui.label("確認中...").classes("text-caption")

        progress_bar = ui.linear_progress(value=0).classes("full-width q-mb-sm")
        progress_bar.set_visibility(False)

        # HuggingFaceトークン（必要な場合）
        with ui.expansion("HuggingFaceトークン設定（オプション）", icon="key").classes(
            "full-width q-mb-sm"
        ):
            ui.label("一部のGGUFモデルはHuggingFace認証が必要です。").classes(
                "text-caption text-grey-6 q-mb-sm"
            )
            token_input = ui.input(
                "HuggingFace Token (hf_...)",
                value=cfg.get("token", ""),
                password=True,
                password_toggle_button=True,
            ).props("dense outlined").classes("full-width q-mb-sm")

        def _check_status():
            filename = ""
            for m in GGUF_MODEL_CATALOG:
                if m["id"] == model_sel.value:
                    filename = m.get("file", "")
                    break
            if gguf_is_downloaded(model_sel.value, filename):
                status_label.set_text("ダウンロード済み")
                status_label.classes("text-positive", remove="text-amber text-red")
            else:
                status_label.set_text("未ダウンロード")
                status_label.classes("text-amber", remove="text-positive text-red")

        def _download():
            filename = ""
            for m in GGUF_MODEL_CATALOG:
                if m["id"] == model_sel.value:
                    filename = m.get("file", "")
                    break
            if not filename:
                ui.notify("ファイル名が不明です", type="warning")
                return

            progress_bar.set_visibility(True)
            progress_bar.set_value(0.3)

            def _on_progress(status, frac, msg):
                status_label.set_text(msg)
                progress_bar.set_value(frac)

            gguf_download(
                model_id=model_sel.value,
                filename=filename,
                token=token_input.value or None,
                on_progress=_on_progress,
            )
            ui.notify("ダウンロードを開始しました", type="info")

        with ui.row().classes("q-gutter-sm"):
            ui.button("ダウンロード開始", on_click=_download).props(
                "unelevated no-caps color=teal dense"
            )
            ui.button("状態を確認", on_click=_check_status).props("flat no-caps color=grey dense")

        ui.separator().classes("q-my-sm")

        # 詳細設定
        with ui.expansion("詳細設定", icon="tune").classes("full-width q-mb-sm"):
            ui.label("GPUレイヤー数 (0=CPUのみ)").classes("text-caption q-mb-xs")
            gpu_layers = ui.number(
                value=cfg.get("n_gpu_layers", 0),
                min=0,
                max=999,
            ).props("dense outlined").classes("full-width q-mb-sm")

            ui.label("コンテキスト長").classes("text-caption q-mb-xs")
            ctx_size = ui.number(
                value=cfg.get("n_ctx", 4096),
                min=512,
                max=131072,
            ).props("dense outlined").classes("full-width q-mb-sm")

        def _save():
            new_cfg = {
                "model_id": model_sel.value,
                "token": token_input.value or "",
                "n_gpu_layers": int(gpu_layers.value or 0),
                "n_ctx": int(ctx_size.value or 4096),
            }
            # filenameも更新
            for m in GGUF_MODEL_CATALOG:
                if m["id"] == model_sel.value:
                    new_cfg["filename"] = m.get("file", "")
                    break
            save_gguf_config(new_cfg)
            ui.notify("GGUF設定を保存しました", type="positive")

        ui.button("設定を保存", on_click=_save).props(
            "unelevated no-caps color=indigo full-width"
        ).classes("q-mt-sm")

    _check_status()


# ── HuggingFace設定 ───────────────────────────────────────────────────────────
def _render_hf_settings() -> None:
    """HuggingFaceローカルLLM設定UI。"""
    try:
        from backend.llm.providers.hf_provider import (
            HF_MODEL_CATALOG,
            load_hf_config,
            save_hf_config,
            get_hf_token,
            is_model_downloaded as hf_is_downloaded,
            download_model_async,
            get_model_info,
            get_download_progress,
        )

        cfg = load_hf_config()
    except ImportError:
        ui.label("transformers がインストールされていません。").classes("text-negative")
        ui.label("pip install transformers torch を実行してください。").classes("text-caption")
        return

    ui.label("HuggingFaceモデル設定").classes("text-h6 q-mb-sm")

    with ui.card().classes("full-width q-pa-md").style("border-radius:8px;"):
        # HuggingFaceトークン
        ui.label("① HuggingFaceトークン").classes("text-subtitle2 text-bold q-mb-sm")
        ui.label(
            "https://huggingface.co/settings/tokens で発行してください。"
            "Gated Modelを使う場合は認証が必要です。"
        ).classes("text-caption text-grey-6 q-mb-sm")

        token_input = ui.input(
            "HuggingFace Token (hf_...)",
            value=cfg.get("token", ""),
            password=True,
            password_toggle_button=True,
        ).props("dense outlined").classes("full-width q-mb-sm")

        ui.link(
            "HuggingFace トークン発行ページを開く",
            "https://huggingface.co/settings/tokens",
            new_tab=True,
        ).classes("text-caption text-indigo q-mb-sm")

        ui.separator().classes("q-my-sm")

        # モデル選択
        ui.label("② モデル選択").classes("text-subtitle2 text-bold q-mb-sm")

        model_options = {m["id"]: f"{m['label']} ({m['size_gb']}GB)" for m in HF_MODEL_CATALOG}
        current_model = cfg.get("model_id", HF_MODEL_CATALOG[0]["id"])

        model_sel = ui.select(
            model_options,
            value=current_model,
            label="使用するモデル",
        ).props("dense outlined").classes("full-width q-mb-sm")

        desc_label = ui.label("").classes("text-caption text-grey-6 q-mb-sm")

        def _update_desc():
            info = get_model_info(model_sel.value)
            desc_label.set_text(info.get("description", ""))

        model_sel.on_value_change(_update_desc)
        _update_desc()

        ui.separator().classes("q-my-sm")

        # ダウンロード
        ui.label("③ モデルのダウンロード").classes("text-subtitle2 text-bold q-mb-sm")

        status_row = ui.row().classes("items-center q-gutter-sm")
        with status_row:
            status_label = ui.label("確認中...").classes("text-caption")

        progress_bar = ui.linear_progress(value=0).classes("full-width q-mb-sm")
        progress_bar.set_visibility(False)

        def _check_status():
            if hf_is_downloaded(model_sel.value):
                status_label.set_text("ダウンロード済み")
                status_label.classes("text-positive", remove="text-amber text-red")
            else:
                status_label.set_text("未ダウンロード")
                status_label.classes("text-amber", remove="text-positive text-red")

        async def _download():
            progress_bar.set_visibility(True)
            progress_bar.set_value(0.1)
            status_label.set_text("ダウンロード開始...")

            def _on_progress(prog):
                status_label.set_text(prog.message)
                progress_bar.set_value(prog.fraction)

            download_model_async(
                model_id=model_sel.value,
                token=token_input.value or "",
                on_progress=_on_progress,
            )

            # ポーリング
            for _ in range(3600):
                await asyncio.sleep(3)
                prog = get_download_progress(model_sel.value)
                if prog.status == "done":
                    progress_bar.set_value(1.0)
                    status_label.set_text("完了")
                    ui.notify("ダウンロード完了", type="positive")
                    break
                elif prog.status == "error":
                    status_label.set_text(f"エラー: {prog.message}")
                    ui.notify(f"エラー: {prog.message}", type="negative")
                    break

        with ui.row().classes("q-gutter-sm"):
            ui.button(
                "ダウンロード開始",
                on_click=lambda: asyncio.ensure_future(_download()),
            ).props("unelevated no-caps color=orange dense")
            ui.button("状態を確認", on_click=_check_status).props(
                "flat no-caps color=grey dense"
            )

        def _save():
            new_cfg = {
                "token": token_input.value or "",
                "model_id": model_sel.value,
            }
            save_hf_config(new_cfg)
            ui.notify("HuggingFace設定を保存しました", type="positive")

        ui.button("設定を保存", on_click=_save).props(
            "unelevated no-caps color=indigo full-width"
        ).classes("q-mt-sm")

    _check_status()


# ── OpenAI設定 ────────────────────────────────────────────────────────────────
def _render_openai_settings() -> None:
    """OpenAI (ChatGPT) 設定UI。"""
    try:
        from backend.llm.providers.openai_provider import (
            OPENAI_MODEL_CATALOG,
            load_openai_config,
            save_openai_config,
            get_model_info,
        )

        cfg = load_openai_config()
    except ImportError:
        ui.label("openai パッケージがインストールされていません。").classes("text-negative")
        ui.label("pip install openai を実行してください。").classes("text-caption")
        return

    ui.label("ChatGPT (OpenAI) 設定").classes("text-h6 q-mb-sm")

    with ui.card().classes("full-width q-pa-md").style("border-radius:8px;"):
        # APIキー
        ui.label("① APIキー設定").classes("text-subtitle2 text-bold q-mb-sm")
        ui.label(
            "https://platform.openai.com/api-keys でAPIキーを発行してください。"
        ).classes("text-caption text-grey-6 q-mb-sm")

        api_key_input = ui.input(
            "OpenAI API Key (sk-...)",
            value=cfg.get("api_key", ""),
            password=True,
            password_toggle_button=True,
        ).props("dense outlined").classes("full-width q-mb-sm")

        ui.link(
            "OpenAI APIキー発行ページを開く",
            "https://platform.openai.com/api-keys",
            new_tab=True,
        ).classes("text-caption text-green q-mb-sm")

        ui.separator().classes("q-my-sm")

        # モデル選択
        ui.label("② モデル選択").classes("text-subtitle2 text-bold q-mb-sm")

        model_options = {m["id"]: m["label"] for m in OPENAI_MODEL_CATALOG}
        model_sel = ui.select(
            model_options,
            value=cfg.get("model_id", "gpt-4o-mini"),
            label="使用するモデル",
        ).props("dense outlined").classes("full-width q-mb-sm")

        desc_label = ui.label("").classes("text-caption text-grey-6 q-mb-sm")

        def _update_desc():
            info = get_model_info(model_sel.value)
            desc_label.set_text(info.get("description", ""))

        model_sel.on_value_change(_update_desc)
        _update_desc()

        ui.separator().classes("q-my-sm")

        # ベースURL（オプション）
        with ui.expansion("詳細設定（ベースURL等）", icon="settings").classes("full-width q-mb-sm"):
            ui.label("カスタムベースURL（OpenAI互換APIを使用する場合）").classes(
                "text-caption q-mb-xs"
            )
            base_url_input = ui.input(
                "Base URL (オプション)",
                value=cfg.get("base_url", ""),
                placeholder="https://api.openai.com/v1",
            ).props("dense outlined").classes("full-width q-mb-sm")

        def _save():
            new_cfg = {
                "api_key": api_key_input.value or "",
                "model_id": model_sel.value,
                "base_url": base_url_input.value or "",
            }
            save_openai_config(new_cfg)
            ui.notify("OpenAI設定を保存しました", type="positive")

        ui.button("設定を保存", on_click=_save).props(
            "unelevated no-caps color=green full-width"
        ).classes("q-mt-sm")


# ── Anthropic設定 ─────────────────────────────────────────────────────────────
def _render_anthropic_settings() -> None:
    """Anthropic (Claude) 設定UI。"""
    try:
        from backend.llm.providers.anthropic_provider import (
            ANTHROPIC_MODEL_CATALOG,
            load_anthropic_config,
            save_anthropic_config,
            get_model_info,
        )

        cfg = load_anthropic_config()
    except ImportError:
        ui.label("anthropic パッケージがインストールされていません。").classes("text-negative")
        ui.label("pip install anthropic を実行してください。").classes("text-caption")
        return

    ui.label("Claude (Anthropic) 設定").classes("text-h6 q-mb-sm")

    with ui.card().classes("full-width q-pa-md").style("border-radius:8px;"):
        # APIキー
        ui.label("① APIキー設定").classes("text-subtitle2 text-bold q-mb-sm")
        ui.label(
            "https://console.anthropic.com/settings/keys でAPIキーを発行してください。"
        ).classes("text-caption text-grey-6 q-mb-sm")

        api_key_input = ui.input(
            "Anthropic API Key (sk-ant-...)",
            value=cfg.get("api_key", ""),
            password=True,
            password_toggle_button=True,
        ).props("dense outlined").classes("full-width q-mb-sm")

        ui.link(
            "Anthropic APIキー発行ページを開く",
            "https://console.anthropic.com/settings/keys",
            new_tab=True,
        ).classes("text-caption text-purple q-mb-sm")

        ui.separator().classes("q-my-sm")

        # モデル選択
        ui.label("② モデル選択").classes("text-subtitle2 text-bold q-mb-sm")

        model_options = {m["id"]: m["label"] for m in ANTHROPIC_MODEL_CATALOG}
        model_sel = ui.select(
            model_options,
            value=cfg.get("model_id", "claude-sonnet-4-20250514"),
            label="使用するモデル",
        ).props("dense outlined").classes("full-width q-mb-sm")

        desc_label = ui.label("").classes("text-caption text-grey-6 q-mb-sm")

        def _update_desc():
            info = get_model_info(model_sel.value)
            desc_label.set_text(info.get("description", ""))

        model_sel.on_value_change(_update_desc)
        _update_desc()

        ui.separator().classes("q-my-sm")

        # ベースURL（オプション）
        with ui.expansion("詳細設定（ベースURL等）", icon="settings").classes("full-width q-mb-sm"):
            ui.label("カスタムベースURL（プロキシ等を使用する場合）").classes(
                "text-caption q-mb-xs"
            )
            base_url_input = ui.input(
                "Base URL (オプション)",
                value=cfg.get("base_url", ""),
                placeholder="https://api.anthropic.com",
            ).props("dense outlined").classes("full-width q-mb-sm")

        def _save():
            new_cfg = {
                "api_key": api_key_input.value or "",
                "model_id": model_sel.value,
                "base_url": base_url_input.value or "",
            }
            save_anthropic_config(new_cfg)
            ui.notify("Anthropic設定を保存しました", type="positive")

        ui.button("設定を保存", on_click=_save).props(
            "unelevated no-caps color=purple full-width"
        ).classes("q-mt-sm")


# ── OpenRouter設定 ────────────────────────────────────────────────────────────
def _render_openrouter_settings() -> None:
    """OpenRouter 設定UI。"""
    try:
        from backend.llm.providers.openrouter_provider import (
            OPENROUTER_MODEL_CATALOG,
            load_openrouter_config,
            save_openrouter_config,
            get_model_info,
        )

        cfg = load_openrouter_config()
    except ImportError:
        ui.label("openai パッケージがインストールされていません。").classes("text-negative")
        ui.label("pip install openai を実行してください（OpenRouterはOpenAI互換APIを使用）。").classes(
            "text-caption"
        )
        return

    ui.label("OpenRouter 設定").classes("text-h6 q-mb-sm")
    ui.label(
        "OpenRouterを使うと、Claude、GPT、Gemini、Llama等、多数のLLMに単一のAPIでアクセスできます。"
    ).classes("text-caption text-grey-6 q-mb-sm")

    with ui.card().classes("full-width q-pa-md").style("border-radius:8px;"):
        # APIキー
        ui.label("① APIキー設定").classes("text-subtitle2 text-bold q-mb-sm")
        ui.label(
            "https://openrouter.ai/keys でAPIキーを発行してください。"
        ).classes("text-caption text-grey-6 q-mb-sm")

        api_key_input = ui.input(
            "OpenRouter API Key (sk-or-v1-...)",
            value=cfg.get("api_key", ""),
            password=True,
            password_toggle_button=True,
        ).props("dense outlined").classes("full-width q-mb-sm")

        ui.link(
            "OpenRouter APIキー発行ページを開く",
            "https://openrouter.ai/keys",
            new_tab=True,
        ).classes("text-caption text-blue q-mb-sm")

        ui.separator().classes("q-my-sm")

        # モデル選択
        ui.label("② モデル選択").classes("text-subtitle2 text-bold q-mb-sm")

        model_options = {m["id"]: m["label"] for m in OPENROUTER_MODEL_CATALOG}
        model_sel = ui.select(
            model_options,
            value=cfg.get("model_id", "anthropic/claude-sonnet-4-20250514"),
            label="使用するモデル",
        ).props("dense outlined").classes("full-width q-mb-sm")

        desc_label = ui.label("").classes("text-caption text-grey-6 q-mb-sm")

        def _update_desc():
            info = get_model_info(model_sel.value)
            desc_label.set_text(info.get("description", ""))

        model_sel.on_value_change(_update_desc)
        _update_desc()

        ui.separator().classes("q-my-sm")

        # 詳細設定
        with ui.expansion("詳細設定", icon="settings").classes("full-width q-mb-sm"):
            ui.label("サイトURL（オプション）").classes("text-caption q-mb-xs")
            site_url = ui.input(
                "Site URL",
                value=cfg.get("site_url", ""),
                placeholder="https://yourapp.com",
            ).props("dense outlined").classes("full-width q-mb-sm")

            ui.label("サイト名（オプション）").classes("text-caption q-mb-xs")
            site_name = ui.input(
                "Site Name",
                value=cfg.get("site_name", "ChemAI2"),
                placeholder="ChemAI2",
            ).props("dense outlined").classes("full-width q-mb-sm")

        def _save():
            new_cfg = {
                "api_key": api_key_input.value or "",
                "model_id": model_sel.value,
                "base_url": "https://openrouter.ai/api/v1",
                "site_url": site_url.value or "",
                "site_name": site_name.value or "ChemAI2",
            }
            save_openrouter_config(new_cfg)
            ui.notify("OpenRouter設定を保存しました", type="positive")

        ui.button("設定を保存", on_click=_save).props(
            "unelevated no-caps color=blue full-width"
        ).classes("q-mt-sm")


# ── カスタム (OpenAI互換) 設定 ───────────────────────────────────────────────
def _render_custom_settings() -> None:
    """カスタムOpenAI互換API設定UI。"""
    ui.label("カスタム (OpenAI互換) 設定").classes("text-h6 q-mb-sm")
    ui.label(
        "ローカルLLMサーバー（Ollama、LM Studio等）や、"
        "カスタムAPIエンドポイントを指定します。"
    ).classes("text-caption text-grey-6 q-mb-sm")

    # 設定ファイルのパス
    config_dir = Path(__file__).parent.parent.parent
    custom_config_file = config_dir / ".custom_llm_config.json"

    if custom_config_file.exists():
        try:
            cfg = json.loads(custom_config_file.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    else:
        cfg = {}

    with ui.card().classes("full-width q-pa-md").style("border-radius:8px;"):
        ui.label("① エンドポイント設定").classes("text-subtitle2 text-bold q-mb-sm")

        base_url_input = ui.input(
            "Base URL",
            value=cfg.get("base_url", "http://localhost:11434/v1"),
            placeholder="http://localhost:11434/v1",
        ).props("dense outlined").classes("full-width q-mb-sm")

        ui.label("例: Ollama → http://localhost:11434/v1, LM Studio → http://localhost:1234/v1").classes(
            "text-caption text-grey-6 q-mb-sm"
        )

        ui.separator().classes("q-my-sm")

        ui.label("② APIキー（オプション）").classes("text-subtitle2 text-bold q-mb-sm")
        api_key_input = ui.input(
            "API Key (オプション)",
            value=cfg.get("api_key", ""),
            password=True,
            password_toggle_button=True,
        ).props("dense outlined").classes("full-width q-mb-sm")

        ui.separator().classes("q-my-sm")

        ui.label("③ モデル名").classes("text-subtitle2 text-bold q-mb-sm")
        model_input = ui.input(
            "Model ID",
            value=cfg.get("model_id", "llama3.2"),
            placeholder="llama3.2",
        ).props("dense outlined").classes("full-width q-mb-sm")

        def _save():
            new_cfg = {
                "base_url": base_url_input.value or "",
                "api_key": api_key_input.value or "",
                "model_id": model_input.value or "llama3.2",
            }
            custom_config_file.write_text(
                json.dumps(new_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # 環境変数にも反映
            import os

            if new_cfg["api_key"]:
                os.environ["OPENAI_API_KEY"] = new_cfg["api_key"]
            if new_cfg["base_url"]:
                os.environ["OPENAI_BASE_URL"] = new_cfg["base_url"]
            ui.notify("カスタムLLM設定を保存しました", type="positive")

        ui.button("設定を保存", on_click=_save).props(
            "unelevated no-caps color=grey full-width"
        ).classes("q-mt-sm")


# ── ユーティリティ ──────────────────────────────────────────────────────────
def _try_import_providers() -> dict:
    """各プロバイダーのインポート可能性をチェックする。"""
    status = {}

    # GGUF
    try:
        import llama_cpp  # noqa

        status["gguf"] = True
    except ImportError:
        status["gguf"] = False

    # HuggingFace
    try:
        import transformers  # noqa

        status["huggingface"] = True
    except ImportError:
        status["huggingface"] = False

    # OpenAI
    try:
        import openai  # noqa

        status["openai"] = True
    except ImportError:
        status["openai"] = False

    # Anthropic
    try:
        import anthropic  # noqa

        status["anthropic"] = True
    except ImportError:
        status["anthropic"] = False

    # OpenRouter (OpenAI互換)
    status["openrouter"] = status.get("openai", False)

    # Custom
    status["custom"] = True

    return status
