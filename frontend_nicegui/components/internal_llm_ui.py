"""
frontend_nicegui/components/internal_llm_ui.py

HuggingFace / GGUF / ローカルLLM 設定・ダウンロード・推論UIコンポーネント。
「内部AI（ローカルモデル）」タブのコンテンツとして descriptor_plugins_ui.py から呼ばれる。
"""
from __future__ import annotations

import asyncio
import logging

from nicegui import ui

logger = logging.getLogger(__name__)


def render_internal_llm_tab(state: dict) -> None:
    """内部AI（ローカルモデル）設定タブ全体を描画する。"""
    # ── プロバイダー別インポート ─────────────────────────────────────────
    # GGUF プロバイダー
    try:
        from backend.llm.providers.gguf_provider import (
            GGUF_MODEL_CATALOG,
            load_gguf_config,
            save_gguf_config,
            is_model_downloaded as gguf_is_downloaded,
            download_model_async as gguf_download,
        )
        _has_gguf = True
    except ImportError:
        GGUF_MODEL_CATALOG = []
        _has_gguf = False

    # HuggingFace プロバイダー
    try:
        from backend.llm.providers.hf_provider import (
            HF_MODEL_CATALOG,
            load_hf_config,
            save_hf_config,
            get_hf_token,
            is_model_downloaded as hf_is_downloaded,
            download_model_async as hf_download,
            get_download_progress,
        )
        _has_hf = True
    except ImportError:
        HF_MODEL_CATALOG = []
        _has_hf = False

    # デフォルト設定（GGUFを優先）
    if _has_gguf:
        cfg = load_gguf_config()
        _ui_state = {
            "provider": "gguf",
            "token": cfg.get("token", ""),
            "model_id": cfg.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            "filename": cfg.get("filename", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        }
    elif _has_hf:
        cfg = load_hf_config()
        _ui_state = {
            "provider": "huggingface",
            "token": cfg.get("token", ""),
            "model_id": cfg.get("model_id", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        }
    else:
        _ui_state = {
            "provider": "none",
            "token": "",
            "model_id": "",
        }

    _ui_state.setdefault("prompt_input", "")
    _ui_state.setdefault("result_code", "")

    # ── ① プロバイダー選択 ─────────────────────────────────────────
    if _has_gguf or _has_hf:
        with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
            "border:1px solid rgba(99,102,241,0.3); border-radius:8px;"
        ):
            ui.label("① プロバイダー選択").classes("text-subtitle2 text-indigo q-mb-xs")

            provider_options = {}
            if _has_gguf:
                provider_options["gguf"] = "GGUF (Qwen2.5 0.5B等の量子化モデル・CPU最適化)"
            if _has_hf:
                provider_options["huggingface"] = "HuggingFace (Qwen, Phi, Gemma等)"

            if len(provider_options) > 1:
                prov_sel = ui.select(
                    provider_options,
                    value=_ui_state["provider"],
                    label="LLMプロバイダー",
                    on_change=lambda e: _on_provider_change(e.value),
                ).props("dense outlined").classes("full-width q-mb-xs")

    # ── ② HuggingFace トークン設定 ─────────────────────────────────────────
    with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
        "border:1px solid rgba(99,102,241,0.3); border-radius:8px;"
    ):
        ui.label("② HuggingFace トークン設定").classes("text-subtitle2 text-indigo q-mb-xs")
        ui.label(
            "https://huggingface.co/settings/tokens でアクセストークンを発行してください。"
            "Gated Model（Gemma等）を使う場合は Read+Write スコープが必要です。"
        ).classes("text-grey-5 text-caption q-mb-xs")

        with ui.row().classes("items-center q-gutter-sm full-width"):
            tok_input = ui.input(
                "HuggingFace Token (hf_...)",
                value=_ui_state["token"],
                password=True,
                password_toggle_button=True,
                on_change=lambda e: _ui_state.update({"token": e.value}),
            ).props("dense outlined").style("min-width:360px;")

            def _save_token():
                tok = _ui_state["token"].strip()
                if not tok:
                    ui.notify("トークンを入力してください", type="warning")
                    return
                _ui_state["token"] = tok
                if _ui_state["provider"] == "gguf" and _has_gguf:
                    cfg = load_gguf_config()
                    cfg["token"] = tok
                    save_gguf_config(cfg)
                elif _ui_state["provider"] == "huggingface" and _has_hf:
                    cfg = load_hf_config()
                    cfg["token"] = tok
                    save_hf_config(cfg)
                ui.notify("トークンを保存しました", type="positive")
                _update_download_status()

            ui.button("保存", on_click=_save_token).props("unelevated no-caps color=indigo dense")

        ui.link(
            "HuggingFace トークン発行ページを開く",
            "https://huggingface.co/settings/tokens",
            new_tab=True,
        ).classes("text-caption text-indigo q-mt-xs")

    # ── ③ モデル選択 & ダウンロード ─────────────────────────────────────────
    with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
        "border:1px solid rgba(99,102,241,0.3); border-radius:8px;"
    ):
        ui.label("③ モデル選択 & ダウンロード").classes("text-subtitle2 text-indigo q-mb-xs")

        # モデル選択
        def _get_model_options():
            if _ui_state["provider"] == "gguf" and _has_gguf:
                return {m["id"]: f"{m['label']}  ({m['size_gb']}GB)" for m in GGUF_MODEL_CATALOG}
            elif _ui_state["provider"] == "huggingface" and _has_hf:
                return {m["id"]: f"{m['label']}  ({m['size_gb']}GB)" for m in HF_MODEL_CATALOG}
            return {}

        model_sel = ui.select(
            _get_model_options(),
            value=_ui_state["model_id"] or None,
            label="使用するモデル",
            on_change=lambda e: _on_model_change(e.value),
        ).props("dense outlined").classes("full-width q-mb-xs")

        # モデル説明
        desc_label = ui.label("").classes("text-grey-5 text-caption q-mb-xs")
        _update_desc(desc_label, _ui_state["model_id"], _ui_state.get("provider", "gguf"), _has_gguf, _has_hf)

        # ダウンロードステータス
        dl_status_label = ui.label("").classes("text-caption q-mb-xs")
        dl_progress = ui.linear_progress(value=0).classes("full-width q-mb-xs")
        dl_progress.set_visibility(False)

        with ui.row().classes("q-gutter-sm"):
            dl_btn = ui.button(
                "ダウンロード開始",
                on_click=lambda: asyncio.ensure_future(
                    _do_download(_ui_state, dl_btn, dl_status_label, dl_progress)
                ),
            ).props("unelevated no-caps color=teal dense")

            ui.button(
                "キャッシュ状態を確認",
                on_click=lambda: _update_download_status(),
            ).props("flat no-caps color=grey dense")

        def _update_download_status():
            provider = _ui_state["provider"]
            mid = _ui_state["model_id"]
            if provider == "gguf" and _has_gguf:
                filename = _ui_state.get("filename", "")
                if gguf_is_downloaded(mid, filename):
                    dl_status_label.set_text(f"[READY] {mid} はダウンロード済みです")
                    dl_status_label.classes("text-positive", remove="text-amber text-red")
                else:
                    dl_status_label.set_text(f"[未取得] {mid} はまだダウンロードされていません")
                    dl_status_label.classes("text-amber", remove="text-positive text-red")
            elif provider == "huggingface" and _has_hf:
                if hf_is_downloaded(mid):
                    dl_status_label.set_text(f"[READY] {mid} はダウンロード済みです")
                    dl_status_label.classes("text-positive", remove="text-amber text-red")
                else:
                    dl_status_label.set_text(f"[未取得] {mid} はまだダウンロードされていません")
                    dl_status_label.classes("text-amber", remove="text-positive text-red")

        def _on_model_change(new_id: str) -> None:
            _ui_state["model_id"] = new_id
            if _has_gguf and _ui_state["provider"] == "gguf":
                cfg = load_gguf_config()
                cfg["model_id"] = new_id
                # ファイル名も更新
                for m in GGUF_MODEL_CATALOG:
                    if m["id"] == new_id:
                        _ui_state["filename"] = m.get("file", "")
                        cfg["filename"] = m.get("file", "")
                        break
                save_gguf_config(cfg)
            elif _has_hf and _ui_state["provider"] == "huggingface":
                cfg = load_hf_config()
                cfg["model_id"] = new_id
                save_hf_config(cfg)
            _update_desc(desc_label, new_id, _ui_state.get("provider", "gguf"), _has_gguf, _has_hf)
            _update_download_status()

        def _on_provider_change(new_provider: str) -> None:
            _ui_state["provider"] = new_provider
            # プロバイダーに応じてモデル選択を更新
            model_sel.options = _get_model_options()
            if _has_gguf and new_provider == "gguf":
                cfg = load_gguf_config()
                mid = cfg.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
                _ui_state["model_id"] = mid
                model_sel.value = mid
                model_sel.update()
            elif _has_hf and new_provider == "huggingface":
                cfg = load_hf_config()
                mid = cfg.get("model_id", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
                _ui_state["model_id"] = mid
                model_sel.value = mid
                model_sel.update()
            _update_desc(desc_label, _ui_state["model_id"], _ui_state.get("provider", "gguf"), _has_gguf, _has_hf)
            _update_download_status()

        _update_download_status()

    # ── ④ 記述子コード生成（推論） ──────────────────────────────────────────
    with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
        "border:1px solid rgba(99,102,241,0.3); border-radius:8px;"
    ):
        ui.label("④ 記述子コードを生成").classes("text-subtitle2 text-indigo q-mb-xs")
        ui.label(
            "モデルがダウンロード済みの場合、ここで記述子の説明を入力すると"
            "Pythonコードを自動生成します。"
        ).classes("text-grey-5 text-caption q-mb-xs")

        prompt_inp = ui.textarea(
            "作りたい記述子の説明",
            placeholder="例: RDKitでqedスコア（druglikeness）を計算する記述子",
            on_change=lambda e: _ui_state.update({"prompt_input": e.value}),
        ).props("outlined dense rows=3").classes("full-width q-mb-xs")

        gen_row = ui.row().classes("q-gutter-sm items-center")
        with gen_row:
            gen_btn = ui.button(
                "コードを生成（内部LLM）",
                on_click=lambda: asyncio.ensure_future(
                    _do_generate(_ui_state, gen_btn, result_area)
                ),
            ).props("unelevated no-caps color=indigo dense")
            gen_status = ui.label("").classes("text-caption text-grey-5")

        result_area = ui.column().classes("full-width")

    # ── スタイル補足 ─────────────────────────────────────────────────────────
    ui.label(
        "注意: 初回のモデルロードには数分かかる場合があります。"
        "CPUでの推論は生成に時間がかかります（Qwen 1.5B で数十秒〜数分）。"
    ).classes("text-grey-6 text-caption q-mt-xs")


# ── ヘルパー関数 ──────────────────────────────────────────────────────────────

def _update_desc(label, model_id: str, provider: str = "gguf", has_gguf: bool = False, has_hf: bool = False) -> None:
    """モデルの説明を更新する。"""
    try:
        if provider == "gguf" and has_gguf:
            from backend.llm.providers.gguf_provider import get_model_info
            info = get_model_info(model_id)
        elif provider == "huggingface" and has_hf:
            from backend.llm.providers.hf_provider import get_model_info
            info = get_model_info(model_id)
        else:
            info = {}
        label.set_text(info.get("description", ""))
    except Exception:
        label.set_text("")


async def _do_download(ui_state: dict, btn, status_lbl, progress_bar) -> None:
    """モデルダウンロードを実行する。"""
    provider = ui_state.get("provider", "gguf")
    mid = ui_state.get("model_id", "")
    tok = ui_state.get("token", "")

    if not mid:
        ui.notify("モデルを選択してください", type="warning")
        return

    if provider == "gguf" and _has_gguf:
        filename = ui_state.get("filename", "")
        if not filename:
            ui.notify("ファイル名が指定されていません", type="warning")
            return
        if gguf_is_downloaded(mid, filename):
            ui.notify(f"{mid} はすでにダウンロード済みです", type="info")
            return
        btn.disable()
        btn.text = "ダウンロード中..."
        progress_bar.set_visibility(True)
        status_lbl.set_text(f"ダウンロード開始: {mid}")
        try:
            gguf_download(
                model_id=mid,
                filename=filename,
                token=tok or None,
                on_progress=lambda status, frac, msg: status_lbl.set_text(msg),
            )
            # ポーリングで進捗を監視
            for _ in range(3600):
                await asyncio.sleep(3)
                from backend.llm.providers.gguf_provider import is_model_downloaded as check_dl
                if check_dl(mid, filename):
                    progress_bar.set_value(1.0)
                    status_lbl.set_text(f"完了: {mid}")
                    status_lbl.classes("text-positive")
                    ui.notify(f"モデルのダウンロードが完了しました: {mid}", type="positive")
                    break
        except Exception as e:
            status_lbl.set_text(f"エラー: {e}")
            status_lbl.classes("text-red")
            ui.notify(f"ダウンロードエラー: {e}", type="negative")
        finally:
            btn.enable()
            btn.text = "ダウンロード開始"
            progress_bar.set_visibility(False)

    elif provider == "huggingface" and _has_hf:
        if hf_is_downloaded(mid):
            ui.notify(f"{mid} はすでにダウンロード済みです", type="info")
            return
        btn.disable()
        btn.text = "ダウンロード中..."
        progress_bar.set_visibility(True)
        status_lbl.set_text(f"ダウンロード開始: {mid}")

        def _on_progress(prog):
            status_lbl.set_text(prog.message)

        download_model_async(mid, tok, on_progress=_on_progress)
        for _ in range(3600):
            await asyncio.sleep(3)
            prog = get_download_progress(mid)
            if prog.status == "done":
                progress_bar.set_value(1.0)
                status_lbl.set_text(f"完了: {mid}")
                status_lbl.classes("text-positive")
                ui.notify(f"モデルのダウンロードが完了しました: {mid}", type="positive")
                break
            elif prog.status == "error":
                status_lbl.set_text(f"エラー: {prog.message}")
                status_lbl.classes("text-red")
                ui.notify(f"ダウンロードエラー: {prog.message}", type="negative")
                break
        btn.enable()
        btn.text = "ダウンロード開始"
        progress_bar.set_visibility(False)


async def _do_generate(ui_state: dict, btn, result_area) -> None:
    """内部LLMで記述子コードを生成する。"""
    provider = ui_state.get("provider", "gguf")
    mid = ui_state.get("model_id", "")
    prompt = ui_state.get("prompt_input", "").strip()

    if not prompt:
        ui.notify("記述子の説明を入力してください", type="warning")
        return

    if provider == "gguf" and _has_gguf:
        filename = ui_state.get("filename", "")
        if not gguf_is_downloaded(mid, filename):
            ui.notify(f"先にモデルをダウンロードしてください: {mid}", type="warning")
            return
    elif provider == "huggingface" and _has_hf:
        if not hf_is_downloaded(mid):
            ui.notify(f"先にモデルをダウンロードしてください: {mid}", type="warning")
            return

    btn.disable()
    btn.text = "生成中..."

    result_area.clear()
    with result_area:
        ui.label("生成中... しばらくお待ちください（初回はモデルロードで数分かかります）").classes(
            "text-grey-5 text-caption"
        )

    try:
        if provider == "gguf" and _has_gguf:
            from backend.llm.providers.gguf_provider import GGUFProvider

            def _run_inference():
                provider = GGUFProvider(model_id=mid)
                return provider.generate_descriptor_code(prompt)
        elif provider == "huggingface" and _has_hf:
            from backend.llm.providers.hf_provider import HuggingFaceProvider

            def _run_inference():
                provider = HuggingFaceProvider(model_id=mid)
                return provider.generate_descriptor_code(prompt)
        else:
            ui.notify("プロバイダーが利用できません", type="negative")
            return

        from nicegui import run as ng_run
        code = await ng_run.io_bound(_run_inference)

        result_area.clear()
        with result_area:
            ui.label("生成されたコード:").classes("text-subtitle2 q-mb-xs")
            ui.code(code, language="python").classes("full-width q-mb-xs")

            review_area = ui.column().classes("full-width q-mb-xs")

            with ui.row().classes("q-gutter-sm items-center"):
                ui.button(
                    "このコードを検証 & 保存",
                    on_click=lambda c=code: _on_save_generated(c, ui_state),
                ).props("unelevated no-caps color=teal dense")

                ui.button(
                    "🔍 LLMでレビュー",
                    on_click=lambda c=code: asyncio.ensure_future(
                        _do_review(c, ui_state.get("prompt_input", ""), mid, review_area)
                    ),
                ).props("unelevated no-caps color=purple dense")

                def _copy():
                    ui.run_javascript(
                        f"navigator.clipboard.writeText({repr(code)})"
                    )
                    ui.notify("クリップボードにコピーしました", type="positive")

                ui.button("コピー", on_click=_copy).props("flat no-caps color=grey dense")

        ui.notify("コード生成完了", type="positive")

    except Exception as e:
        result_area.clear()
        with result_area:
            ui.label(f"エラー: {e}").classes("text-red text-caption")
        ui.notify(f"生成エラー: {e}", type="negative")
        logger.exception("内部LLM生成エラー")

    finally:
        btn.enable()
        btn.text = "コードを生成（内部LLM）"


async def _do_review(code: str, intent: str, model_id: str, review_area) -> None:
    """LLMでコードをレビューする。"""
    provider_type = _ui_state.get("provider", "gguf")

    review_area.clear()
    with review_area:
        ui.label("レビュー中...").classes("text-grey-5 text-caption")

    try:
        if provider_type == "gguf" and _has_gguf:
            if not gguf_is_downloaded(model_id, _ui_state.get("filename", "")):
                review_area.clear()
                with review_area:
                    ui.label("[静的チェックのみ] モデル未取得のため静的解析で代替します").classes("text-amber text-caption")
                from backend.llm.reviewer import _static_fallback_review
                result = _static_fallback_review(code)
            else:
                from backend.llm.providers.gguf_provider import GGUFProvider
                from backend.llm.reviewer import LLMCodeReviewer

                def _run_review():
                    provider = GGUFProvider(model_id=model_id)
                    reviewer = LLMCodeReviewer(provider)
                    return reviewer.review(code, user_intent=intent)

                from nicegui import run as ng_run
                result = await ng_run.io_bound(_run_review)
        elif provider_type == "huggingface" and _has_hf:
            if not hf_is_downloaded(model_id):
                review_area.clear()
                with review_area:
                    ui.label("[静的チェックのみ] モデル未取得のため静的解析で代替します").classes("text-amber text-caption")
                from backend.llm.reviewer import _static_fallback_review
                result = _static_fallback_review(code)
            else:
                from backend.llm.providers.hf_provider import HuggingFaceProvider
                from backend.llm.reviewer import LLMCodeReviewer

                def _run_review():
                    provider = HuggingFaceProvider(model_id=model_id)
                    reviewer = LLMCodeReviewer(provider)
                    return reviewer.review(code, user_intent=intent)

                from nicegui import run as ng_run
                result = await ng_run.io_bound(_run_review)
        else:
            from backend.llm.reviewer import _static_fallback_review
            result = _static_fallback_review(code)

        review_area.clear()
        with review_area:
            _render_review_result(result)

    except Exception as e:
        review_area.clear()
        with review_area:
            ui.label(f"レビューエラー: {e}").classes("text-red text-caption")
        logger.exception("LLMレビューエラー")


def _render_review_result(result) -> None:
    """CodeReviewResult をUIに表示する。"""
    color_map = {"PASS": "positive", "WARN": "warning", "FAIL": "negative"}
    border_map = {
        "PASS": "rgba(74,222,128,0.5)",
        "WARN": "rgba(251,191,36,0.5)",
        "FAIL": "rgba(239,68,68,0.5)",
    }
    icon_map = {"PASS": "check_circle", "WARN": "warning", "FAIL": "cancel"}

    verdict = result.verdict
    color = border_map.get(verdict, "rgba(99,102,241,0.3)")

    with ui.card().classes("full-width q-pa-sm").style(
        f"border:1px solid {color}; border-radius:8px; margin-top:4px;"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
            ui.icon(icon_map.get(verdict, "info"), color=color_map.get(verdict, "grey")).classes("text-h5")
            ui.label(f"レビュー結果: {verdict}").classes("text-body1 text-bold")
            ui.badge(f"スコア {result.score}/100").props(
                f"color={color_map.get(verdict, 'grey')}"
            )

        ui.label(result.summary).classes("text-caption q-mb-xs")

        if result.issues:
            sev_icon = {"ERROR": "error", "WARN": "warning_amber", "INFO": "info"}
            sev_color = {"ERROR": "red", "WARN": "amber", "INFO": "grey"}
            cat_label = {
                "chemistry": "化学", "robustness": "堅牢性",
                "completeness": "完全性", "performance": "性能",
            }
            for issue in result.issues:
                with ui.row().classes("items-start q-gutter-xs q-mb-xs"):
                    ui.icon(
                        sev_icon.get(issue.severity, "info"),
                        color=sev_color.get(issue.severity, "grey"),
                    ).classes("text-body1")
                    with ui.column().classes("q-gutter-none"):
                        ui.label(
                            f"[{issue.severity}][{cat_label.get(issue.category, issue.category)}] {issue.message}"
                        ).classes("text-caption text-bold")
                        if issue.suggestion:
                            ui.label(f"→ {issue.suggestion}").classes("text-caption text-grey-5")
        else:
            ui.label("問題は検出されませんでした").classes("text-caption text-positive")


def _on_save_generated(code: str, ui_state: dict) -> None:
    """生成コードを検証して保存する（外部AIと同じパイプラインを利用）。"""
    try:
        from backend.llm.generator import (
            LLMDescriptorGenerator,
            _check_security,
            _validate_code_format,
            _strip_code_fences,
        )
        from backend.llm.provider import StubLLMProvider
        from backend.chem.descriptors import get_custom_dir, invalidate_cache
        import re as _re

        clean_code = _strip_code_fences(code)

        warns = _check_security(clean_code)
        blocked = [w for w in warns if "[BLOCKED]" in w]
        if blocked:
            ui.notify(f"セキュリティエラー: {'; '.join(blocked)}", type="negative")
            return

        plugin_name, ferr = _validate_code_format(clean_code)
        if ferr:
            ui.notify(f"形式エラー: {ferr}", type="negative")
            return

        safe_name = _re.sub(r"[^\w]", "_", plugin_name.lower())[:40]
        save_path = get_custom_dir() / f"hf_{safe_name}.py"
        i = 2
        while save_path.exists():
            save_path = get_custom_dir() / f"hf_{safe_name}_{i}.py"
            i += 1

        header = (
            f"# HuggingFace内部LLM生成プラグイン\n"
            f"# このファイルはローカルLLMにより自動生成されました。\n"
            f"# 内容を確認・編集のうえ使用してください。\n\n"
        )
        save_path.write_text(header + clean_code, encoding="utf-8")
        invalidate_cache()
        ui.notify(f"保存しました: {save_path.name}", type="positive")

    except Exception as e:
        ui.notify(f"保存エラー: {e}", type="negative")
        logger.exception("内部LLM生成コード保存エラー")
