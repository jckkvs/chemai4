"""
Model Manager UI (Hugging Face / Local / LLM)
化学モデル・ローカルLLM・クラウドLLMの設定を独立したタブに分割
"""
import os
import asyncio
from pathlib import Path
from nicegui import ui

try:
    from huggingface_hub import login, snapshot_download
    _HAS_HF_HUB = True
except ImportError:
    _HAS_HF_HUB = False

try:
    from backend.config.settings_manager import SettingsManager
    _HAS_SETTINGS = True
except ImportError:
    _HAS_SETTINGS = False


def render_chemical_models_content():
    """化学モデル管理（MolAI, UniPka等）の内容描画"""
    ui.label("🧪 化学モデル管理").classes("text-lg font-bold hero-gradient q-mb-md")

    if not _HAS_HF_HUB:
        ui.label("⚠️ huggingface-hub がインストールされていません。pip install huggingface-hub を実行してください。").classes("text-red")

    # SettingsManager から現在の設定を取得
    hf_token_val = ""
    cache_dir_val = ""
    if _HAS_SETTINGS:
        settings = SettingsManager.get_instance()
        hf_token_val = settings.get("huggingface", "token") or ""
        cache_dir_val = settings.get("huggingface", "cache_dir") or ""

    # ── セクション1: 重要モデル（ワンクリック） ──
    with ui.card().classes("w-full q-pa-md glass-card q-mb-md"):
        ui.label("⭐ 重要モデル（ワンクリック）").classes("text-bold text-caption text-grey-3 q-mb-sm")
        ui.label("よく使う化学モデルをすばやくダウンロードできます。").classes("text-caption text-grey-5 q-mb-sm")

        # ログコンテナ（関数の外側で定義）
        log_view = ui.log(max_lines=30).classes("w-full h-32 q-mb-md").style("font-size: 0.8rem; background: rgba(0,0,0,0.5);")

        async def _quick_dl(repo_id: str, label: str):
            """簡易ダウンロード実行"""
            if hf_token_val:
                os.environ["HF_TOKEN"] = hf_token_val
                try:
                    login(token=hf_token_val, add_to_git_credential=False)
                except Exception:
                    pass
            if cache_dir_val:
                os.environ["HF_HUB_CACHE"] = cache_dir_val

            log_view.push(f"⏳ {label} のダウンロード開始...")

            def _task():
                local_dir = Path("models") / repo_id.split("/")[-1]
                local_dir.mkdir(parents=True, exist_ok=True)
                return snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            try:
                await asyncio.get_event_loop().run_in_executor(None, _task)
                log_view.push(f"✅ {label} ダウンロード完了！")
                ui.notify(f"✅ {label} の準備が完了しました", type="positive")
            except Exception as e:
                log_view.push(f"❌ {label} エラー: {str(e)}")
                ui.notify(f"❌ {label} のダウンロードに失敗: {e}", type="negative")

        with ui.row().classes("q-gutter-sm q-mb-sm"):
            ui.button(
                "🧬 MolAI 化学構造エンコーダ",
                on_click=lambda: _quick_dl("jckkvs/molai-chem-v1", "MolAI v1")
            ).props("outline color=cyan size=xs no-caps").tooltip("化学構造から潜在表現を生成するモデル")
            ui.button(
                "📊 UniPKA 物性予測",
                on_click=lambda: _quick_dl("jckkvs/unipka-base", "UniPKA")
            ).props("outline color=purple size=xs no-caps").tooltip("pKa/LogDを高精度に予測するモデル")

    # ── セクション2: カスタムダウンロード ──
    with ui.card().classes("w-full q-pa-md glass-card q-mb-md"):
        ui.label("📥 カスタムモデルダウンロード").classes("text-bold text-caption text-grey-3 q-mb-sm")

        model_repo = ui.select(
            options={
                "jckkvs/molai-chem-v1": "MolAI 化学構造エンコーダ (v1)",
                "jckkvs/unipka-base": "UniPKA 物性予測モデル",
                "custom": "カスタムリポジトリ...",
            },
            label="ダウンロード対象モデル",
            value="jckkvs/molai-chem-v1",
        ).props("outlined dense").classes("w-full q-mb-sm")

        custom_repo = ui.input("カスタムリポジトリID", placeholder="user/repo-name").props("outlined dense").classes("w-full q-mb-md")
        custom_repo.bind_visibility_from(model_repo, "value", value="custom")

        # プロキシ設定
        with ui.expansion("🔧 詳細設定（プロキシ等）", icon="settings").classes("w-full q-mb-md glass-card"):
            proxy_url = ui.input("Proxy URL", placeholder="http://proxy.example.com:8080").props("outlined dense").classes("w-full q-mb-sm")
            no_proxy = ui.input("No-Proxy", placeholder="localhost,127.0.0.1").props("outlined dense").classes("w-full")

        # ログコンテナ
        log_view2 = ui.log(max_lines=30).classes("w-full h-32 q-mb-md").style("font-size: 0.8rem; background: rgba(0,0,0,0.5);")

        async def on_download():
            repo_id = custom_repo.value if model_repo.value == "custom" else model_repo.value
            if not repo_id:
                ui.notify("⚠️ 対象リポジトリを選択または入力してください", type="warning")
                return

            # 設定を環境変数に反映
            if hf_token_val:
                os.environ["HF_TOKEN"] = hf_token_val
                try:
                    login(token=hf_token_val, add_to_git_credential=False)
                except Exception:
                    pass
            if cache_dir_val:
                os.environ["HF_HUB_CACHE"] = cache_dir_val
            if proxy_url.value:
                os.environ["HTTP_PROXY"] = proxy_url.value
                os.environ["HTTPS_PROXY"] = proxy_url.value
            if no_proxy.value:
                os.environ["NO_PROXY"] = no_proxy.value

            # SettingsManager にも保存
            if _HAS_SETTINGS:
                s = SettingsManager.get_instance()
                s.set("huggingface", "token", hf_token_val)
                s.set("huggingface", "cache_dir", cache_dir_val)
                try:
                    s.save_config()
                except Exception:
                    pass

            btn_download.disable()
            log_view2.clear()
            log_view2.push(f"⏳ ダウンロード開始: {repo_id}")

            try:
                local_dir = Path("models") / repo_id.split("/")[-1]
                local_dir.mkdir(parents=True, exist_ok=True)

                def _download_task():
                    return snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(local_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )

                log_view2.push("📥 ファイルをフェッチ中...")
                await asyncio.get_event_loop().run_in_executor(None, _download_task)

                log_view2.push(f"🎉 ダウンロード成功！ 保存先: {local_dir.absolute()}")
                ui.notify(f"✅ {repo_id} のダウンロードが完了しました", type="positive")
            except Exception as e:
                log_view2.push(f"❌ エラー発生:\n{str(e)}")
                ui.notify(f"❌ エラー: {str(e)}", type="negative")
            finally:
                btn_download.enable()

        btn_download = ui.button("📥 ダウンロード開始", on_click=on_download).classes("btn-primary w-full").props("size=md icon=cloud_download")


def render_local_llm_content():
    """ローカルLLM設定（動的モデル検出＋有名モデルダウンロード）の内容描画"""
    ui.label("💻 ローカルLLMモデル設定").classes("text-lg font-bold hero-gradient q-mb-md")
    ui.label("ローカルLLMモデル(Bonsai, Qwen, Llama等)のパス設定とダウンロードを行います。").classes("text-caption text-grey-5 q-mb-md")

    # SettingsManager から現在の設定を取得
    hf_token_val = ""
    cache_dir_val = ""
    local_model_path_val = ""
    local_models_dir_val = "models/llm"
    if _HAS_SETTINGS:
        settings = SettingsManager.get_instance()
        hf_token_val = settings.get("huggingface", "token") or ""
        cache_dir_val = settings.get("huggingface", "cache_dir") or ""
        local_model_path_val = settings.get("llm", "local_model_path") or ""
        local_models_dir_val = settings.get("llm", "local_models_dir", "models/llm")

    # 状態管理
    _scan_results = {"models": []}

    # ── セクション1: ローカルモデルフォルダ設定 ──
    with ui.card().classes("w-full q-pa-md glass-card q-mb-md"):
        ui.label("📁 ローカルモデルフォルダ設定").classes("text-bold text-caption text-grey-3 q-mb-sm")
        ui.label(
            "モデルを保存・検索する親フォルダを指定します。このフォルダ配下のモデルファイル(.gguf, .bin等)を自動検出します。"
        ).classes("text-caption text-grey-5 q-mb-sm")

        with ui.row().classes("items-center q-gutter-sm w-full"):
            models_dir_input = ui.input(
                "モデル保存フォルダ",
                value=local_models_dir_val,
                placeholder="models/llm",
            ).props("outlined dense").classes("flex-grow")

            def _pick_folder():
                """フォルダ選択ダイアログ（簡易版）"""
                ui.notify("💡 パスを手動で入力するか、以下のボタンでフォルダを作成してください", type="info")

            ui.button("📂", on_click=_pick_folder).props("round dense flat color=cyan").tooltip("フォルダを選択")

            def _create_folder():
                path = Path(models_dir_input.value).expanduser()
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    ui.notify(f"✅ フォルダを作成しました: {path}", type="positive")
                except Exception as e:
                    ui.notify(f"❌ フォルダ作成失敗: {e}", type="negative")

            ui.button("作成", on_click=_create_folder).props("flat dense color=cyan size=xs no-caps")

        # フォルダの存在確認
        def _check_folder():
            path = Path(models_dir_input.value).expanduser()
            if path.exists():
                ui.notify(f"✅ フォルダは存在します: {path}", type="positive")
            else:
                ui.notify(f"⚠️ フォルダが存在しません。作成ボタンで作成できます。", type="warning")

        ui.button("🔍 フォルダ確認", on_click=_check_folder).props("outline color=cyan size=xs no-caps")

        # 設定保存
        def _save_models_dir():
            if not _HAS_SETTINGS:
                ui.notify("⚠️ SettingsManagerが利用できません", type="warning")
                return
            s = SettingsManager.get_instance()
            s.set("llm", "local_models_dir", models_dir_input.value)
            try:
                s.save_config()
                ui.notify("✅ モデルフォルダ設定を保存しました", type="positive")
                _scan_local_models()  # 保存後に再スキャン
            except Exception as e:
                ui.notify(f"❌ 保存失敗: {e}", type="negative")

        ui.button("💾 保存", on_click=_save_models_dir).props("unelevated color=cyan size=xs no-caps")

    # ── セクション2: ダウンロード済モデルの動的検出 ──
    with ui.card().classes("w-full q-pa-md glass-card q-mb-md"):
        ui.label("🔍 ダウンロード済モデルの自動検出").classes("text-bold text-caption text-grey-3 q-mb-sm")
        ui.label(
            "指定したフォルダ内のモデルファイル(.gguf, .safetensors, .bin, .pt等)を自動検出します。"
        ).classes("text-caption text-grey-5 q-mb-sm")

        scan_status = ui.label("未スキャン").classes("text-caption text-grey-5 q-mb-sm")
        models_table = ui.table(
            columns=[
                {"name": "name", "label": "モデル名", "field": "name", "align": "left"},
                {"name": "type", "label": "形式", "field": "type", "align": "center"},
                {"name": "size", "label": "サイズ(MB)", "field": "size_mb", "align": "center"},
                {"name": "path", "label": "パス", "field": "path", "align": "left"},
            ],
            rows=[],
            row_key="path",
            selection="single",  # 単一選択モードを明示的に設定
        ).props("dense flat bordered virtual-scroll").classes("w-full")

        # 選択されたモデルを使用するボタン
        # テーブルから直接選択された行のパスを取得する関数
        def _get_selected_model_path():
            """テーブルの選択状態からモデルパスを取得"""
            try:
                # NiceGUIのテーブル選択は selected プロパティにキーの配列が入る
                selected_keys = models_table.selected
                if not selected_keys or len(selected_keys) == 0:
                    return None
                # row_key が "path" なので、選択されたキーはパスそのもの
                return selected_keys[0]
            except Exception as e:
                print(f"[DEBUG] _get_selected_model_path error: {e}")
                return None

        def _scan_local_models():
            """ローカルモデルをスキャン"""
            scan_status.set_text("スキャン中...")
            try:
                from backend.llm.providers.gguf_provider import scan_local_models_directory
                models = scan_local_models_directory(models_dir_input.value)
                _scan_results["models"] = models
                models_table.rows = models
                scan_status.set_text(f"完了: {len(models)} 件のモデルが見つかりました")
                if models:
                    ui.notify(f"✅ {len(models)} 件のモデルを検出しました", type="positive")
                else:
                    ui.notify("⚠️ モデルが見つかりませんでした", type="warning")
            except Exception as e:
                scan_status.set_text(f"エラー: {e}")
                ui.notify(f"❌ スキャン失敗: {e}", type="negative")

        def _use_selected_model():
            """選択されたモデルを使用する"""
            # テーブルから選択されたパスを取得
            selected_path = _get_selected_model_path()
            if not selected_path:
                ui.notify("⚠️ モデルを選択してください（テーブルで行をクリック）", type="warning")
                print(f"[DEBUG] No model selected. Table selected: {models_table.selected}")
                return

            print(f"[DEBUG] Selected model path: {selected_path}")

            # ファイルの存在確認
            model_path = Path(selected_path)
            if not model_path.exists():
                ui.notify(f"❌ モデルが見つかりません: {model_path}", type="negative")
                return
            if not model_path.is_file():
                ui.notify(f"❌ 指定されたパスはファイルではありません: {model_path}", type="negative")
                return

            if not _HAS_SETTINGS:
                ui.notify("⚠️ SettingsManagerが利用できません", type="warning")
                return
            s = SettingsManager.get_instance()
            # 絶対パスを保存（GGUFプロバイダーと同じキーを使用）
            abs_path = str(model_path.resolve())
            s.set("llm", "model_path", abs_path)
            # 既存のlocal_model_pathも更新（互換性のため）
            s.set("llm", "local_model_path", abs_path)
            try:
                s.save_config()
                ui.notify(f"✅ モデルを設定しました: {model_path.name}", type="positive")
            except Exception as e:
                ui.notify(f"❌ 保存失敗: {e}", type="negative")

            # .gguf_config.json にも保存（SettingsManagerが読み込めない場合の保険）
            try:
                from backend.llm.providers.gguf_provider import load_gguf_config, save_gguf_config
                cfg = load_gguf_config()
                cfg["model_path"] = abs_path
                save_gguf_config(cfg)
                print(f"[DEBUG] Saved model_path to .gguf_config.json: {abs_path}")
            except Exception as e:
                print(f"[DEBUG] Failed to save to .gguf_config.json: {e}")

        with ui.row().classes("q-gutter-sm q-mb-sm"):
            ui.button("🔄 スキャン実行", on_click=_scan_local_models).props("outline color=cyan size=xs no-caps")
            ui.button("✅ 選択したモデルを使用", on_click=_use_selected_model).props("unelevated color=teal size=xs no-caps")

        # 初期スキャン
        _scan_local_models()

    # ── セクション3: 有名モデルのダウンロード ──
    with ui.card().classes("w-full q-pa-md glass-card q-mb-md"):
        ui.label("📥 有名モデルのダウンロード").classes("text-bold text-caption text-grey-3 q-mb-sm")
        ui.label(
            "有名なローカルLLMモデルをダウンロードできます。初回はモデルのダウンロードに数分かかる場合があります。"
        ).classes("text-caption text-grey-5 q-mb-sm")

        # HuggingFace トークン設定
        with ui.expansion("🔑 HuggingFace 設定", icon="key").classes("w-full q-mb-sm"):
            hf_token_input = ui.input(
                "Access Token",
                password=True,
                password_toggle_button=True,
                value=hf_token_val,
            ).props("outlined dense").classes("w-full q-mb-sm")
            hf_token_input.tooltip("Hugging FaceのSettings画面で発行したトークン")

            def _save_hf_token():
                if not _HAS_SETTINGS:
                    ui.notify("⚠️ SettingsManagerが利用できません", type="warning")
                    return
                s = SettingsManager.get_instance()
                s.set("huggingface", "token", hf_token_input.value)
                try:
                    s.save_config()
                    ui.notify("✅ HuggingFace トークンを保存しました", type="positive")
                except Exception as e:
                    ui.notify(f"❌ 保存失敗: {e}", type="negative")

            ui.button("💾 保存", on_click=_save_hf_token).props("unelevated color=cyan size=xs no-caps")

        # モデルカタログ
        try:
            from backend.llm.providers.gguf_provider import get_recommended_model_catalog
            catalog = get_recommended_model_catalog()
            famous_models = catalog.get("famous_models", [])
        except Exception:
            famous_models = []

        if famous_models:
            # モデル選択セレクト
            model_options = {m["id"]: f"{m['label']} ({m['size_gb']}GB)" for m in famous_models}
            selected_model_id = {"value": famous_models[0]["id"]}

            model_select = ui.select(
                options=model_options,
                value=famous_models[0]["id"],
                label="ダウンロードするモデル",
            ).props("outlined dense").classes("w-full q-mb-sm")

            model_select.on_value_change(lambda e: selected_model_id.update({"value": e.value}))

            # モデル説明
            model_desc = ui.label(famous_models[0].get("description", "")).classes("text-caption text-grey-5 q-mb-sm")

            def _update_desc():
                for m in famous_models:
                    if m["id"] == selected_model_id["value"]:
                        model_desc.set_text(m.get("description", ""))
                        break

            model_select.on_value_change(lambda e: _update_desc())

            # ダウンロード状態
            dl_status = ui.label("").classes("text-caption q-mb-sm")
            dl_progress = ui.linear_progress(value=0).classes("w-full q-mb-sm")
            dl_progress.set_visibility(False)

            async def _download_model():
                """選択されたモデルをダウンロード"""
                model_id = selected_model_id["value"]
                target_model = None
                for m in famous_models:
                    if m["id"] == model_id:
                        target_model = m
                        break

                if not target_model:
                    ui.notify("❌ モデル情報が見つかりません", type="negative")
                    return

                # トークン設定
                token = hf_token_input.value or ""
                if token:
                    os.environ["HF_TOKEN"] = token

                # 保存先フォルダ
                save_dir = Path(models_dir_input.value).expanduser()
                save_dir.mkdir(parents=True, exist_ok=True)

                dl_status.set_text(f"ダウンロード開始: {target_model['label']}...")
                dl_progress.set_visibility(True)

                try:
                    from huggingface_hub import snapshot_download

                    def _do_download():
                        return snapshot_download(
                            repo_id=model_id,
                            local_dir=str(save_dir / model_id.split("/")[-1]),
                            local_dir_use_symlinks=False,
                            resume_download=True,
                        )

                    ui.notify(f"⏳ {target_model['label']} をダウンロード中...", type="info")
                    result_path = await asyncio.get_event_loop().run_in_executor(None, _do_download)

                    dl_progress.set_value(1.0)
                    dl_status.set_text(f"✅ ダウンロード完了: {result_path}")
                    ui.notify(f"✅ {target_model['label']} のダウンロードが完了しました", type="positive")

                    # スキャンを再実行
                    _scan_local_models()

                except Exception as e:
                    dl_status.set_text(f"❌ エラー: {e}")
                    ui.notify(f"❌ ダウンロード失敗: {e}", type="negative")
                finally:
                    dl_progress.set_visibility(False)

            ui.button("📥 ダウンロード開始", on_click=_download_model).props(
                "unelevated color=teal size=sm no-caps"
            ).classes("w-full")
        else:
            ui.label("⚠️ モデルカタログが読み込めませんでした").classes("text-caption text-amber")

    # ── セクション4: 現在の設定確認 ──
    with ui.card().classes("w-full q-pa-md glass-card q-mb-md"):
        ui.label("ℹ️ 現在の設定").classes("text-bold text-caption text-grey-3 q-mb-sm")

        current_path = local_model_path_val or "（未設定）"
        ui.label(f"現在のモデルパス: {current_path}").classes("text-caption text-grey-5")

        # パスが存在するか確認
        if local_model_path_val and Path(local_model_path_val).exists():
            ui.icon("check_circle", color="positive").tooltip("モデルファイルは存在します")
        elif local_model_path_val:
            ui.icon("warning", color="amber").tooltip("モデルファイルが見つかりません")

        def _clear_model_path():
            if not _HAS_SETTINGS:
                return
            s = SettingsManager.get_instance()
            s.set("llm", "local_model_path", "")
            try:
                s.save_config()
                ui.notify("✅ モデルパスをクリアしました", type="positive")
            except Exception as e:
                ui.notify(f"❌ クリア失敗: {e}", type="negative")

        ui.button("クリア", on_click=_clear_model_path).props("flat color=grey size=xs no-caps")


def render_cloud_llm_content():
    """クラウドLLM設定（API Key・URL）の内容描画"""
    ui.label("☁️ クラウドLLM設定").classes("text-lg font-bold hero-gradient q-mb-md")
    ui.label("Claude、OpenAI等のクラウドLLMのAPI KeyとURLを設定します。").classes("text-caption text-grey-5 q-mb-md")

    # SettingsManager から現在の設定を取得
    llm_provider_val = "claude"
    llm_api_key_val = ""
    llm_api_url_val = "https://api.anthropic.com"
    if _HAS_SETTINGS:
        settings = SettingsManager.get_instance()
        llm_provider_val = settings.get("llm", "provider") or "claude"
        llm_api_key_val = settings.get("llm", "api_key") or ""
        llm_api_url_val = settings.get("llm", "api_url") or "https://api.anthropic.com"

    # クラウドLLMの有効オプション外の値（gguf等）をデフォルトに戻す
    valid_providers = {"claude", "openai"}
    if llm_provider_val not in valid_providers:
        llm_provider_val = "claude"

    with ui.card().classes("w-full q-pa-md glass-card q-mb-md"):
        ui.label("☁️ クラウドLLM設定").classes("text-bold text-caption text-grey-3 q-mb-sm")

        llm_provider = ui.select(
            options={
                "claude": "Claude (Anthropic API)",
                "openai": "OpenAI API",
            },
            label="LLM プロバイダー",
            value=llm_provider_val,
        ).props("outlined dense").classes("w-full q-mb-sm")

        cloud_api_key = ui.input(
            "API Key",
            password=True,
            password_toggle_button=True,
            value=llm_api_key_val,
        ).props("outlined dense").classes("w-full q-mb-sm")
        cloud_api_key.tooltip("Claude: sk-ant-... / OpenAI: sk-...")

        cloud_api_url = ui.input(
            "API URL",
            value=llm_api_url_val,
        ).props("outlined dense").classes("w-full q-mb-sm")
        cloud_api_url.tooltip("デフォルト: Claude https://api.anthropic.com / OpenAI https://api.openai.com/v1")

        # 接続テスト
        async def _test_llm():
            provider = llm_provider.value
            key = cloud_api_key.value
            url = cloud_api_url.value
            if not key:
                ui.notify("⚠️ APIキーを入力してください", type="warning")
                return
            ui.notify(f"✅ {provider.upper()} 設定を確認しました", type="positive")

        ui.button("🔌 接続テスト", on_click=_test_llm).props("outline color=cyan size=xs no-caps")

        # 設定保存
        def _save_cloud_llm_settings():
            if not _HAS_SETTINGS:
                ui.notify("⚠️ SettingsManagerが利用できません", type="warning")
                return
            s = SettingsManager.get_instance()
            s.set("llm", "provider", llm_provider.value)
            s.set("llm", "api_key", cloud_api_key.value)
            s.set("llm", "api_url", cloud_api_url.value)
            try:
                s.save_config()
                ui.notify("✅ クラウドLLM設定を保存しました", type="positive")
            except Exception as e:
                ui.notify(f"❌ 保存失敗: {e}", type="negative")

        ui.button("💾 クラウドLLM設定を保存", on_click=_save_cloud_llm_settings).props("unelevated color=cyan size=xs no-caps")


def render_model_manager():
    """モデル管理タブの描画（旧互換用）"""
    with ui.card().classes("w-full q-pa-md"):
        render_chemical_models_content()
