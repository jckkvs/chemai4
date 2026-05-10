"""
backend/llm/providers/gguf_provider.py

GGUF/GGML形式の量子化モデルをローカルで実行するLLMプロバイダー。
llama-cpp-python を使用。Bonsai 8B、Mistral、Llama等の量子化モデルに対応。

対応モデル例:
  - prism-ml/Bonsai-8B-gguf        (Q1_0, 1.15GB, 1-bit quantization)
  - lmstudio-community/Qwen2.5-Coder-1.5B-Instruct-GGUF
  - lmstudio-community/Phi-3.5-mini-instruct-GGUF
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable

from backend.llm.provider import LLMProvider, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# ── 設定ファイルパス ─────────────────────────────────────────────────────────
_CONFIG_FILE = Path(__file__).parent.parent.parent.parent / ".gguf_config.json"

# ── GGUFモデルカタログ（HuggingFace Hub上のGGUFモデル）─────────────────────
# CPU最適化: 小さいモデルから順に配置
# 注: lmstudio-community 配下のモデルのみを使用（確実に存在する）
GGUF_MODEL_CATALOG: list[dict] = [
    {
        "id": "lmstudio-community/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
        "label": "Qwen2.5-Coder 1.5B (Q4_K_M, 1.0GB, 推奨)",
        "size_gb": 1.0,
        "description": "コード生成特化。CPUでも実用速度。最小構成。",
        "require_gpu": False,
        "context_length": 32768,
        "cpu_optimized": True,
        "auto_download": True,  # 初回起動時に自動ダウンロード
    },
    {
        "id": "lmstudio-community/Phi-3.5-mini-instruct-GGUF",
        "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "label": "Phi-3.5 Mini (Q4_K_M, ~2.2GB)",
        "size_gb": 2.2,
        "description": "Microsoft製軽量モデル。高速・低メモリ。",
        "require_gpu": False,
        "context_length": 4096,
        "cpu_optimized": True,
    },
    {
        "id": "lmstudio-community/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "label": "Llama 3.2 3B (Q4_K_M, ~2.0GB)",
        "size_gb": 2.0,
        "description": "Meta製3Bモデル。高速・高品質。",
        "require_gpu": False,
        "context_length": 8192,
        "cpu_optimized": True,
    },
    {
        "id": "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "label": "Qwen2.5-Coder 7B (Q4_K_M, ~4.4GB)",
        "size_gb": 4.4,
        "description": "コード生成特化。7Bの量子化版。",
        "require_gpu": False,
        "context_length": 32768,
    },
    {
        "id": "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF",
        "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "label": "Mistral 7B (Q4_K_M, ~4.4GB)",
        "size_gb": 4.4,
        "description": "Mistral 7Bの量子化版。バランス型。",
        "require_gpu": False,
        "context_length": 8192,
    },
    {
        "id": "prism-ml/Bonsai-8B-gguf",
        "file": "Bonsai-8B-Q4_0.gguf",
        "label": "Bonsai 8B (Q4_0, ~5GB, 高精度)",
        "size_gb": 5.0,
        "description": "4-bit量子化版。精度とサイズのバランス型。",
        "require_gpu": False,
        "context_length": 65536,
    },
]


# ── 設定管理 ──────────────────────────────────────────────────────────────────

def load_gguf_config() -> dict:
    """GGUF設定を読み込む（設定ファイル優先、次にSettingsManager）。"""
    config = {
        "model_path": "",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "n_gpu_layers": 0,
        "n_ctx": 2048,
        "n_batch": 512,
    }

    # 設定ファイルから読み込み
    if _CONFIG_FILE.exists():
        try:
            file_config = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            config.update(file_config)
            print(f"[DEBUG] Loaded from .gguf_config.json: model_path={config.get('model_path')}")
        except Exception:
            pass

    # SettingsManagerからも読み込み（ユーザー設定を優先）
    try:
        from backend.config.settings_manager import SettingsManager
        settings = SettingsManager.get_instance()
        model_path = settings.get("llm", "model_path") or settings.get("llm", "local_model_path")
        if model_path:
            print(f"[DEBUG] SettingsManager override: model_path={model_path}")
            config["model_path"] = model_path
    except Exception:
        pass

    print(f"[DEBUG] Final load_gguf_config: model_path={config.get('model_path')}")
    return config


def save_gguf_config(config: dict) -> None:
    """GGUF設定を保存する。"""
    _CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_model_info(model_id: str) -> dict:
    """モデルIDに対応するカタログ情報を返す。"""
    for m in GGUF_MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    return {
        "id": model_id,
        "file": "",
        "label": model_id,
        "size_gb": 0,
        "description": "カスタムGGUFモデル",
        "require_gpu": False,
        "context_length": 4096,
    }


# ── ローカルモデルスキャン ──────────────────────────────────────────────────


def scan_local_models_directory(models_dir: str | Path | None = None) -> list[dict]:
    """
    ローカルモデルディレクトリをスキャンして、利用可能なモデルファイルを動的に検出する。
    親フォルダがわかれば、その配下のモデルファイルも検出できる。

    Returns:
        list[dict]: {"path": str, "name": str, "size_mb": float, "type": str}
    """
    if models_dir is None:
        # SettingsManagerから読み込む
        try:
            from backend.config.settings_manager import SettingsManager
            settings = SettingsManager.get_instance()
            models_dir = settings.get("llm", "local_models_dir", "models/llm")
        except Exception:
            models_dir = "models/llm"

    models_dir = Path(models_dir).expanduser().resolve()
    if not models_dir.exists():
        return []

    found_models = []
    supported_extensions = {
        ".gguf": "GGUF (llama.cpp)",
        ".bin": "GGML/PyTorch",
        ".safetensors": "SafeTensors",
        ".onnx": "ONNX",
        ".pt": "PyTorch",
        ".pth": "PyTorch",
    }

    for fpath in models_dir.rglob("*"):
        if fpath.is_file() and fpath.suffix.lower() in supported_extensions:
            try:
                size_mb = fpath.stat().st_size / (1024 * 1024)
                abs_path = str(fpath.resolve())
                found_models.append({
                    "path": abs_path,
                    "name": fpath.name,
                    "size_mb": round(size_mb, 1),
                    "type": supported_extensions[fpath.suffix.lower()],
                    "parent_dir": str(fpath.parent.resolve()),
                })
            except Exception:
                continue

    # サイズ順にソート
    found_models.sort(key=lambda x: x["size_mb"], reverse=True)
    return found_models


def get_recommended_model_catalog() -> dict[str, list[dict]]:
    """
    有名なローカルLLMモデルのカタログを返す。
    ダウンロード選択肢として使用。
    """
    return {
        "gguf": GGUF_MODEL_CATALOG,
        "famous_models": [
            {
                "id": "prism-ml/Bonsai-8B-gguf",
                "file": "Bonsai-8B-Q4_0.gguf",
                "label": "Bonsai 8B (Q4_0, ~5GB) - 高精度日本語",
                "size_gb": 5.0,
                "description": "日本語に最適化された8Bモデル。Bonsai特化。",
                "require_gpu": False,
                "context_length": 65536,
                "download_url": "https://huggingface.co/prism-ml/Bonsai-8B-gguf",
            },
            {
                "id": "lmstudio-community/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                "file": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
                "label": "Qwen2.5-Coder 1.5B (Q4_K_M, ~1GB) - 推奨",
                "size_gb": 1.0,
                "description": "コード生成特化。CPUでも実用速度。最小構成。",
                "require_gpu": False,
                "context_length": 32768,
                "download_url": "https://huggingface.co/lmstudio-community/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            },
            {
                "id": "lmstudio-community/Llama-3.2-3B-Instruct-GGUF",
                "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "label": "Llama 3.2 3B (Q4_K_M, ~2GB)",
                "size_gb": 2.0,
                "description": "Meta製3Bモデル。高速・高品質。",
                "require_gpu": False,
                "context_length": 8192,
                "download_url": "https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF",
            },
            {
                "id": "lmstudio-community/Phi-3.5-mini-instruct-GGUF",
                "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
                "label": "Phi-3.5 Mini (Q4_K_M, ~2.2GB)",
                "size_gb": 2.2,
                "description": "Microsoft製軽量モデル。高速・低メモリ。",
                "require_gpu": False,
                "context_length": 4096,
                "download_url": "https://huggingface.co/lmstudio-community/Phi-3.5-mini-instruct-GGUF",
            },
            {
                "id": "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF",
                "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                "label": "Mistral 7B (Q4_K_M, ~4.4GB)",
                "size_gb": 4.4,
                "description": "Mistral 7Bの量子化版。バランス型。",
                "require_gpu": False,
                "context_length": 8192,
                "download_url": "https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF",
            },
            {
                "id": "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF",
                "file": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
                "label": "Qwen2.5-Coder 7B (Q4_K_M, ~4.4GB)",
                "size_gb": 4.4,
                "description": "コード生成特化の7B量子化版。",
                "require_gpu": False,
                "context_length": 32768,
                "download_url": "https://huggingface.co/lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF",
            },
        ],
    }


# ── モデル管理 ───────────────────────────────────────────────────────────────

_loaded_models: dict[str, object] = {}  # model_path -> Llama instance
_model_lock = threading.Lock()


def is_gguf_available() -> bool:
    """llama-cpp-pythonが利用可能か確認。"""
    try:
        import llama_cpp  # noqa
        return True
    except ImportError:
        return False


def is_model_downloaded(model_id: str, filename: str) -> bool:
    """GGUFモデルがキャッシュ済みか確認。"""
    try:
        from huggingface_hub import try_to_load_from_cache

        if not filename:
            return False
        snapshot = try_to_load_from_cache(model_id, filename=filename)
        return snapshot is not None and snapshot != ""
    except Exception:
        return False


def download_model_async(
    model_id: str,
    filename: str,
    token: str | None = None,
    on_progress: Callable | None = None,
) -> None:
    """GGUFモデルを非同期でダウンロード。"""

    def _run():
        try:
            from huggingface_hub import hf_hub_download

            logger.info(f"[GGUF] Downloading: {model_id}/{filename}")
            if on_progress:
                on_progress("downloading", 0.0, f"ダウンロード開始: {filename}")

            file_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                token=token or None,
            )

            logger.info(f"[GGUF] Download complete: {file_path}")
            if on_progress:
                on_progress("done", 1.0, f"完了: {file_path}")

        except Exception as e:
            logger.exception(f"[GGUF] Download failed: {e}")
            if on_progress:
                on_progress("error", 0.0, f"エラー: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def load_gguf_model(
    model_path: str,
    n_gpu_layers: int = 0,
    n_ctx: int = 2048,
    n_batch: int = 512,
) -> object:
    """GGUFモデルをロードする（CPU最適化）。"""
    cache_key = f"{model_path}_{n_ctx}_{n_batch}_{n_gpu_layers}"

    with _model_lock:
        if cache_key in _loaded_models:
            return _loaded_models[cache_key]

        try:
            from llama_cpp import Llama

            logger.info(f"[GGUF] Loading model: {model_path}")
            # CPU専用設定
            model = Llama(
                model_path=model_path,
                n_gpu_layers=0,  # CPU専用
                n_ctx=min(n_ctx, 2048),  # CPUメモリ節約
                n_batch=n_batch,
                verbose=False,
                use_mmap=True,  # メモリマップ使用
                use_mlock=False,  # メモリロックしない
            )
            _loaded_models[cache_key] = model
            logger.info(f"[GGUF] Model loaded: {model_path}")
            return model

        except ImportError as e:
            raise ImportError(
                "llama-cpp-python がインストールされていません。"
                "pip install llama-cpp-python を実行してください。"
            ) from e
        except Exception as e:
            raise RuntimeError(f"GGUFモデルのロードに失敗: {e}") from e


# ── GGUFプロバイダー本体 ─────────────────────────────────────────────────────

class GGUFProvider(LLMProvider):
    """
    GGUF形式の量子化モデルでローカル推論するプロバイダー。

    設定は .gguf_config.json で管理。
    初回はダウンロードしてから使う。
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_id: str | None = None,
        filename: str | None = None,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        n_batch: int = 512,
    ) -> None:
        cfg = load_gguf_config()
        # SettingsManagerからも直接読み込む（model_pathを優先）
        if not model_path:
            try:
                from backend.config.settings_manager import SettingsManager
                settings = SettingsManager.get_instance()
                model_path = settings.get("llm", "model_path") or settings.get("llm", "local_model_path")
            except Exception:
                pass
        # CPU最適化のため小さいデフォルト値を使用
        self._model_path = model_path or cfg.get("model_path", "")
        self._model_id = model_id or cfg.get(
            "model_id", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        )
        self._filename = filename or cfg.get("filename", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        self._n_gpu_layers = n_gpu_layers or cfg.get("n_gpu_layers", 0)
        self._n_ctx = n_ctx or cfg.get("n_ctx", 2048)
        self._n_batch = n_batch or cfg.get("n_batch", 512)

    @property
    def name(self) -> str:
        return "gguf"

    @property
    def display_name(self) -> str:
        info = get_model_info(self._model_id)
        return f"GGUF: {info.get('label', self._model_id)}"

    @property
    def description(self) -> str:
        return f"GGUF量子化モデル ({self._model_id})"

    @property
    def is_available(self) -> bool:
        if not is_gguf_available():
            return False
        # モデルパスがあれば利用可能
        return bool(self._model_path) or bool(self._model_id)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """GGUFモデルで推論を実行する。"""
        try:
            from llama_cpp import Llama

            # モデルパスの解決
            model_path = self._resolve_model_path()

            # モデルロード（CPU最適化）
            model: Llama = load_gguf_model(
                model_path,
                n_gpu_layers=0,  # CPU専用
                n_ctx=min(self._n_ctx, 2048, request.max_tokens + 512),
                n_batch=self._n_batch,
            )

            # プロンプト構築
            prompt = self._build_prompt(
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
            )

            # 推論実行
            result = model(
                prompt,
                max_tokens=request.max_tokens,
                temperature=max(request.temperature, 0.01),
                echo=False,
            )

            content = result["choices"][0]["text"].strip()
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            if not tokens_used:
                tokens_used = len(result.get("prompt_tokens", [])) + len(
                    result.get("completion_tokens", [])
                )

            return LLMResponse(
                content=content,
                model=self._model_id or self._model_path,
                tokens_used=tokens_used,
                is_truncated=len(content.split()) >= request.max_tokens - 10,
            )

        except ImportError:
            raise ImportError(
                "llama-cpp-python がインストールされていません。"
                "pip install llama-cpp-python を実行してください。"
            )
        except Exception as e:
            raise RuntimeError(f"GGUF推論エラー: {e}") from e

    def _resolve_model_path(self) -> str:
        """モデルパスを解決（ダウンロード済みファイルを探す）。"""
        if self._model_path and Path(self._model_path).exists():
            return self._model_path

        # HuggingFace Hubキャッシュから探す
        if self._model_id and self._filename:
            try:
                from huggingface_hub import try_to_load_from_cache

                cached = try_to_load_from_cache(
                    repo_id=self._model_id, filename=self._filename
                )
                if cached and isinstance(cached, (str, Path)) and Path(cached).exists():
                    return str(cached)
            except Exception:
                pass

        # モデルIDからファイル名を推測
        if self._model_id and not self._filename:
            info = get_model_info(self._model_id)
            guessed_file = info.get("file", "")
            if guessed_file:
                try:
                    from huggingface_hub import try_to_load_from_cache

                    cached = try_to_load_from_cache(
                        repo_id=self._model_id, filename=guessed_file
                    )
                    if cached and isinstance(cached, (str, Path)) and Path(cached).exists():
                        return str(cached)
                except Exception:
                    pass

        # 詳細なエラーメッセージを作成
        debug_info = []
        if self._model_path:
            debug_info.append(f"model_path='{self._model_path}' (exists: {Path(self._model_path).exists() if self._model_path else False})")
        if self._model_id:
            debug_info.append(f"model_id='{self._model_id}'")
        if self._filename:
            debug_info.append(f"filename='{self._filename}'")

        error_msg = "GGUFモデルが見つかりません。\n"
        error_msg += "\n".join(debug_info)
        error_msg += "\n\n確認事項:\n"
        error_msg += "1. モデルファイルが指定したパスに存在するか\n"
        error_msg += "2. モデルがダウンロード済みか\n"
        error_msg += "3. 設定 → ローカルLLM → モデルフォルダ設定が正しいか"

        raise FileNotFoundError(error_msg)

    @staticmethod
    def _build_prompt(system_prompt: str, user_prompt: str) -> str:
        """LLM用のプロンプトを構築（ChatML形式）。"""
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
