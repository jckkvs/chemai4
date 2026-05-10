"""
backend/llm/providers/hf_provider.py

HuggingFace Hub から量子化モデルをダウンロードして
ローカルで推論を実行するLLMプロバイダー。

対応モデル（コード生成特化 / CPUで動作可能な小型モデル優先）:
  - Qwen/Qwen2.5-Coder-1.5B-Instruct   (~3GB, 高速)
  - Qwen/Qwen2.5-Coder-7B-Instruct     (~14GB, 高品質)
  - microsoft/phi-4-mini-instruct       (~8GB, バランス)
  - google/gemma-3-1b-it                (~2GB, 超軽量)
  - ibm-granite/granite-3.1-2b-instruct (~4GB, コード特化)
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from backend.llm.provider import LLMProvider, LLMProviderError, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# ── 設定ファイルパス ─────────────────────────────────────────────────────────
# プロジェクトルートの .hf_config.json に保存
_CONFIG_FILE = Path(__file__).parent.parent.parent.parent / ".hf_config.json"

# ── 推奨モデルカタログ ────────────────────────────────────────────────────────
# CPU最適化: 小さいモデルから順に配置
HF_MODEL_CATALOG: list[dict] = [
    {
        "id": "google/gemma-3-1b-it",
        "label": "Gemma 3 1B（超軽量・推奨）",
        "size_gb": 2.0,
        "description": "Google製超軽量モデル。低スペックPCでも動作。最初の選択肢。",
        "require_gpu": False,
        "chat_template": "gemma",
        "cpu_optimized": True,
    },
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B（最小・超軽量）",
        "size_gb": 1.0,
        "description": "Qwen最小モデル。最も軽いが性能は限定的。",
        "require_gpu": False,
        "chat_template": "chatml",
        "cpu_optimized": True,
    },
    {
        "id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "label": "Qwen2.5-Coder 1.5B（バランス型）",
        "size_gb": 3.0,
        "description": "コード生成特化。CPUでも実用速度。",
        "require_gpu": False,
        "chat_template": "chatml",
        "cpu_optimized": True,
    },
    {
        "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "label": "Qwen2.5-Coder 7B（高品質）",
        "size_gb": 14.0,
        "description": "上位モデル。高品質なコードを生成。RAM 16GB以上推奨。",
        "require_gpu": False,
        "chat_template": "chatml",
    },
    {
        "id": "microsoft/Phi-4-mini-instruct",
        "label": "Phi-4 Mini（バランス型）",
        "size_gb": 8.0,
        "description": "Microsoft製。推論品質とサイズのバランスが良い。",
        "require_gpu": False,
        "chat_template": "phi",
    },
    {
        "id": "ibm-granite/granite-3.3-2b-instruct",
        "label": "IBM Granite 3.3 2B（コード特化）",
        "size_gb": 4.0,
        "description": "IBM製コード生成モデル。科学技術分野に強い。",
        "require_gpu": False,
        "chat_template": "granite",
    },
]


# ── 設定管理 ──────────────────────────────────────────────────────────────────

def load_hf_config() -> dict:
    """HF設定を読み込む。"""
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    # デフォルトはCPUで最も軽量なモデル
    return {"token": "", "model_id": "google/gemma-3-1b-it"}


def save_hf_config(config: dict) -> None:
    """HF設定を保存する。"""
    _CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    # 環境変数にも反映
    if config.get("token"):
        os.environ["HUGGINGFACE_TOKEN"] = config["token"]
        os.environ["HF_TOKEN"] = config["token"]


def get_hf_token() -> str:
    """HuggingFaceトークンを取得（設定ファイル > 環境変数の順）。"""
    cfg = load_hf_config()
    return (
        cfg.get("token")
        or os.environ.get("HF_TOKEN", "")
        or os.environ.get("HUGGINGFACE_TOKEN", "")
    )


def get_model_info(model_id: str) -> dict:
    """モデルIDに対応するカタログ情報を返す。"""
    for m in HF_MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    return {"id": model_id, "label": model_id, "size_gb": 0, "description": "カスタムモデル"}


# ── ダウンロード管理 ──────────────────────────────────────────────────────────

@dataclass
class DownloadProgress:
    """ダウンロード進行状況。"""
    model_id: str
    status: str = "idle"       # idle / downloading / done / error
    message: str = ""
    fraction: float = 0.0      # 0.0 ~ 1.0


_download_progress: dict[str, DownloadProgress] = {}
_loaded_models: dict[str, tuple] = {}  # model_id -> (model, tokenizer)
_model_lock = threading.Lock()


def get_download_progress(model_id: str) -> DownloadProgress:
    if model_id not in _download_progress:
        _download_progress[model_id] = DownloadProgress(model_id=model_id)
    return _download_progress[model_id]


def is_model_downloaded(model_id: str) -> bool:
    """モデルがキャッシュ済みかどうかを確認する。"""
    try:
        from huggingface_hub import try_to_load_from_cache
        snapshot = try_to_load_from_cache(model_id, filename="config.json")
        return snapshot is not None and snapshot != ""
    except Exception:
        return False


def download_model_async(
    model_id: str,
    token: str,
    on_progress: Callable[[DownloadProgress], None] | None = None,
) -> None:
    """モデルを非同期でダウンロードする。"""
    prog = get_download_progress(model_id)
    prog.status = "downloading"
    prog.fraction = 0.0
    prog.message = "ダウンロード開始..."

    def _run():
        try:
            from huggingface_hub import snapshot_download

            logger.info(f"[HF] Downloading model: {model_id}")
            prog.message = f"{model_id} をダウンロード中..."
            if on_progress:
                on_progress(prog)

            # トークンを設定
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGINGFACE_TOKEN"] = token

            # ダウンロード実行（ignore_patterns でモデル以外を除外）
            snapshot_download(
                repo_id=model_id,
                token=token or None,
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
            )

            prog.status = "done"
            prog.fraction = 1.0
            prog.message = f"ダウンロード完了: {model_id}"
            logger.info(f"[HF] Download complete: {model_id}")

        except Exception as e:
            prog.status = "error"
            prog.message = f"エラー: {e}"
            logger.exception(f"[HF] Download failed: {model_id}")

        if on_progress:
            on_progress(prog)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def load_model(model_id: str, token: str | None = None) -> tuple:
    """モデルとトークナイザーをロードする（キャッシュ済み）。"""
    with _model_lock:
        if model_id in _loaded_models:
            return _loaded_models[model_id]

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"[HF] Loading model: {model_id}")
            hf_token = token or get_hf_token() or None

            # デバイス設定（CUDA優先、なければCPU）
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[HF] Using device: {device}")

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token,
                trust_remote_code=True,
            )

            # CPUの場合はfloat32、GPUならauto
            dtype = "auto" if device == "cuda" else None
            model_kwargs: dict = {
                "token": hf_token,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if dtype:
                import torch as _torch
                model_kwargs["torch_dtype"] = _torch.float32

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            model.eval()

            if device == "cuda":
                model = model.to(device)

            _loaded_models[model_id] = (model, tokenizer)
            logger.info(f"[HF] Model loaded: {model_id} on {device}")
            return model, tokenizer

        except Exception as e:
            raise LLMProviderError(f"モデルロード失敗 ({model_id}): {e}") from e


# ── HuggingFaceプロバイダー本体 ───────────────────────────────────────────────

class HuggingFaceProvider(LLMProvider):
    """
    HuggingFace Hub からダウンロードしたモデルでローカル推論するプロバイダー。

    設定は .hf_config.json で管理。
    初回は download_model_async() でモデルをキャッシュしてから使う。
    """

    def __init__(self, model_id: str | None = None) -> None:
        cfg = load_hf_config()
        self._model_id = model_id or cfg.get("model_id", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        self._token = get_hf_token()

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def display_name(self) -> str:
        info = get_model_info(self._model_id)
        return f"HuggingFace: {info.get('label', self._model_id)}"

    @property
    def description(self) -> str:
        return f"ローカル推論 ({self._model_id})"

    @property
    def is_available(self) -> bool:
        try:
            from transformers import AutoModelForCausalLM  # noqa
            return is_model_downloaded(self._model_id)
        except ImportError:
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """ローカルモデルで推論を実行する。"""
        try:
            import torch
            model, tokenizer = load_model(self._model_id, self._token)

            # メッセージ構築
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.user_prompt})

            # chat_template が利用可能な場合は使用
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = tokenizer(text, return_tensors="pt")
                except Exception:
                    # fallback: システム+ユーザープロンプトを結合
                    text = f"{request.system_prompt}\n\n{request.user_prompt}"
                    inputs = tokenizer(text, return_tensors="pt")
            else:
                text = f"{request.system_prompt}\n\n{request.user_prompt}"
                inputs = tokenizer(text, return_tensors="pt")

            # デバイス確認
            device = next(model.parameters()).device

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"].to(device),
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=request.max_tokens,
                    temperature=max(request.temperature, 0.01),
                    do_sample=request.temperature > 0.01,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # 入力部分を除いた生成トークンのみデコード
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            content = tokenizer.decode(generated, skip_special_tokens=True).strip()

            tokens_used = len(generated)
            is_truncated = tokens_used >= request.max_tokens - 10

            return LLMResponse(
                content=content,
                model=self._model_id,
                tokens_used=tokens_used,
                is_truncated=is_truncated,
            )

        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(f"推論エラー ({self._model_id}): {e}") from e
