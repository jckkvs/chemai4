"""
backend/llm/providers/openrouter_provider.py

OpenRouter APIを使ったLLMプロバイダー実装。
OpenRouterは多数のLLMモデル（Claude、GPT、Gemini等）にアクセス可能。

使用方法:
    import os
    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-..."

    from backend.llm import register_llm_provider
    from backend.llm.providers.openrouter_provider import OpenRouterProvider
    register_llm_provider("openrouter", OpenRouterProvider)

必要パッケージ:
    pip install openai  # OpenRouterはOpenAI互換APIを使用
"""
from __future__ import annotations

import logging
import os

from backend.llm.provider import LLMProvider, LLMProviderError, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# ── 設定ファイルパス ─────────────────────────────────────────────────────────
_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".openrouter_config.json"
)

# ── 推奨モデルカタログ ────────────────────────────────────────────────────────
OPENROUTER_MODEL_CATALOG: list[dict] = [
    {
        "id": "anthropic/claude-sonnet-4-20250514",
        "label": "Claude Sonnet 4 (Anthropic via OpenRouter)",
        "description": "Claude Sonnet 4。OpenRouter経由でアクセス。",
    },
    {
        "id": "anthropic/claude-opus-4-20250514",
        "label": "Claude Opus 4 (Anthropic via OpenRouter)",
        "description": "Claude Opus 4。最高品質・複雑なタスク向け。",
    },
    {
        "id": "openai/gpt-4o",
        "label": "GPT-4o (OpenAI via OpenRouter)",
        "description": "OpenAI GPT-4o。高性能マルチモーダルモデル。",
    },
    {
        "id": "openai/gpt-4o-mini",
        "label": "GPT-4o Mini (OpenAI via OpenRouter)",
        "description": "OpenAI GPT-4o Mini。高速・低コスト。",
    },
    {
        "id": "google/gemini-pro-1.5",
        "label": "Gemini Pro 1.5 (Google via OpenRouter)",
        "description": "Google Gemini Pro 1.5。長文脈対応。",
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "label": "Llama 3.3 70B (Meta via OpenRouter)",
        "description": "Meta Llama 3.3 70B。オープンウェイトモデル。",
    },
    {
        "id": "qwen/qwen-2.5-coder-32b-instruct",
        "label": "Qwen 2.5 Coder 32B (via OpenRouter)",
        "description": "Qwen 2.5 Coder 32B。コード生成特化。",
    },
    {
        "id": "mistralai/mistral-large",
        "label": "Mistral Large (Mistral via OpenRouter)",
        "description": "Mistral Large。欧州製高性能モデル。",
    },
]


def load_openrouter_config() -> dict:
    """OpenRouter設定を読み込む。"""
    import json
    from pathlib import Path

    config_file = Path(_CONFIG_FILE)
    if config_file.exists():
        try:
            return json.loads(config_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "api_key": "",
        "model_id": "anthropic/claude-sonnet-4-20250514",
        "base_url": "https://openrouter.ai/api/v1",
        "site_url": "",
        "site_name": "",
    }


def save_openrouter_config(config: dict) -> None:
    """OpenRouter設定を保存する。"""
    import json
    from pathlib import Path

    Path(_CONFIG_FILE).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 環境変数にも反映
    if config.get("api_key"):
        os.environ["OPENROUTER_API_KEY"] = config["api_key"]


def get_model_info(model_id: str) -> dict:
    """モデルIDに対応するカタログ情報を返す。"""
    for m in OPENROUTER_MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    return {"id": model_id, "label": model_id, "description": "カスタムモデル"}


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter APIを使ったプロバイダー。

    OpenRouterはOpenAI互換APIを提供しているため、
    openaiパッケージを使用してアクセスする。

    対応モデル: Anthropic Claude, OpenAI GPT, Google Gemini, Meta Llama等
    多数のモデルに単一のAPIキーでアクセス可能。
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        cfg = load_openrouter_config()
        self.model = model or cfg.get("model_id", "anthropic/claude-sonnet-4-20250514")
        self._api_key = api_key or cfg.get("api_key", "") or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url or cfg.get("base_url", "https://openrouter.ai/api/v1")
        self._site_url = cfg.get("site_url", "")
        self._site_name = cfg.get("site_name", "ChemAI2")

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        info = get_model_info(self.model)
        return f"OpenRouter ({info.get('label', self.model)})"

    @property
    def description(self) -> str:
        return "OpenRouter経由で多数のLLMモデルにアクセスします。"

    @property
    def is_available(self) -> bool:
        """OPENROUTER_API_KEY が設定されており、openaiパッケージが存在するか確認。"""
        if not self._api_key:
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """OpenRouter API（OpenAI互換）を呼び出す。"""
        try:
            import openai
        except ImportError as e:
            raise LLMProviderError(
                "openaiパッケージが未インストールです。`pip install openai` を実行してください。"
            ) from e

        if not self._api_key:
            raise LLMProviderError(
                "OPENROUTER_API_KEY が設定されていません。"
                "https://openrouter.ai/keys でAPIキーを発行してください。"
            )

        try:
            client_kwargs = {
                "api_key": self._api_key,
                "base_url": self._base_url,
            }
            client = openai.OpenAI(**client_kwargs)

            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.user_prompt})

            # OpenRouter用のextraヘッダー
            extra_headers = {}
            if self._site_url:
                extra_headers["HTTP-Referer"] = self._site_url
            if self._site_name:
                extra_headers["X-Title"] = self._site_name

            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                extra_headers=extra_headers if extra_headers else None,
            )

            content = completion.choices[0].message.content or ""
            is_truncated = completion.choices[0].finish_reason != "stop"
            tokens_used = (
                completion.usage.total_tokens if completion.usage else 0
            )

            return LLMResponse(
                content=content.strip(),
                model=self.model,
                tokens_used=tokens_used,
                is_truncated=is_truncated,
                raw=completion,
            )

        except Exception as e:
            raise LLMProviderError(f"OpenRouter APIエラー: {e}") from e
