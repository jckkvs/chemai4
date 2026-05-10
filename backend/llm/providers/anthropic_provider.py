"""
backend/llm/providers/anthropic_provider.py

Anthropic Claude APIを使ったLLMプロバイダー実装。

使用方法:
    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

    from backend.llm import register_llm_provider
    from backend.llm.providers.anthropic_provider import AnthropicProvider
    register_llm_provider("anthropic", AnthropicProvider)

必要パッケージ:
    pip install anthropic
"""
from __future__ import annotations

import logging
import os

from backend.llm.provider import LLMProvider, LLMProviderError, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# ── 設定ファイルパス ─────────────────────────────────────────────────────────
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".anthropic_config.json")

# ── 推奨モデルカタログ ────────────────────────────────────────────────────────
ANTHROPIC_MODEL_CATALOG: list[dict] = [
    {
        "id": "claude-sonnet-4-20250514",
        "label": "Claude Sonnet 4 (推奨・高性能)",
        "description": "Claude 4 Sonnet。バランスの取れた高性能モデル。",
    },
    {
        "id": "claude-sonnet-4-20250514",
        "label": "Claude Opus 4.7",
        "description": "Claude Opus 4.7。最高品質・複雑なタスク向け。",
    },
    {
        "id": "claude-haiku-4-5-20251001",
        "label": "Claude Haiku 4.5 (高速・軽量)",
        "description": "Claude Haiku 4.5。高速処理・低コスト。",
    },
    {
        "id": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6",
        "description": "Claude Sonnet 4.6。安定版。",
    },
]


def load_anthropic_config() -> dict:
    """Anthropic設定を読み込む。"""
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
        "model_id": "claude-sonnet-4-20250514",
        "base_url": "",  # 空の場合はデフォルトURLを使用
    }


def save_anthropic_config(config: dict) -> None:
    """Anthropic設定を保存する。"""
    import json
    from pathlib import Path

    Path(_CONFIG_FILE).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 環境変数にも反映
    if config.get("api_key"):
        os.environ["ANTHROPIC_API_KEY"] = config["api_key"]


def get_model_info(model_id: str) -> dict:
    """モデルIDに対応するカタログ情報を返す。"""
    for m in ANTHROPIC_MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    return {"id": model_id, "label": model_id, "description": "カスタムモデル"}


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude APIを使ったプロバイダー。

    対応モデル: claude-sonnet-4-*, claude-opus-4-*, claude-haiku-4-* 等
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        cfg = load_anthropic_config()
        self.model = model or cfg.get("model_id", "claude-sonnet-4-20250514")
        self._api_key = api_key or cfg.get("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = base_url or cfg.get("base_url", "") or None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        info = get_model_info(self.model)
        return f"Anthropic Claude ({info.get('label', self.model)})"

    @property
    def description(self) -> str:
        return "Anthropic Claude APIを使って記述子コードを生成します。"

    @property
    def is_available(self) -> bool:
        """ANTHROPIC_API_KEY が設定されており、anthropicパッケージが存在するか確認。"""
        if not self._api_key:
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Anthropic Messages APIを呼び出す。"""
        try:
            import anthropic
        except ImportError as e:
            raise LLMProviderError(
                "anthropicパッケージが未インストールです。`pip install anthropic` を実行してください。"
            ) from e

        if not self._api_key:
            raise LLMProviderError(
                "ANTHROPIC_API_KEY が設定されていません。"
                "環境変数 ANTHROPIC_API_KEY を設定するか、"
                "AnthropicProvider(api_key='sk-ant-...') で指定してください。"
            )

        try:
            client_kwargs = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            client = anthropic.Anthropic(**client_kwargs)

            messages = [{"role": "user", "content": request.user_prompt}]
            system = request.system_prompt if request.system_prompt else None

            kwargs = {
                "model": self.model,
                "max_tokens": request.max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)

            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            tokens_used = 0
            if hasattr(response, "usage") and response.usage:
                tokens_used = (
                    (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
                )

            is_truncated = len(content.split()) >= request.max_tokens - 10

            return LLMResponse(
                content=content.strip(),
                model=self.model,
                tokens_used=tokens_used,
                is_truncated=is_truncated,
                raw=response,
            )

        except Exception as e:
            raise LLMProviderError(f"Anthropic APIエラー: {e}") from e
