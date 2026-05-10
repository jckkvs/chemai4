"""
backend/llm/providers/openai_provider.py

OpenAI APIを使ったLLMプロバイダー実装。
カスタムbase_urlを指定すれば、OpenAI互換API（ローカルLLM等）にも対応。

使用方法:
    import os
    os.environ["OPENAI_API_KEY"] = "sk-..."

    from backend.llm import register_llm_provider
    from backend.llm.providers.openai_provider import OpenAIProvider
    register_llm_provider("openai", OpenAIProvider)

    # カスタムAPIエンドポイント（ローカルLLM等）の場合
    provider = OpenAIProvider(
        model="local-model",
        api_key="not-needed",
        base_url="http://localhost:8000/v1",
    )

必要パッケージ:
    pip install openai
"""
from __future__ import annotations

import logging
import os

from backend.llm.provider import LLMProvider, LLMProviderError, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# ── 設定ファイルパス ─────────────────────────────────────────────────────────
_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".openai_config.json"
)

# ── 推奨モデルカタログ ────────────────────────────────────────────────────────
OPENAI_MODEL_CATALOG: list[dict] = [
    {
        "id": "gpt-4o",
        "label": "GPT-4o (推奨・高性能)",
        "description": "GPT-4o。マルチモーダル対応の高性能モデル。",
    },
    {
        "id": "gpt-4o-mini",
        "label": "GPT-4o Mini (高速・低コスト)",
        "description": "GPT-4o Mini。高速処理・低コスト。",
    },
    {
        "id": "gpt-4-turbo",
        "label": "GPT-4 Turbo",
        "description": "GPT-4 Turbo。128Kコンテキスト。",
    },
    {
        "id": "gpt-3.5-turbo",
        "label": "GPT-3.5 Turbo (軽量)",
        "description": "GPT-3.5 Turbo。軽量・高速。",
    },
]


def load_openai_config() -> dict:
    """OpenAI設定を読み込む。"""
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
        "model_id": "gpt-4o-mini",
        "base_url": "",
    }


def save_openai_config(config: dict) -> None:
    """OpenAI設定を保存する。"""
    import json
    from pathlib import Path

    Path(_CONFIG_FILE).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 環境変数にも反映
    if config.get("api_key"):
        os.environ["OPENAI_API_KEY"] = config["api_key"]
    if config.get("base_url"):
        os.environ["OPENAI_BASE_URL"] = config["base_url"]


def get_model_info(model_id: str) -> dict:
    """モデルIDに対応するカタログ情報を返す。"""
    for m in OPENAI_MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    return {"id": model_id, "label": model_id, "description": "カスタムモデル"}


class OpenAIProvider(LLMProvider):
    """
    OpenAI Chat Completions APIを使ったプロバイダー。

    対応モデル: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo 等
    カスタムbase_urlを指定すれば、OpenAI互換API（ローカルLLM等）にも対応。
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        cfg = load_openai_config()
        self.model = model or cfg.get("model_id", "gpt-4o-mini")
        self._api_key = api_key or cfg.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or cfg.get("base_url", "") or os.environ.get("OPENAI_BASE_URL", "") or None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        info = get_model_info(self.model)
        return f"OpenAI ({info.get('label', self.model)})"

    @property
    def description(self) -> str:
        return "OpenAI APIを使って記述子コードを生成します。"

    @property
    def is_available(self) -> bool:
        """OPENAI_API_KEY が設定されており、openaiパッケージが存在するか確認。"""
        if not self._api_key and not self._base_url:
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """OpenAI Chat Completions APIを呼び出す。"""
        try:
            import openai
        except ImportError as e:
            raise LLMProviderError(
                "openaiパッケージが未インストールです。`pip install openai` を実行してください。"
            ) from e

        if not self._api_key and not self._base_url:
            raise LLMProviderError(
                "OPENAI_API_KEY が設定されていません。"
                "環境変数 OPENAI_API_KEY を設定するか、"
                "OpenAIProvider(api_key='sk-...') で指定してください。"
            )

        client_kwargs = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        client = openai.OpenAI(**client_kwargs)

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            content = completion.choices[0].message.content or ""
            is_truncated = completion.choices[0].finish_reason != "stop"
            tokens_used = (
                completion.usage.total_tokens if completion.usage else 0
            )
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                is_truncated=is_truncated,
                raw=completion,
            )
        except openai.APIError as e:
            raise LLMProviderError(f"OpenAI APIエラー: {e}") from e
