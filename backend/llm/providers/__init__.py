# backend/llm/providers/__init__.py
"""
LLMプロバイダーパッケージ。

利用可能なプロバイダー:
    - openai_provider.py      : OpenAI GPT-4o/4o-mini
    - anthropic_provider.py   : Anthropic Claude
    - openrouter_provider.py   : OpenRouter (多数のLLMにアクセス)
    - hf_provider.py          : HuggingFace ローカルLLM (Qwen, Phi, Gemma等)
    - gguf_provider.py        : GGUF量子化モデル (Bonsai, Qwen等)

新しいプロバイダーを追加するには:
    1. このディレクトリに xxx_provider.py を作成
    2. LLMProvider を継承し generate() を実装
    3. backend/llm/__init__.py の register_llm_provider() で登録
"""
