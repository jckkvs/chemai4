"""
Test LLM providers functionality.
"""
import pytest
from backend.llm.provider import (
    LLMProvider,
    StubLLMProvider,
    LLMRequest,
    LLMResponse
)
from backend.llm import get_llm_provider, register_llm_provider
from backend.llm.registry import LLMProviderRegistry


def test_stub_provider_basic():
    """Stub provider should return predictable responses."""
    provider = StubLLMProvider()

    # Test basic generation
    request = LLMRequest(user_prompt="Test prompt")
    response = provider.generate(request)
    assert isinstance(response, LLMResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

    # Test with different parameters
    request = LLMRequest(user_prompt="Another prompt", max_tokens=10, temperature=0.5)
    response = provider.generate(request)
    assert isinstance(response, LLMResponse)


def test_provider_registry():
    """Provider registry should work correctly."""
    registry = LLMProviderRegistry()

    # Register stub provider
    registry.register("stub", StubLLMProvider)

    # Get provider
    provider = registry.get("stub")
    assert isinstance(provider, StubLLMProvider)

    # Test unknown provider
    with pytest.raises(KeyError):
        registry.get("unknown_provider")


def test_get_llm_provider():
    """Global get_llm_provider function should work."""
    provider = get_llm_provider("stub")
    assert isinstance(provider, StubLLMProvider)


def test_register_new_provider():
    """Should be able to register new provider types."""
    class DummyProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "dummy"

        def generate(self, request) -> LLMResponse:
            return LLMResponse(
                content="dummy_response",
                model="dummy",
                tokens_used=0,
                is_truncated=False
            )

    register_llm_provider("dummy", DummyProvider)
    provider = get_llm_provider("dummy")
    assert isinstance(provider, DummyProvider)
    request = LLMRequest(user_prompt="test")
    assert provider.generate(request).content == "dummy_response"


# Test HuggingFace provider if transformers is available
def test_hf_provider_if_available():
    """Test HuggingFace provider if dependencies are installed."""
    try:
        from backend.llm.providers.hf_provider import HuggingFaceProvider
        provider = HuggingFaceProvider(model_id="dummy-model")
        # Just test instantiation for now
        assert provider is not None
    except ImportError:
        pytest.skip("transformers not installed")


# Test GGUF provider if llama-cpp-python is available
def test_gguf_provider_if_available():
    """Test GGUF provider if dependencies are installed."""
    try:
        from backend.llm.providers.gguf_provider import GGUFProvider
        # Test that we can instantiate (without actual model file)
        provider = GGUFProvider(model_path="dummy.gguf")
        assert provider is not None
    except ImportError:
        pytest.skip("llama-cpp-python not installed")


# Test OpenAI provider if available
def test_openai_provider_if_available():
    """Test OpenAI provider if dependencies are installed."""
    try:
        from backend.llm.providers.openai_provider import OpenAIProvider
        # Test instantiation (won't actually connect without API key)
        provider = OpenAIProvider(api_key="dummy-key")
        assert provider is not None
    except ImportError:
        pytest.skip("openai not installed")


# Test Anthropic provider if available
def test_anthropic_provider_if_available():
    """Test Anthropic provider if dependencies are installed."""
    try:
        from backend.llm.providers.anthropic_provider import AnthropicProvider
        # Test instantiation (won't actually connect without API key)
        provider = AnthropicProvider(api_key="dummy-key")
        assert provider is not None
    except ImportError:
        pytest.skip("anthropic not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])