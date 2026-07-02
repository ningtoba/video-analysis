"""
Tests for the unified LLM provider abstraction.

Tests are structured to be fast and not require actual LLM endpoints:
- LLMProviderConfig tests verify configuration parsing
- Provider factory tests verify caching and provider creation logic
"""

from __future__ import annotations

import pytest


from video_analysis.llm_provider import (
    LLMProvider,
    LLMProviderConfig,
    get_llm_provider,
    reset_provider_cache,
)




# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def default_config():
    return LLMProviderConfig(
        provider="openai",
        api_base="http://localhost:11434/v1",
        api_key="",
        model="qwen2.5",
        max_tokens=2048,
        temperature=0.3,
    )


@pytest.fixture
def openai_config():
    return LLMProviderConfig(
        provider="openai",
        api_base="http://localhost:8000/v1",
        api_key="test-key",
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0.1,
    )


@pytest.fixture(autouse=True)
def clear_cache():
    reset_provider_cache()
    yield
    reset_provider_cache()


# =========================================================================
# LLMProviderConfig
# =========================================================================


class TestLLMProviderConfig:
    def test_defaults(self):
        cfg = LLMProviderConfig()
        assert cfg.provider == "openai"
        assert cfg.api_base == ""
        assert cfg.api_key == ""
        assert cfg.model == ""
        assert cfg.max_tokens == 4096


    def test_init(self):
        cfg = LLMProviderConfig(
            provider="openai",
            api_base="http://vllm:8000/v1",
            model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            max_tokens=4096,
        )
        assert cfg.provider == "openai"
        assert cfg.api_base == "http://vllm:8000/v1"
        assert cfg.model == "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
        assert cfg.max_tokens == 4096




# =========================================================================
# Factory function (get_llm_provider)
# =========================================================================


class TestGetLLMProvider:
    def test_default_openai(self):
        provider = get_llm_provider(LLMProviderConfig(provider="openai"))
        assert isinstance(provider, LLMProvider)

    def test_openai(self, openai_config):
        provider = get_llm_provider(openai_config)
        assert isinstance(provider, LLMProvider)

    def test_caching(self, default_config):
        p1 = get_llm_provider(default_config)
        p2 = get_llm_provider(default_config)
        assert p1 is p2  # same instance

    def test_reset_cache(self, default_config):
        p1 = get_llm_provider(default_config)
        reset_provider_cache()
        p2 = get_llm_provider(default_config)
        assert p1 is not p2
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider: nonexistent"):
            get_llm_provider(LLMProviderConfig(provider="nonexistent"))

    def test_from_env_via_factory(self):
        config = LLMProviderConfig(
            provider="openai",
            api_base="http://test:8000/v1",
            api_key="",
        )
        provider = get_llm_provider(config)
        assert isinstance(provider, LLMProvider)
