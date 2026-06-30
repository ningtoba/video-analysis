"""
Tests for the self-contained LLM provider abstraction (v0.39.0).

Tests are structured to be fast and not require actual LLM endpoints:
- HermesProvider tests mock subprocess
- OpenAIProvider tests mock requests
- Provider factory tests verify caching and fallback logic
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from video_analysis.llm_provider import (
    HermesProvider,
    LLMProvider,
    LLMProviderConfig,
    OpenAIProvider,
    get_llm_provider,
    reset_provider_cache,
)


class _AsyncStreamCM:
    """Async context manager for mocking httpx client.stream()."""

    def __init__(self, mock_response):
        self._response = mock_response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return None


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def default_config():
    return LLMProviderConfig(
        provider="hermes",
        api_base="http://localhost:11434/v1",
        api_key="",
        model="qwen2.5",
        max_tokens=2048,
        temperature=0.3,
        timeout=30,
        hermes_model="deepseek-ai/DeepSeek-V4-Flash",
        hermes_max_tokens=2048,
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
        timeout=30,
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
        assert cfg.provider == "hermes"
        assert cfg.api_base == "http://localhost:11434/v1"
        assert cfg.api_key == ""
        assert cfg.model == "qwen2.5"
        assert cfg.max_tokens == 2048

    def test_from_env(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_BASE": "http://localhost:8000/v1",
                "OPENAI_MODEL": "llama3",
            },
        ):
            cfg = LLMProviderConfig.from_env()
            assert cfg.provider == "openai"
            assert cfg.api_base == "http://localhost:8000/v1"
            assert cfg.model == "llama3"

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
# HermesProvider
# =========================================================================


class TestHermesProvider:
    def test_init(self, default_config):
        provider = HermesProvider(default_config)
        assert provider.name == "hermes"
        assert isinstance(provider, LLMProvider)

    def test_chat_success(self, default_config):
        provider = HermesProvider(default_config)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Test response\n"
        mock_result.stderr = ""

        with patch.object(subprocess, "run", return_value=mock_result):
            response = provider.chat("What happened in the video?")
            assert response == "Test response"

    def test_chat_failure(self, default_config):
        provider = HermesProvider(default_config)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"

        with patch.object(subprocess, "run", return_value=mock_result):
            response = provider.chat("What happened?")
            assert response == ""

    def test_chat_file_not_found(self, default_config):
        provider = HermesProvider(default_config)
        with patch.object(subprocess, "run", side_effect=FileNotFoundError()):
            response = provider.chat("What happened?")
            assert response == ""

    def test_chat_timeout(self, default_config):
        provider = HermesProvider(default_config)
        with patch.object(
            subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="hermes", timeout=30),
        ):
            response = provider.chat("What happened?")
            assert response == ""

    def test_chat_with_system(self, default_config):
        provider = HermesProvider(default_config)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Concise answer."

        with patch.object(subprocess, "run", return_value=mock_result) as mock_run:
            provider.chat("Hello", system="Be brief.")
            args, kwargs = mock_run.call_args
            # Verify system prompt was passed
            assert "-s" in args[0]

    def test_available_true(self, default_config):
        provider = HermesProvider(default_config)
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch.object(subprocess, "run", return_value=mock_result):
            assert provider.available is True

    def test_available_false(self, default_config):
        provider = HermesProvider(default_config)
        with patch.object(subprocess, "run", side_effect=FileNotFoundError()):
            assert provider.available is False

    def test_structured_chat_success(self, default_config):
        provider = HermesProvider(default_config)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"route": "text", "confidence": 0.9})

        with patch.object(subprocess, "run", return_value=mock_result):
            result = provider.structured_chat("Classify this")
            assert result is not None
            assert result["route"] == "text"
            assert result["confidence"] == 0.9

    def test_structured_chat_json_in_code_block(self, default_config):
        provider = HermesProvider(default_config)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '```json\n{"title": "Hello", "summary": "Test"}\n```'

        with patch.object(subprocess, "run", return_value=mock_result):
            result = provider.structured_chat("Generate title")
            assert result is not None
            assert result["title"] == "Hello"

    def test_structured_chat_brace_extraction(self, default_config):
        provider = HermesProvider(default_config)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'Here is the JSON: {"answer": 42}. That\'s it.'

        with patch.object(subprocess, "run", return_value=mock_result):
            result = provider.structured_chat("Answer")
            assert result is not None
            assert result["answer"] == 42

    def test_structured_chat_empty_output(self, default_config):
        provider = HermesProvider(default_config)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch.object(subprocess, "run", return_value=mock_result):
            result = provider.structured_chat("Answer")
            assert result is None

    # ------------------------------------------------------------------
    # stream_chat tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_chat_success(self, default_config):
        """stream_chat should yield tokens from subprocess stdout."""
        provider = HermesProvider(default_config)

        # Use a simpler approach: patch create_subprocess_exec to return
        # a mock process whose communicate returns our test output.
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Hello world\nHow are you?\n", b""))

        with patch.object(asyncio, "create_subprocess_exec", return_value=mock_process):
            tokens = []
            async for token in provider.stream_chat("Hello"):
                tokens.append(token)

            full = "".join(tokens)
            assert "Hello world" in full or "Hello" in full
            assert len(tokens) >= 1

    @pytest.mark.asyncio
    async def test_stream_chat_fallback_on_failure(self, default_config):
        """When the subprocess fails, stream_chat should fall back to
        word-by-word emission from the synchronous chat result."""
        provider = HermesProvider(default_config)

        # Make the async subprocess fail
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"error message"))

        with patch.object(asyncio, "create_subprocess_exec", return_value=mock_process):
            # Patch synchronous chat to return a known result
            with patch.object(provider, "chat", return_value="Fallback answer."):
                tokens = []
                async for token in provider.stream_chat("Hello"):
                    tokens.append(token)

                full = "".join(tokens).strip()
                assert "Fallback" in full

    @pytest.mark.asyncio
    async def test_stream_chat_fallback_on_timeout(self, default_config):
        """Timeout should trigger the word-by-word fallback."""
        provider = HermesProvider(default_config)

        with patch.object(
            asyncio,
            "create_subprocess_exec",
            side_effect=asyncio.TimeoutError(),
        ):
            with patch.object(provider, "chat", return_value="Timed out answer."):
                tokens = []
                async for token in provider.stream_chat("Hello"):
                    tokens.append(token)

                full = "".join(tokens).strip()
                assert "Timed out" in full

    @pytest.mark.asyncio
    async def test_stream_chat_fallback_on_not_found(self, default_config):
        """FileNotFoundError should trigger the word-by-word fallback."""
        provider = HermesProvider(default_config)

        with patch.object(
            asyncio,
            "create_subprocess_exec",
            side_effect=FileNotFoundError(),
        ):
            with patch.object(provider, "chat", return_value="Not found answer."):
                tokens = []
                async for token in provider.stream_chat("Hello"):
                    tokens.append(token)

                full = "".join(tokens).strip()
                assert "Not found" in full


# =========================================================================
# OpenAIProvider
# =========================================================================


class TestOpenAIProvider:
    def test_init(self, openai_config):
        provider = OpenAIProvider(openai_config)
        assert provider.name == "openai"
        assert isinstance(provider, LLMProvider)

    def test_chat_success(self, openai_config):
        provider = OpenAIProvider(openai_config)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "The video shows a car chase."}}]
        }
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        with patch.object(provider, "_get_session", return_value=mock_session):
            response = provider.chat("What happens?")
            assert response == "The video shows a car chase."

    def test_chat_empty_response(self, openai_config):
        provider = OpenAIProvider(openai_config)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        with patch.object(provider, "_get_session", return_value=mock_session):
            response = provider.chat("What happens?")
            assert response == ""

    def test_chat_http_error(self, openai_config):
        provider = OpenAIProvider(openai_config)
        mock_session = MagicMock()
        mock_session.post.side_effect = Exception("HTTP 500")

        with patch.object(provider, "_get_session", return_value=mock_session):
            response = provider.chat("What happens?")
            assert response == ""

    def test_chat_with_system(self, openai_config):
        provider = OpenAIProvider(openai_config)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        with patch.object(provider, "_get_session", return_value=mock_session) as mock_get:
            provider.chat("Hello", system="Be concise")
            call_kwargs = mock_session.post.call_args[1]
            payload = call_kwargs["json"]
            assert len(payload["messages"]) == 2
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][0]["content"] == "Be concise"

    def test_available_true(self, openai_config):
        provider = OpenAIProvider(openai_config)
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp):
            assert provider.available is True

    def test_available_false(self, openai_config):
        provider = OpenAIProvider(openai_config)
        with patch("requests.get", side_effect=Exception("Connection refused")):
            assert provider.available is False

    def test_structured_chat_success(self, openai_config):
        provider = OpenAIProvider(openai_config)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"route": "text", "confidence": 0.95})}}]
        }
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        with patch.object(provider, "_get_session", return_value=mock_session):
            result = provider.structured_chat("Classify this")
            assert result is not None
            assert result["route"] == "text"
            assert result["confidence"] == 0.95

    def test_structured_chat_json_in_text(self, openai_config):
        provider = OpenAIProvider(openai_config)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": 'The answer is {"result": "found", "count": 3}'}}]
        }
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        with patch.object(provider, "_get_session", return_value=mock_session):
            result = provider.structured_chat("Count items")
            assert result is not None
            assert result["result"] == "found"
            assert result["count"] == 3

    def test_build_messages_without_system(self, openai_config):
        provider = OpenAIProvider(openai_config)
        msgs = provider._build_messages("Hello")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"

    def test_build_messages_with_system(self, openai_config):
        provider = OpenAIProvider(openai_config)
        msgs = provider._build_messages("Hello", system="Be brief")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be brief"
        assert msgs[1]["role"] == "user"

    def test_api_url_formatting(self, openai_config):
        provider = OpenAIProvider(openai_config)
        # Config has api_base ending in /v1
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        with patch.object(provider, "_get_session", return_value=mock_session):
            provider.chat("Hello")
            call_args = mock_session.post.call_args[0]
            url = call_args[0]
            assert url.endswith("/chat/completions")

    def test_api_url_already_has_completions(self):
        cfg = LLMProviderConfig(
            provider="openai",
            api_base="http://localhost:8000/chat/completions",
            api_key="",
            model="gpt",
        )
        provider = OpenAIProvider(cfg)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        with patch.object(provider, "_get_session", return_value=mock_session):
            provider.chat("Hello")
            call_args = mock_session.post.call_args[0]
            url = call_args[0]
            assert url == "http://localhost:8000/chat/completions"

    # ------------------------------------------------------------------
    # stream_chat tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_chat_success(self, openai_config):
        """Should yield tokens from SSE data events."""
        import httpx

        provider = OpenAIProvider(openai_config)

        # Simulate SSE data events
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            "data: [DONE]",
        ]

        async def _mock_lines():
            for line in sse_lines:
                yield line

        mock_stream_response = MagicMock()
        mock_stream_response.aiter_lines = _mock_lines
        mock_stream_response.raise_for_status = MagicMock()

        mock_stream_cm = _AsyncStreamCM(mock_stream_response)

        with patch.object(httpx.AsyncClient, "stream", return_value=mock_stream_cm):
            tokens = []
            async for token in provider.stream_chat("Hello"):
                tokens.append(token)

            full = "".join(tokens)
            assert full == "Hello world!"

    @pytest.mark.asyncio
    async def test_stream_chat_empty_choices(self, openai_config):
        """Empty choices should be skipped."""
        import httpx

        provider = OpenAIProvider(openai_config)

        sse_lines = [
            'data: {"choices":[]}',
            'data: {"choices":[{"delta":{}}]}',
            'data: {"choices":[{"delta":{"content":"only"}}]}',
            "data: [DONE]",
        ]

        async def _mock_lines():
            for line in sse_lines:
                yield line

        mock_stream_response = MagicMock()
        mock_stream_response.aiter_lines = _mock_lines
        mock_stream_response.raise_for_status = MagicMock()

        mock_stream_cm = _AsyncStreamCM(mock_stream_response)

        with patch.object(httpx.AsyncClient, "stream", return_value=mock_stream_cm):
            tokens = []
            async for token in provider.stream_chat("Hello"):
                tokens.append(token)

            full = "".join(tokens)
            assert full == "only"

    @pytest.mark.asyncio
    async def test_stream_chat_fallback_on_failure(self, openai_config):
        """When the stream API call fails, fall back to synchronous."""
        import httpx

        provider = OpenAIProvider(openai_config)

        mock_stream_cm = _AsyncStreamCM(None)
        # Make __aenter__ raise
        mock_stream_cm.__aenter__ = AsyncMock(side_effect=Exception("API unreachable"))

        with patch.object(httpx.AsyncClient, "stream", return_value=mock_stream_cm):
            # Patch the synchronous chat method
            with patch.object(provider, "chat", return_value="Fallback response"):
                tokens = []
                async for token in provider.stream_chat("Hello"):
                    tokens.append(token)

                full = "".join(tokens)
                assert "Fallback response" in full

    @pytest.mark.asyncio
    async def test_stream_chat_handles_done_signal(self, openai_config):
        """Stream should stop cleanly on [DONE] signal."""
        import httpx

        provider = OpenAIProvider(openai_config)

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"first "}}]}',
            "data: [DONE]",
            # This should never be reached
            'data: {"choices":[{"delta":{"content":"EXTRA"}}]}',
        ]

        async def _mock_lines():
            for line in sse_lines:
                yield line

        mock_stream_response = MagicMock()
        mock_stream_response.aiter_lines = _mock_lines
        mock_stream_response.raise_for_status = MagicMock()

        mock_stream_cm = _AsyncStreamCM(mock_stream_response)

        with patch.object(httpx.AsyncClient, "stream", return_value=mock_stream_cm):
            tokens = []
            async for token in provider.stream_chat("Hello"):
                tokens.append(token)

            full = "".join(tokens)
            assert full == "first "
            assert "EXTRA" not in full


# =========================================================================
# Factory function (get_llm_provider)
# =========================================================================


class TestGetLLMProvider:
    def test_default_hermes(self):
        provider = get_llm_provider(LLMProviderConfig(provider="hermes"))
        assert isinstance(provider, HermesProvider)

    def test_openai(self, openai_config):
        provider = get_llm_provider(openai_config)
        assert isinstance(provider, OpenAIProvider)

    def test_auto_fallback_to_hermes(self):
        config = LLMProviderConfig(provider="auto")
        with patch(
            "video_analysis.llm_provider.OpenAIProvider.available",
            new_callable=MagicMock(return_value=False),
        ):
            provider = get_llm_provider(config)
            assert isinstance(provider, HermesProvider)

    def test_auto_prefers_openai(self, openai_config):
        config = LLMProviderConfig(provider="auto")
        with patch(
            "video_analysis.llm_provider.OpenAIProvider.available",
            new_callable=MagicMock(return_value=True),
        ):
            provider = get_llm_provider(config)
            assert isinstance(provider, OpenAIProvider)

    def test_force_overrides(self, default_config):
        provider = get_llm_provider(default_config, force="openai")
        assert isinstance(provider, OpenAIProvider)

    def test_caching(self, default_config):
        p1 = get_llm_provider(default_config)
        p2 = get_llm_provider(default_config)
        assert p1 is p2  # same instance

    def test_reset_cache(self, default_config):
        p1 = get_llm_provider(default_config)
        reset_provider_cache()
        p2 = get_llm_provider(default_config)
        assert p1 is not p2

    def test_unknown_provider_falls_back(self):
        config = LLMProviderConfig(provider="nonexistent")
        provider = get_llm_provider(config)
        assert isinstance(provider, HermesProvider)

    def test_force_hermes(self, openai_config):
        provider = get_llm_provider(openai_config, force="hermes")
        assert isinstance(provider, HermesProvider)

    @patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_BASE": "http://test:8000/v1"})
    def test_from_env_via_factory(self):
        provider = get_llm_provider()
        assert isinstance(provider, OpenAIProvider)
