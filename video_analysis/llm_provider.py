"""
Self-Contained LLM Provider — abstract over LLM backends to remove the hard
dependency on Hermes CLI.

Supports three modes:
  1. **hermes** (default) — existing `hermes chat -q` subprocess
  2. **openai** — any OpenAI-compatible API (vLLM, Ollama, llama.cpp, TGI, etc.)
  3. **auto** — try openai first, fall back to hermes

Usage:
    from video_analysis.llm_provider import get_llm_provider

    llm = get_llm_provider(config)
    answer = llm.chat("What happened in the video?")
    answer = llm.chat(prompt, system="Be concise.", temperature=0.1)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LLMProviderConfig:
    """Configuration for the LLM provider abstraction.

    Read from Config (env vars) when ``from_config`` is used.
    """

    provider: str = "hermes"  # "hermes", "openai", or "auto"
    api_base: str = "http://localhost:11434/v1"  # OpenAI-compatible base URL
    api_key: str = ""  # API key (empty OK for local servers)
    model: str = "qwen2.5"  # Model name for the API
    max_tokens: int = 2048
    temperature: float = 0.3
    timeout: int = 120
    # Hermes-specific
    hermes_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    hermes_max_tokens: int = 2048

    @classmethod
    def from_env(cls) -> "LLMProviderConfig":
        """Read config from environment variables."""
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "hermes").lower(),
            api_base=os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("OPENAI_MODEL", "qwen2.5"),
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "2048")),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.3")),
            timeout=int(os.environ.get("LLM_TIMEOUT", "120")),
            hermes_model=os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
            hermes_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2048")),
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Send a chat prompt and return the response text.

        Args:
            prompt: The user message / prompt text.
            system: Optional system prompt.
            temperature: Temperature override (None = config default).
            max_tokens: Max tokens override (None = config default).
            timeout: Timeout override in seconds (None = config default).

        Returns:
            The response text, or empty string on failure.
        """
        ...

    @abstractmethod
    def structured_chat(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a chat prompt and return a structured JSON response.

        The prompt should instruct the model to respond in JSON. Returns
        the parsed dict, or None on failure.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. 'hermes', 'openai')."""
        ...

    @property
    def available(self) -> bool:
        """Whether this provider is available (can make calls)."""
        return True


# ---------------------------------------------------------------------------
# Hermes CLI provider
# ---------------------------------------------------------------------------


class HermesProvider(LLMProvider):
    """LLM backend using ``hermes chat -q`` subprocess.

    This is the existing/default provider — works when Hermes Agent CLI
    is installed and configured on the system PATH.
    """

    def __init__(self, config: Optional[LLMProviderConfig] = None):
        self._config = config or LLMProviderConfig.from_env()

    @property
    def name(self) -> str:
        return "hermes"

    @property
    def available(self) -> bool:
        try:
            result = subprocess.run(
                ["hermes", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        try:
            cmd = [
                "hermes",
                "chat",
                "-q",
                "-m",
                self._config.hermes_model,
                "-t",
                str(
                    temperature if temperature is not None else self._config.temperature
                ),
                "--max-tokens",
                str(
                    max_tokens
                    if max_tokens is not None
                    else self._config.hermes_max_tokens
                ),
            ]
            if system:
                cmd += ["-s", system]

            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout if timeout is not None else self._config.timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()

            logger.warning(
                "Hermes CLI call failed (rc=%d): %s",
                result.returncode,
                result.stderr[:200],
            )
            return ""
        except FileNotFoundError:
            logger.warning("Hermes CLI not found on PATH")
            return ""
        except subprocess.TimeoutExpired:
            logger.warning("Hermes CLI call timed out")
            return ""
        except Exception as e:
            logger.error("Hermes CLI error: %s", e)
            return ""

    def structured_chat(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        raw = self.chat(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if not raw:
            return None
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response."""
        raw = raw.strip()
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        import re

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find { ... } block
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(raw[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """LLM backend using any OpenAI-compatible API endpoint.

    Works with:
    - vLLM (``http://localhost:8000/v1``)
    - Ollama (``http://localhost:11434/v1``)
    - llama.cpp (``http://localhost:8080/v1``)
    - Text Generation Inference
    - OpenAI / Azure OpenAI (with API key)
    - Any OpenAI-compatible chat completion API
    """

    def __init__(self, config: Optional[LLMProviderConfig] = None):
        self._config = config or LLMProviderConfig.from_env()
        self._session = None  # lazy-imported requests.Session

    def _get_session(self):
        """Lazy-imported requests session (avoids import-time dep)."""
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._config.api_key}",
                }
            )
        return self._session

    @property
    def name(self) -> str:
        return "openai"

    @property
    def available(self) -> bool:
        """Check availability by querying the model list endpoint."""
        try:
            import requests

            base = self._config.api_base.rstrip("/")
            url = f"{base}/models" if "chat/completions" not in base else base

            resp = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                },
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        result = self._call_api(
            messages=self._build_messages(prompt, system),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if result is None:
            return ""

        # Extract content from response
        try:
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                return content.strip() if content else ""
        except (IndexError, KeyError, AttributeError):
            pass

        return ""

    def structured_chat(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        # Add JSON instruction to the system prompt
        json_system = (system + "\n\n" if system else "") + (
            "Respond in valid JSON only, no markdown formatting."
        )
        result = self._call_api(
            messages=self._build_messages(prompt, json_system),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if result is None:
            return None

        try:
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if content:
                    return self._parse_json(content)
        except (IndexError, KeyError, AttributeError):
            pass

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: str, system: str = "") -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make the OpenAI-compatible chat completions API call."""
        session = self._get_session()
        base = self._config.api_base.rstrip("/")
        # Ensure the URL ends with /chat/completions
        if not base.endswith("/chat/completions"):
            base = base.rstrip("/") + "/chat/completions"

        payload: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self._config.temperature
            ),
            "max_tokens": (
                max_tokens if max_tokens is not None else self._config.max_tokens
            ),
        }

        try:
            resp = session.post(
                base,
                json=payload,
                timeout=timeout if timeout is not None else self._config.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("OpenAI API call failed: %s", e)
            return None

    @staticmethod
    def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response."""
        raw = raw.strip()
        # Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Markdown code blocks
        import re

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # { ... } extraction
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(raw[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_CACHE: Dict[str, LLMProvider] = {}


def get_llm_provider(
    config: Optional[LLMProviderConfig] = None,
    force: Optional[str] = None,
) -> LLMProvider:
    """Get the best available LLM provider.

    Args:
        config: Provider configuration (reads from env if None).
        force: Override provider type ("hermes", "openai", "auto").

    Returns:
        An initialized LLMProvider instance.

    The provider is cached by type so repeated calls reuse the same instance.
    """
    cfg = config or LLMProviderConfig.from_env()
    provider_type = (force or cfg.provider).lower()

    cache_key = provider_type
    if cache_key in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[cache_key]

    if provider_type == "hermes":
        provider: LLMProvider = HermesProvider(cfg)
    elif provider_type == "openai":
        provider = OpenAIProvider(cfg)
    elif provider_type == "auto":
        # Try OpenAI first (more capable), fall back to Hermes
        openai_provider = OpenAIProvider(cfg)
        if openai_provider.available:
            provider = openai_provider
        else:
            provider = HermesProvider(cfg)
    else:
        logger.warning(
            "Unknown provider type %r, falling back to hermes", provider_type
        )
        provider = HermesProvider(cfg)

    _PROVIDER_CACHE[cache_key] = provider
    return provider


def reset_provider_cache():
    """Reset the provider cache (useful for testing)."""
    _PROVIDER_CACHE.clear()
