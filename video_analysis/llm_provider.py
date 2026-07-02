"""
LLM provider abstraction — supports OpenAI, Anthropic, Gemini, DeepSeek, and any OpenAI-compatible API.

All vision tasks (scene understanding, object detection, OCR) are handled by the LLM Vision API.
No local vision models needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMProviderConfig:
    """Configuration for the LLM provider."""
    provider: str = "openai"  # openai, anthropic, gemini, deepseek, ollama
    api_key: str = ""
    api_base: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    http_client: Optional[httpx.Client] = None


class LLMProvider:
    """Unified LLM provider supporting text and vision chat.

    Handles OpenAI-compatible APIs (OpenAI, DeepSeek, Ollama, vLLM, etc.)
    and native Anthropic/Gemini APIs.
    """

    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self._client: Optional[httpx.Client] = config.http_client
        self._headers: Dict[str, str] = {}

        if config.provider in ("openai", "deepseek", "ollama"):
            self._setup_openai_compatible()
        elif config.provider == "anthropic":
            self._setup_anthropic()
        elif config.provider == "gemini":
            self._setup_gemini()
        else:
            raise ValueError(f"Unknown LLM provider: {config.provider}")

    def _setup_openai_compatible(self):
        base = self.config.api_base or {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com",
            "ollama": "http://localhost:11434/v1",
        }.get(self.config.provider, "https://api.openai.com/v1")

        self._api_base = base.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if not self._client:
            self._client = httpx.Client(timeout=120)

    def _setup_anthropic(self):
        self._api_base = "https://api.anthropic.com/v1"
        self._headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if not self._client:
            self._client = httpx.Client(timeout=120)

    def _setup_gemini(self):
        self._api_base = "https://generativelanguage.googleapis.com/v1beta"
        self._headers = {"Content-Type": "application/json"}
        if not self._client:
            self._client = httpx.Client(timeout=120)

    @property
    def name(self) -> str:
        return f"{self.config.provider}/{self.config.model}"

    # ── Text-only chat ──────────────────────────────────────────────

    def chat(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> Optional[str]:
        """Send a text-only chat completion request.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            system: Optional system prompt

        Returns:
            Response text or None on failure.
        """
        if self.config.provider == "anthropic":
            return self._chat_anthropic(messages, system)
        elif self.config.provider == "gemini":
            return self._chat_gemini(messages, system)
        else:
            return self._chat_openai(messages, system)

    def _chat_openai(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> Optional[str]:
        full_messages = list(messages)
        if system:
            full_messages.insert(0, {"role": "system", "content": system})

        try:
            resp = self._client.post(
                f"{self._api_base}/chat/completions",
                headers=self._headers,
                json={
                    "model": self.config.model,
                    "messages": full_messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("OpenAI chat failed: %s", e)
            return None

    def _chat_anthropic(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> Optional[str]:
        try:
            body: Dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": messages,
            }
            if system:
                body["system"] = system

            resp = self._client.post(
                f"{self._api_base}/messages",
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]
        except Exception as e:
            logger.error("Anthropic chat failed: %s", e)
            return None

    def _chat_gemini(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> Optional[str]:
        try:
            contents = []
            for msg in messages:
                contents.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [{"text": msg["content"]}],
                })

            body: Dict[str, Any] = {
                "contents": contents,
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens,
                },
            }
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}

            resp = self._client.post(
                f"{self._api_base}/models/{self.config.model}:generateContent",
                headers=self._headers,
                params={"key": self.config.api_key},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error("Gemini chat failed: %s", e)
            return None

    # ── Vision chat (send images) ───────────────────────────────────

    def chat_with_images(
        self,
        messages: List[Dict[str, str]],
        images: List[str],
        system: Optional[str] = None,
    ) -> Optional[str]:
        """Send a chat with images (base64-encoded).

        Args:
            messages: Text messages
            images: List of base64-encoded image strings
            system: Optional system prompt

        Returns:
            Response text or None on failure.
        """
        if self.config.provider == "anthropic":
            return self._vision_anthropic(messages, images, system)
        elif self.config.provider == "gemini":
            return self._vision_gemini(messages, images, system)
        else:
            return self._vision_openai(messages, images, system)

    def _vision_openai(
        self,
        messages: List[Dict[str, str]],
        images: List[str],
        system: Optional[str] = None,
    ) -> Optional[str]:
        content: List[Dict[str, Any]] = []

        # Add images
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            })

        # Add text messages
        for msg in messages:
            content.append({"type": "text", "text": msg["content"]})

        full_messages = [{"role": "user", "content": content}]
        if system:
            full_messages.insert(0, {"role": "system", "content": system})

        try:
            resp = self._client.post(
                f"{self._api_base}/chat/completions",
                headers=self._headers,
                json={
                    "model": self.config.model,
                    "messages": full_messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Vision chat failed: %s", e)
            return None

    def _vision_anthropic(
        self,
        messages: List[Dict[str, str]],
        images: List[str],
        system: Optional[str] = None,
    ) -> Optional[str]:
        content: List[Dict[str, Any]] = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img,
                },
            })
        for msg in messages:
            content.append({"type": "text", "text": msg["content"]})

        body: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            body["system"] = system

        try:
            resp = self._client.post(
                f"{self._api_base}/messages",
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]
        except Exception as e:
            logger.error("Anthropic vision failed: %s", e)
            return None

    def _vision_gemini(
        self,
        messages: List[Dict[str, str]],
        images: List[str],
        system: Optional[str] = None,
    ) -> Optional[str]:
        parts: List[Dict[str, Any]] = []
        for img in images:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img,
                },
            })
        for msg in messages:
            parts.append({"text": msg["content"]})

        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            resp = self._client.post(
                f"{self._api_base}/models/{self.config.model}:generateContent",
                headers=self._headers,
                params={"key": self.config.api_key},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error("Gemini vision failed: %s", e)
            return None

    # ── Streaming chat ──────────────────────────────────────────────

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response tokens."""
        import asyncio

        loop = asyncio.get_event_loop()
        result = self.chat(messages, system)
        if result:
            for chunk in result.split():
                yield chunk + " "
                loop.run_until_complete(asyncio.sleep(0))
        else:
            yield ""


# ── Factory ──────────────────────────────────────────────────────────

_PROVIDER_CACHE: Dict[str, LLMProvider] = {}


def get_llm_provider(config: LLMProviderConfig) -> LLMProvider:
    """Get or create an LLM provider instance."""
    cache_key = f"{config.provider}:{config.model}:{config.api_base}"
    if cache_key in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[cache_key]

    provider = LLMProvider(config)
    _PROVIDER_CACHE[cache_key] = provider
    return provider


def reset_provider_cache():
    _PROVIDER_CACHE.clear()
