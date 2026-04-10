"""
LLM client — app/core/llm.py

Unified interface to talk to any LLM provider in NeuroGraph.
Every agent (planner, critic, writer) uses this module instead
of making raw HTTP calls.

Supported providers:
    - Ollama (local): any model via localhost:11434/v1
    - Groq (cloud): free tier via api.groq.com/openai/v1

Both expose OpenAI-compatible chat/completions endpoints, so
one client handles both. The provider is selected by config.

Architecture:
    LLMClient (core client)
      -> complete()          raw text completion
      -> complete_json()     structured JSON output with validation
      -> complete_chat()     multi-turn conversation

    get_client()             factory, returns configured client
    get_chat_model()         LangChain ChatModel for LangGraph (Phase 3)

Design decisions:
    - httpx over requests: async-ready, better timeout handling
    - OpenAI-compatible API over provider SDKs: one client for all
    - Retry with exponential backoff: local servers are flaky
    - Token tracking: know your costs and context usage
    - JSON extraction: regex fallback when LLM wraps JSON in markdown
    - Temperature per call: planner needs low temp, writer needs higher
    - System prompt separation: keeps agent prompts clean
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# Response data structure

@dataclass
class LLMResponse:
    """
    Structured response from any LLM call.

    Wraps the raw text with metadata about the call itself.
    Token counts help track context window usage and cost.
    The parsed field holds extracted JSON when complete_json() is used.
    """
    text: str                                   # raw model output
    parsed: dict | list | None = None           # extracted JSON if applicable
    model: str = ""                             # which model responded
    provider: str = ""                          # ollama or groq
    prompt_tokens: int = 0                      # tokens in the prompt
    completion_tokens: int = 0                  # tokens in the response
    total_tokens: int = 0                       # sum of both
    latency_ms: float = 0.0                     # round-trip time
    attempts: int = 1                           # how many tries it took

    @property
    def has_parsed(self) -> bool:
        return self.parsed is not None


# JSON extraction

def _extract_json(text: str) -> dict | list | None:
    """
    Extract JSON from LLM output, handling common formatting issues.

    LLMs often wrap JSON in markdown code fences or add preamble
    text before the JSON object. This function handles:

    1. Clean JSON (just parse it)
    2. ```json ... ``` fenced blocks
    3. ``` ... ``` generic fenced blocks
    4. JSON embedded in prose (find first { or [)

    Returns None if no valid JSON can be extracted.
    """
    cleaned = text.strip()

    # Try direct parse first (fastest path)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    fence_patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",
        r"```\s*\n?(.*?)\n?\s*```",
    ]
    for pattern in fence_patterns:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Try finding JSON object or array in the text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = cleaned.find(start_char)
        if start_idx == -1:
            continue

        # Find the matching closing bracket by counting depth
        depth = 0
        for i in range(start_idx, len(cleaned)):
            if cleaned[i] == start_char:
                depth += 1
            elif cleaned[i] == end_char:
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start_idx : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    return None


# Provider configuration

@dataclass
class ProviderConfig:
    """
    Connection details and capabilities for one LLM provider.

    Encapsulates base URL, model name, API key, and capability
    flags so the client handles provider differences explicitly
    rather than through exception-driven fallbacks.
    """
    name: str
    base_url: str
    model: str
    api_key: str = ""
    extra_headers: dict = field(default_factory=dict)

    # Capability flags — set per provider in _build_provider_config()
    supports_json_mode: bool = False        # can accept response_format=json_object
    supports_token_usage: bool = True       # returns usage stats in response
    supports_system_role: bool = True       # accepts system messages

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h


def _build_provider_config() -> ProviderConfig:
    """
    Build provider config from app settings.

    Ollama uses localhost, no API key.
    Groq uses their cloud endpoint with a Bearer token.
    Both speak OpenAI-compatible chat/completions.

    Capability flags are set explicitly per provider so the
    client never relies on try/except to discover features.
    """
    if settings.llm_provider == "groq":
        return ProviderConfig(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            supports_json_mode=True,
            supports_token_usage=True,
        )

    # Default: Ollama (local)
    return ProviderConfig(
        name="ollama",
        base_url=f"{settings.ollama_base_url}/v1",
        model=settings.ollama_model,
        supports_json_mode=False,
        supports_token_usage=False,
    )


# Core LLM client

class LLMClient:
    """
    Unified LLM client for all NeuroGraph agents.

    Talks to any OpenAI-compatible endpoint (Ollama, Groq, etc.)
    via httpx. Handles retries, token tracking, JSON extraction,
    and structured output.

    Usage:
        client = get_client()

        # Simple text completion
        response = client.complete("What is active inference?")

        # Structured JSON output
        response = client.complete_json(
            prompt="Classify this query...",
            system="You are a query classifier. Respond in JSON.",
        )

        # Multi-turn conversation
        response = client.complete_chat([
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "Summarize this evidence..."},
        ])
    """

    DEFAULT_TIMEOUT = 120.0
    DEFAULT_RETRIES = 3
    DEFAULT_RETRY_DELAY = 2.0
    DEFAULT_TEMPERATURE = 0.1

    def __init__(
        self,
        provider: ProviderConfig | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ):
        self._provider = provider or _build_provider_config()
        self._timeout = timeout
        self._retries = retries
        self._retry_delay = retry_delay

        # Cumulative token tracking across all calls in this session
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_calls = 0

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def model_name(self) -> str:
        return self._provider.model

    @property
    def token_usage(self) -> dict:
        """Cumulative token usage across all calls."""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "total_calls": self._total_calls,
        }

    def _call_api(
        self,
        messages: list[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """
        Make a single API call with retry logic.

        Builds the request payload, sends it, parses the response,
        and tracks token usage. Retries on transient failures
        (timeouts, 5xx errors, connection issues).

        Args:
            messages: OpenAI-format message list
            temperature: randomness (0.0 = deterministic)
            max_tokens: max response length
            response_format: optional {"type": "json_object"} for JSON mode

        Returns:
            LLMResponse with text, token counts, and latency
        """
        import httpx

        payload: dict[str, Any] = {
            "model": self._provider.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Some providers support JSON mode natively
        if response_format:
            payload["response_format"] = response_format

        last_error: Exception | None = None
        attempts = 0

        for attempt in range(1, self._retries + 1):
            attempts = attempt
            start = time.monotonic()

            try:
                resp = httpx.post(
                    self._provider.chat_url,
                    headers=self._provider.headers,
                    json=payload,
                    timeout=self._timeout,
                )
                latency = (time.monotonic() - start) * 1000

                resp.raise_for_status()
                data = resp.json()

                # Extract response text
                text = data["choices"][0]["message"]["content"]

                # Extract token usage (may not be present in all providers)
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Track cumulative usage
                self._total_prompt_tokens += prompt_tokens
                self._total_completion_tokens += completion_tokens
                self._total_calls += 1

                return LLMResponse(
                    text=text,
                    model=self._provider.model,
                    provider=self._provider.name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=round(latency, 1),
                    attempts=attempts,
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code

                # Don't retry client errors (except 429 rate limit)
                if 400 <= status < 500 and status != 429:
                    raise RuntimeError(
                        f"LLM API error {status}: {e.response.text}"
                    ) from e

                logger.warning(
                    "LLM call attempt %d/%d failed (HTTP %d). Retrying in %.1fs",
                    attempt, self._retries, status, self._retry_delay,
                )

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                logger.warning(
                    "LLM call attempt %d/%d failed (%s). Retrying in %.1fs",
                    attempt, self._retries, type(e).__name__, self._retry_delay,
                )

            if attempt < self._retries:
                # Exponential backoff: 2s, 4s, 8s...
                delay = self._retry_delay * (2 ** (attempt - 1))
                time.sleep(delay)

        raise RuntimeError(
            f"LLM call failed after {self._retries} attempts: {last_error}"
        ) from last_error

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Simple text completion with optional system prompt.

        This is the most common call pattern. Send a prompt,
        get text back. Used by agents for reasoning steps.

        Args:
            prompt: the user message
            system: optional system instruction
            temperature: 0.0-1.0 (lower = more deterministic)
            max_tokens: max response length

        Returns:
            LLMResponse with the model's text in .text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self._call_api(messages, temperature, max_tokens)

    def complete_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        retry_parse: bool = True,
    ) -> LLMResponse:
        """
        Completion that expects structured JSON output.

        Attempts JSON mode if the provider supports it. Extracts
        JSON from the response text, handling markdown fences and
        preamble text. Optionally retries once with a stricter
        prompt if the first attempt returns unparseable output.

        The parsed JSON is stored in response.parsed. If parsing
        fails even after retry, response.parsed is None and the
        raw text is still available in response.text.

        Args:
            prompt: the user message (should ask for JSON output)
            system: system prompt (should mention JSON format)
            temperature: default 0.0 for deterministic structured output
            max_tokens: max response length
            retry_parse: if True, retry with stricter prompt on parse failure

        Returns:
            LLMResponse with .parsed containing the extracted dict/list
        """
        # Use JSON mode only if the provider explicitly supports it
        response_format = (
            {"type": "json_object"}
            if self._provider.supports_json_mode
            else None
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._call_api(
            messages, temperature, max_tokens, response_format
        )

        # Extract JSON from the response
        parsed = _extract_json(response.text)

        if parsed is not None:
            response.parsed = parsed
            return response

        # First parse failed. Retry with explicit instruction if allowed.
        if retry_parse:
            logger.warning("JSON parse failed, retrying with stricter prompt")

            retry_msg = (
                "Your previous response was not valid JSON. "
                "Respond with ONLY a JSON object, no markdown fences, "
                "no explanation, no text before or after the JSON. "
                "Start your response with { and end with }."
            )
            messages.append({"role": "assistant", "content": response.text})
            messages.append({"role": "user", "content": retry_msg})

            retry_response = self._call_api(messages, 0.0, max_tokens)
            parsed = _extract_json(retry_response.text)

            if parsed is not None:
                retry_response.parsed = parsed
                retry_response.attempts = response.attempts + retry_response.attempts
                return retry_response

            logger.error(
                "JSON parse failed after retry. Raw text: %s",
                retry_response.text[:200],
            )
            # Return the retry response with parsed=None
            retry_response.attempts = response.attempts + retry_response.attempts
            return retry_response

        return response

    def complete_chat(
        self,
        messages: list[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Multi-turn chat completion.

        Takes a full conversation history in OpenAI format:
        [{"role": "system"|"user"|"assistant", "content": "..."}]

        Used by agents that need conversational context, like
        the critic agent asking follow-up questions about evidence.

        Args:
            messages: full conversation history
            temperature: randomness
            max_tokens: max response length

        Returns:
            LLMResponse with the assistant's reply
        """
        return self._call_api(messages, temperature, max_tokens)

    def is_available(self) -> bool:
        """
        Check if the LLM provider is reachable.

        Makes a minimal health check request. Used by agents
        to decide whether to use LLM-based planning or fall
        back to rule-based planning.
        """
        import httpx

        try:
            if self._provider.name == "ollama":
                # Ollama health check: GET /api/tags
                base = self._provider.base_url.replace("/v1", "")
                resp = httpx.get(f"{base}/api/tags", timeout=5.0)
                return resp.status_code == 200

            # Groq/OpenAI: try a minimal completion
            resp = httpx.post(
                self._provider.chat_url,
                headers=self._provider.headers,
                json={
                    "model": self._provider.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
                timeout=10.0,
            )
            return resp.status_code == 200

        except Exception as e:
            logger.debug("Provider health check failed: %s", e)
            return False


# Factory functions

_client_cache: LLMClient | None = None


def get_client(force_new: bool = False) -> LLMClient:
    """
    Return a shared LLMClient instance.

    Cached so all agents share token tracking and connection
    settings. Pass force_new=True to create a fresh client
    (useful in tests or after config changes).
    """
    global _client_cache

    if _client_cache is None or force_new:
        _client_cache = LLMClient()
        logger.info(
            "LLM client initialized: provider=%s, model=%s",
            _client_cache.provider_name,
            _client_cache.model_name,
        )

    return _client_cache


def get_chat_model():
    """
    Return a LangChain ChatModel for LangGraph integration.

    Phase 3 uses LangGraph which needs LangChain chat models.
    This factory returns the right one based on config.

    Lazy import: LangChain is only loaded if this function is called.
    Agents that don't need LangGraph use get_client() instead.

    Returns:
        ChatOllama or ChatGroq instance
    """
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.1,
            max_tokens=2048,
        )

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        num_predict=2048,
    )