"""Centralized configuration + shared LLM client for the AquaMind AI Recommendation Layer.

Both Recommendation-Layer components (the Recommendation Decision and the
Recommendation Generator) call the SAME LLM (OpenCode Zen ``deepseek-v4-flash-free``
via the OpenAI SDK) with deterministic settings. Rather than duplicate the
throttle / retry / backoff logic in each component, that behaviour lives once in
:class:`LlmClient` here and is dependency-injected into both components.

This module mirrors the existing Supervisor Planner / Data-Agent LLM
configuration (same base URL, API key env var, model, temperature and
rate-limit handling); it introduces no new provider and no new environment
variables.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger("aquamind.recommendation")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

RECOMMENDATION_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = RECOMMENDATION_DIR.parents[1]  # recommendation_agent -> agents -> root
ENV_PATH: Path = PROJECT_ROOT / ".env"

DECISION_PROMPT_PATH: Path = RECOMMENDATION_DIR / "decision" / "decision_prompt.txt"
GENERATOR_PROMPT_PATH: Path = RECOMMENDATION_DIR / "generator" / "recommendation_prompt.txt"

# --------------------------------------------------------------------------- #
# LLM configuration (aligned with the Supervisor Planner / Data Agent)
# --------------------------------------------------------------------------- #

# Recommendation Decision LLM -> OpenCode Zen (DeepSeek Flash 4). Model and
# request/response format are unchanged; only the provider API key differs.
MODEL_NAME: str = "deepseek-v4-flash-free"
API_KEY_ENV_VAR: str = "OPENCODE_API_KEY"
OPENCODE_BASE_URL: str = "https://opencode.ai/zen/v1"

# Recommendation Generator LLM -> Groq (llama-3.3-70b-versatile) via its
# OpenAI-compatible endpoint. Same client behaviour, different provider.
GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"
GROQ_API_KEY_ENV_VAR: str = "GROQ_API_KEY"
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

DEFAULT_REQUESTS_PER_MINUTE: int = 15
RPM_ENV_VAR: str = "REQUESTS_PER_MINUTE"

REQUEST_TIMEOUT_SECONDS: int = 60
MAX_RETRIES: int = 5
RETRY_BASE_DELAY_SECONDS: float = 5.0
RETRY_MAX_DELAY_SECONDS: float = 60.0
TEMPERATURE: float = 0.0  # deterministic outputs
MAX_OUTPUT_TOKENS: int = 2048  # headroom so multi-recommendation JSON is never truncated


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class RecommendationError(Exception):
    """Base error for the Recommendation Layer."""


class ConfigurationError(RecommendationError):
    """Missing/invalid configuration (e.g. API key or prompt file)."""


class LlmApiError(RecommendationError):
    """The LLM API call failed after retries or returned no usable text."""


class ResponseParsingError(RecommendationError):
    """The LLM response could not be parsed into the expected JSON object."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_prompt(path: Path) -> str:
    """Load a prompt file; raise ``ConfigurationError`` if missing/empty."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"Prompt file not found/readable: {path}") from error
    if not text.strip():
        raise ConfigurationError(f"Prompt file is empty: {path}")
    return text


def _resolve_api_key(env_var: str = API_KEY_ENV_VAR) -> str:
    load_dotenv(ENV_PATH)
    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        raise ConfigurationError(
            f"{env_var} is not set. Add it to {ENV_PATH} or the environment."
        )
    return api_key


def _resolve_rpm() -> int:
    raw = os.environ.get(RPM_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_REQUESTS_PER_MINUTE
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_REQUESTS_PER_MINUTE
    except ValueError:
        return DEFAULT_REQUESTS_PER_MINUTE


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse exactly one JSON object from the model output.

    Tolerates a surrounding markdown code fence and leading/trailing prose by
    extracting the first balanced ``{...}`` block. Raises ``ResponseParsingError``
    if no valid JSON object can be parsed.
    """
    text = (raw_text or "").strip()
    if not text:
        raise ResponseParsingError("LLM returned an empty response.")

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Conservative repair for the most common LLM slip: trailing commas
            # before a closing } or ]. (Does not attempt anything unsafe.)
            repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as error:
                raise ResponseParsingError(f"Response was not valid JSON: {error}") from error

    raise ResponseParsingError("No JSON object found in the LLM response.")


# --------------------------------------------------------------------------- #
# Shared LLM client (dependency-injected into both components)
# --------------------------------------------------------------------------- #

class LlmClient:
    """Deterministic OpenCode Zen client with throttling and retry/backoff.

    A single instance is shared by the Recommendation Decision and the
    Recommendation Generator, so rate limiting is respected across both.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = MODEL_NAME,
        requests_per_minute: int | None = None,
        request_timeout: int = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        temperature: float = TEMPERATURE,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        base_url: str = OPENCODE_BASE_URL,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        key = api_key or _resolve_api_key()
        self._model_name = model_name
        self._client = OpenAI(
            base_url=base_url,
            api_key=key,
            timeout=request_timeout,
            max_retries=0,  # backoff handled here
        )
        rpm = requests_per_minute or _resolve_rpm()
        self.requests_per_minute = rpm
        self._min_interval = 60.0 / rpm
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        # Optional reasoning-effort cap. Only sent when explicitly set, so
        # providers/models that do not accept it (and existing callers) are
        # unaffected. Used to stop reasoning models (e.g. gemini-3.5-flash) from
        # spending the output-token budget on internal thinking and truncating
        # the JSON reply.
        self._reasoning_effort = reasoning_effort
        # Optional response-format constraint (e.g. {"type": "json_object"}).
        # Only sent when explicitly set, so existing callers are unaffected. When
        # enabled, the provider guarantees syntactically valid JSON output
        # (including correct escaping of quotes inside string values).
        self._response_format = response_format
        self._last_call_time = 0.0
        self.rate_limit_hits = 0
        self.last_call_seconds = 0.0
        logger.info("Recommendation LlmClient ready (model=%s, rpm=%d).", model_name, rpm)

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, prompt: str) -> str:
        """Return the model's text completion for ``prompt`` (retried on transients)."""
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            self._throttle()
            try:
                start = time.monotonic()
                create_kwargs: dict[str, Any] = {
                    "model": self._model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self._temperature,
                    "max_tokens": self._max_output_tokens,
                }
                if self._reasoning_effort is not None:
                    create_kwargs["reasoning_effort"] = self._reasoning_effort
                if self._response_format is not None:
                    create_kwargs["response_format"] = self._response_format
                completion = self._client.chat.completions.create(**create_kwargs)
                text = completion.choices[0].message.content if completion.choices else None
                if not text or not text.strip():
                    last_error = LlmApiError("LLM returned an empty response.")
                    time.sleep(self._backoff_delay(attempt))
                    continue
                self.last_call_seconds = time.monotonic() - start
                return text
            except openai.RateLimitError as error:
                last_error = error
                self.rate_limit_hits += 1
                time.sleep(self._backoff_delay(attempt))
            except (openai.APITimeoutError, openai.APIConnectionError,
                    openai.InternalServerError) as error:
                last_error = error
                time.sleep(self._backoff_delay(attempt))
            except openai.APIError as error:
                raise LlmApiError(f"LLM API call failed: {error}") from error
        raise LlmApiError(
            f"LLM API call failed after {self._max_retries} retries: {last_error}"
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.monotonic()

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        return min(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SECONDS)


# --------------------------------------------------------------------------- #
# Provider-specific client factories
# --------------------------------------------------------------------------- #

def build_generator_client() -> "LlmClient":
    """LLM client for the Recommendation Generator -> Groq (llama-3.3-70b-versatile).

    Uses Groq's OpenAI-compatible endpoint so the client behaviour (throttle,
    retry/backoff, prompt/response format) stays identical; only the provider,
    model and API key differ.
    """
    return LlmClient(
        api_key=_resolve_api_key(GROQ_API_KEY_ENV_VAR),
        model_name=GROQ_MODEL_NAME,
        base_url=GROQ_BASE_URL,
    )


# --------------------------------------------------------------------------- #
# Evidence view (what the Recommendation Layer is allowed to see)
# --------------------------------------------------------------------------- #

def build_evidence_view(aggregated: Any) -> dict[str, Any]:
    """Project an AggregatedUniversalAgentResponse down to its agent evidence.

    Accepts either an ``AggregatedUniversalAgentResponse`` (anything with
    ``to_dict``) or a plain dict. It deliberately EXCLUDES the planner decision,
    request/session ids and the original user query, so the Recommendation Layer
    reasons ONLY over the specialist agents' evidence and outcome -- never over
    the query, memory, or planner decision.
    """
    data = aggregated.to_dict() if hasattr(aggregated, "to_dict") else dict(aggregated)
    return {
        "status": data.get("status"),
        "executed_agents": list(data.get("executed_agents", [])),
        "failed_agents": list(data.get("failed_agents", [])),
        "responses": list(data.get("responses", [])),
    }
