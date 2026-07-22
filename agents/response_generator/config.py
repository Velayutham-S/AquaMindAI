"""Configuration for the AquaMind AI Response Generator.

The Response Generator reuses the project's established deterministic OpenCode
Zen client and strict JSON parser. It does not define a second provider client or
introduce new environment variables.
"""

from __future__ import annotations

from pathlib import Path

from recommendation_agent.config import (
    ConfigurationError,
    LlmApiError,
    LlmClient,
    ResponseParsingError,
    _resolve_api_key,
    extract_json_object,
    load_prompt,
)

RESPONSE_GENERATOR_DIR = Path(__file__).resolve().parent
RESPONSE_SYSTEM_PROMPT_PATH = RESPONSE_GENERATOR_DIR / "response_system_prompt.txt"

# Response Generator LLM -> Google Gemini (gemini-3.5-flash) via its
# OpenAI-compatible endpoint. Reuses the shared LlmClient behaviour (throttle,
# retry/backoff, prompt/response format); only the provider, model and API key
# differ.
GEMINI_RESPONSE_MODEL_NAME: str = "gemini-3.5-flash"
GEMINI_RESPONSE_API_KEY_ENV_VAR: str = "GEMINI_RESPONSE_API_KEY"
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

# gemini-3.5-flash is a THINKING model: its internal reasoning tokens are drawn
# from the same output-token budget as the reply. Unlike short data/prediction
# answers, long knowledge/mixed answers combined with uncapped reasoning exhaust
# a tight budget, so the completion is truncated (finish_reason="length") BEFORE
# the closing brace -- leaving JSON with an opening '{' but no '}', which
# extract_json_object() correctly rejects as "No JSON object found". Cap thinking
# to "low" and give the (potentially long) final answer ample room. This mirrors
# the identical fix already applied to the Supervisor Planner.
GEMINI_RESPONSE_MAX_OUTPUT_TOKENS: int = 8192
GEMINI_RESPONSE_REASONING_EFFORT: str = "low"


def build_response_client() -> LlmClient:
    """LLM client for the Response Generator -> Gemini (gemini-3.5-flash)."""
    return LlmClient(
        api_key=_resolve_api_key(GEMINI_RESPONSE_API_KEY_ENV_VAR),
        model_name=GEMINI_RESPONSE_MODEL_NAME,
        base_url=GEMINI_BASE_URL,
        max_output_tokens=GEMINI_RESPONSE_MAX_OUTPUT_TOKENS,
        reasoning_effort=GEMINI_RESPONSE_REASONING_EFFORT,
    )


__all__ = [
    "ConfigurationError",
    "LlmApiError",
    "LlmClient",
    "ResponseParsingError",
    "build_response_client",
    "extract_json_object",
    "load_prompt",
    "RESPONSE_SYSTEM_PROMPT_PATH",
]
