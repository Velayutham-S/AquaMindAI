"""Supervisor Planner for AquaMind AI.

Single responsibility: assemble the planning prompt, call the Planner LLM
(OpenCode Zen ``deepseek-v4-flash-free`` via the OpenAI SDK), parse the returned
JSON, validate it, and return a strongly-typed ``PlannerDecision``.

It does NOT execute any specialist agent, retrieve SQL, retrieve documents, call
FAISS, run prediction models, generate answers, or update Conversation Memory.
It ONLY plans (decides which agent(s) should run and in what order).

Public entry point:
    Planner().plan(user_query, conversation_memory=None) -> PlannerDecision

Configuration mirrors the Data Agent's SQL generator: same base URL, API key
env var, model, deterministic temperature, and retry/backoff/throttle behaviour.
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

from .planner_models import (
    ConfigurationError,
    EmptyQueryError,
    PlannerApiError,
    PlannerDecision,
    PlannerResponseError,
)
from .prompt_builder import PromptBuilder
from .planner_validator import PlannerValidator

logger = logging.getLogger("aquamind.supervisor.planner")

# --------------------------------------------------------------------------- #
# Paths & configuration (aligned with the Data Agent)
# --------------------------------------------------------------------------- #

PLANNER_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PLANNER_DIR.parents[2]  # planner -> supervisor_agent -> agents -> root
ENV_PATH: Path = PROJECT_ROOT / ".env"

MODEL_NAME: str = "deepseek-v4-flash-free"
API_KEY_ENV_VAR: str = "OPENCODE_API_KEY"
OPENCODE_BASE_URL: str = "https://opencode.ai/zen/v1"

DEFAULT_REQUESTS_PER_MINUTE: int = 15
RPM_ENV_VAR: str = "REQUESTS_PER_MINUTE"

REQUEST_TIMEOUT_SECONDS: int = 60
MAX_RETRIES: int = 5
RETRY_BASE_DELAY_SECONDS: float = 5.0
RETRY_MAX_DELAY_SECONDS: float = 60.0
PLANNING_TEMPERATURE: float = 0.0  # deterministic routing
MAX_OUTPUT_TOKENS: int = 1024


def _resolve_api_key() -> str:
    load_dotenv(ENV_PATH)
    api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise ConfigurationError(
            f"{API_KEY_ENV_VAR} is not set. Add it to {ENV_PATH} or the environment."
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
        logger.warning("Invalid %s=%r; using %d.", RPM_ENV_VAR, raw, DEFAULT_REQUESTS_PER_MINUTE)
        return DEFAULT_REQUESTS_PER_MINUTE


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse exactly one JSON object from the model output.

    Tolerates a surrounding markdown code fence and leading/trailing prose by
    extracting the first balanced ``{...}`` block. Raises
    ``PlannerResponseError`` if no valid JSON object can be parsed.
    """
    text = (raw_text or "").strip()
    if not text:
        raise PlannerResponseError("Planner returned an empty response.", raw=raw_text)

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
        except json.JSONDecodeError as error:
            raise PlannerResponseError(
                f"Planner response was not valid JSON: {error}", raw=raw_text
            ) from error

    raise PlannerResponseError("No JSON object found in planner response.", raw=raw_text)


class Planner:
    """Assembles the prompt, calls the LLM, and returns a validated decision."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = MODEL_NAME,
        prompt_builder: PromptBuilder | None = None,
        validator: PlannerValidator | None = None,
        requests_per_minute: int | None = None,
        request_timeout: int = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        # Dependency injection with sensible production defaults.
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._validator = validator or PlannerValidator()

        key = api_key or _resolve_api_key()
        self._model_name = model_name
        self._client = OpenAI(
            base_url=OPENCODE_BASE_URL,
            api_key=key,
            timeout=request_timeout,
            max_retries=0,  # backoff handled below
        )

        rpm = requests_per_minute or _resolve_rpm()
        self.requests_per_minute = rpm
        self._min_interval = 60.0 / rpm
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._last_call_time = 0.0

        # Observability for the most recent plan() call.
        self.rate_limit_hits = 0
        self.last_planning_seconds = 0.0
        self.last_prompt_length = 0
        self.last_raw_response = ""
        logger.info("Planner ready (model=%s, rpm=%d).", model_name, rpm)

    # -- public API ------------------------------------------------------- #

    def plan(
        self,
        user_query: str,
        conversation_memory: str | dict[str, Any] | None = None,
    ) -> PlannerDecision:
        """Return a validated ``PlannerDecision`` for ``user_query``.

        ``conversation_memory`` may be a pre-rendered text block, a session
        snapshot dict (``Session.to_dict()``), or ``None``.

        Raises ``EmptyQueryError``, ``PlannerApiError``, ``PlannerResponseError``
        or ``PlannerValidationError``.
        """
        if not user_query or not user_query.strip():
            raise EmptyQueryError("User query is empty.")

        prompt = self._prompt_builder.build(conversation_memory, user_query)
        self.last_prompt_length = len(prompt)

        raw = self._call_llm(prompt)
        self.last_raw_response = raw

        payload = extract_json_object(raw)
        return self._validator.validate(payload)

    # -- rate limiting ---------------------------------------------------- #

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.monotonic()

    # -- LLM call --------------------------------------------------------- #

    def _call_llm(self, prompt: str) -> str:
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            self._throttle()
            try:
                start = time.monotonic()
                completion = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=PLANNING_TEMPERATURE,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                text = completion.choices[0].message.content if completion.choices else None
                if not text or not text.strip():
                    # An empty completion is a transient free-tier hiccup; retry it.
                    last_error = PlannerApiError("Planner LLM returned an empty response.")
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Empty response on attempt %d/%d; retrying in %.1fs.",
                        attempt, self._max_retries, delay,
                    )
                    time.sleep(delay)
                    continue
                self.last_planning_seconds = time.monotonic() - start
                return text
            except openai.RateLimitError as error:
                last_error = error
                self.rate_limit_hits += 1
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Rate limited (429) on attempt %d/%d; backing off %.1fs.",
                    attempt, self._max_retries, delay,
                )
                time.sleep(delay)
            except (openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError) as error:
                last_error = error
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Transient LLM error on attempt %d/%d (%s); retrying in %.1fs.",
                    attempt, self._max_retries, type(error).__name__, delay,
                )
                time.sleep(delay)
            except openai.APIError as error:
                raise PlannerApiError(f"Planner LLM API call failed: {error}") from error

        raise PlannerApiError(
            f"Planner LLM API call failed after {self._max_retries} retries: {last_error}"
        )

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        return min(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SECONDS)
