"""SQL Generation LLM for the AquaMind AI Data Agent.

Single responsibility: convert a natural-language groundwater question into
exactly one valid SQLite ``SELECT`` query, using OpenCode Zen
(``deepseek-v4-flash-free``) through the OpenAI SDK.

This module does NOT execute SQL, connect to SQLite for execution, retrieve
rows, answer the user's question, format evidence, or build any downstream
response object. Its only output is one SQL query string.

Inputs (all loaded dynamically at runtime, never hardcoded):
  1. ``llm_inputs/sql_system_prompt.txt`` — instructions for the model.
  2. ``llm_inputs/database_schema.md``    — the authoritative schema reference.
  3. The user's question (passed by the Supervisor Agent in production).

The public entry point is ``SqlGenerator.generate(user_query) -> str``. The
Supervisor Agent calls this method with the original user query.

Run directly for a minimal smoke test:
    python sql_generator.py
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import openai
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger("aquamind.sql_generator")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

LLM_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = LLM_DIR.parents[2]  # llm -> data_agent -> agents -> root
INPUTS_DIR: Path = LLM_DIR / "llm_inputs"
SYSTEM_PROMPT_PATH: Path = INPUTS_DIR / "sql_system_prompt.txt"
SCHEMA_PATH: Path = INPUTS_DIR / "database_schema.md"
ENV_PATH: Path = PROJECT_ROOT / ".env"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_NAME: str = "deepseek-v4-flash-free"
API_KEY_ENV_VAR: str = "OPENCODE_API_KEY"
OPENCODE_BASE_URL: str = "https://opencode.ai/zen/v1"

#: Requests-per-minute cap for LLM calls. Configurable via env; not hardcoded
#: to a provider-specific value. Used to space out sequential calls.
DEFAULT_REQUESTS_PER_MINUTE: int = 15
RPM_ENV_VAR: str = "REQUESTS_PER_MINUTE"

REQUEST_TIMEOUT_SECONDS: int = 60
MAX_RETRIES: int = 5
RETRY_BASE_DELAY_SECONDS: float = 5.0
RETRY_MAX_DELAY_SECONDS: float = 60.0
GENERATION_TEMPERATURE: float = 0.0  # deterministic SQL
MAX_OUTPUT_TOKENS: int = 2048

#: SQL keywords that must never appear in generated output (read-only guarantee).
FORBIDDEN_SQL_KEYWORDS: tuple[str, ...] = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REPLACE",
)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class SqlGeneratorError(Exception):
    """Base error for the SQL Generation LLM."""


class ConfigurationError(SqlGeneratorError):
    """Missing/invalid configuration (e.g. API key)."""


class InputFileError(SqlGeneratorError):
    """A required input file (system prompt or schema) is missing/unreadable."""


class EmptyQueryError(SqlGeneratorError):
    """The user query was empty or blank."""


class LlmApiError(SqlGeneratorError):
    """The LLM API call failed (after retries) or returned no usable text."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_text_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise InputFileError(f"Required input file not found: {path}") from error
    except OSError as error:
        raise InputFileError(f"Could not read input file {path}: {error}") from error
    if not text.strip():
        raise InputFileError(f"Required input file is empty: {path}")
    return text


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
        logger.warning("Invalid %s=%r; falling back to %d.", RPM_ENV_VAR, raw, DEFAULT_REQUESTS_PER_MINUTE)
        return DEFAULT_REQUESTS_PER_MINUTE


def clean_sql(raw_text: str) -> str:
    """Strip markdown fences / stray labels and return the bare SQL statement."""
    text = raw_text.strip()

    # Remove a fenced code block if the model wrapped the SQL in one.
    fence = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        # Remove stray leading fence / "sql" label lines if only one side present.
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    text = text.strip()
    # Drop a single trailing semicolon so the result is one clean statement.
    if text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def validate_sql(sql: str) -> tuple[bool, list[str]]:
    """Check the SQL is a single, read-only ``SELECT`` with no markdown/JSON.

    Returns ``(is_valid, reasons)``; ``reasons`` lists every failed check.
    """
    reasons: list[str] = []
    text = sql.strip()

    if not text:
        return False, ["empty SQL"]

    if "```" in text:
        reasons.append("contains markdown fences")
    if text[0] in "{[":
        reasons.append("looks like JSON")

    lowered = text.lstrip().lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        reasons.append("does not start with SELECT/WITH")

    # Exactly one statement: no semicolons remain after the trailing one is stripped.
    if ";" in text:
        reasons.append("contains multiple statements (embedded ';')")

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", text, flags=re.IGNORECASE):
            reasons.append(f"contains forbidden keyword '{keyword}'")

    return (not reasons), reasons


# --------------------------------------------------------------------------- #
# SQL Generator
# --------------------------------------------------------------------------- #

class SqlGenerator:
    """Generates one SQLite SELECT query from a natural-language question."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = MODEL_NAME,
        requests_per_minute: int | None = None,
        request_timeout: int = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._system_prompt = _load_text_file(SYSTEM_PROMPT_PATH)
        self._schema = _load_text_file(SCHEMA_PATH)

        key = api_key or _resolve_api_key()
        self._model_name = model_name
        # Disable the SDK's own retries; retry/backoff is handled below.
        self._client = OpenAI(
            base_url=OPENCODE_BASE_URL,
            api_key=key,
            timeout=request_timeout,
            max_retries=0,
        )

        rpm = requests_per_minute or _resolve_rpm()
        self.requests_per_minute = rpm
        self._min_interval = 60.0 / rpm
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._last_call_time = 0.0
        self.rate_limit_hits = 0            # number of 429 responses observed
        self.last_generation_seconds = 0.0  # latency of the most recent API call
        logger.info("SqlGenerator ready (model=%s, rpm=%d).", model_name, rpm)

    # -- prompt ----------------------------------------------------------- #

    def _build_prompt(self, user_query: str) -> str:
        return (
            f"{self._system_prompt}\n\n"
            "==========================================================\n"
            "DATABASE SCHEMA (authoritative)\n"
            "==========================================================\n"
            f"{self._schema}\n\n"
            "==========================================================\n"
            "USER QUESTION\n"
            "==========================================================\n"
            f"{user_query.strip()}\n\n"
            "Return only the SQLite SELECT query."
        )

    # -- rate limiting ---------------------------------------------------- #

    def _throttle(self) -> None:
        """Space sequential requests to respect the requests-per-minute cap."""
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
                    temperature=GENERATION_TEMPERATURE,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                text = completion.choices[0].message.content if completion.choices else None
                if not text or not text.strip():
                    raise LlmApiError("LLM returned an empty response.")
                self.last_generation_seconds = time.monotonic() - start
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
                raise LlmApiError(f"LLM API call failed: {error}") from error

        raise LlmApiError(
            f"LLM API call failed after {self._max_retries} retries: {last_error}"
        )

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        return min(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SECONDS)

    # -- public API ------------------------------------------------------- #

    def generate(self, user_query: str) -> str:
        """Return exactly one SQLite SELECT query for ``user_query``.

        Raises ``EmptyQueryError`` for a blank query and ``LlmApiError`` on
        API failure or an empty/unusable response.
        """
        if not user_query or not user_query.strip():
            raise EmptyQueryError("User query is empty.")

        prompt = self._build_prompt(user_query)
        raw = self._call_llm(prompt)
        sql = clean_sql(raw)
        if not sql:
            raise LlmApiError("LLM response contained no SQL after cleaning.")
        return sql


# --------------------------------------------------------------------------- #
# Minimal smoke test (invoked only when this module is run directly)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generator = SqlGenerator()
    print(generator.generate("What is the groundwater level in Salem?"))
