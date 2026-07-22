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

MODEL_NAME: str = "gemini-3.5-flash"
API_KEY_ENV_VAR: str = "GEMINI_SQL_API_KEY"
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

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

#: The only tables the generated SQL may reference (must match groundwater.db).
VALID_TABLES: frozenset[str] = frozenset({
    "district", "firka", "groundwater_level",
    "rainfall", "river_discharge", "river_water_level",
})


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


class ValidationError(SqlGeneratorError):
    """The generated SQL failed validation (malformed or unsafe).

    The offending SQL is attached as ``.sql`` for diagnostics.
    """

    def __init__(self, message: str, sql: str | None = None) -> None:
        super().__init__(message)
        self.sql = sql


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


def _referenced_tables(sql: str) -> set[str]:
    """Return the lower-cased table names referenced after FROM / JOIN.

    Subqueries (``FROM (SELECT ...)``) yield no name here; the inner table is
    still captured by its own FROM.
    """
    pattern = re.compile(r"(?i)\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)")
    return {match.group(1).lower() for match in pattern.finditer(sql)}


def validate_sql(sql: str) -> None:
    """Validate that ``sql`` is a single, safe, well-formed SQLite SELECT.

    Enforces, in order: non-empty; no markdown; no JSON; begins with SELECT;
    contains a FROM clause; at most one (trailing) semicolon / exactly one
    statement; no destructive keywords; and references only documented tables.

    Raises ``ValidationError`` (with a message describing every failed check and
    the offending SQL attached as ``.sql``) when the SQL is invalid. Returns
    ``None`` when the SQL is valid. It never silently returns a boolean.
    """
    text = (sql or "").strip()

    # Rule 7: reject empty / whitespace-only responses.
    if not text:
        raise ValidationError("SQL is empty or whitespace-only.", sql=sql)

    reasons: list[str] = []

    # Rule 5: reject markdown.
    if "```" in text:
        reasons.append("contains markdown code fences")

    # Rule 6: reject JSON.
    if text[0] in "{[":
        reasons.append("looks like JSON (starts with '{' or '[')")

    # Rule 2: must begin with SELECT.
    if not re.match(r"(?i)select\b", text):
        reasons.append("does not begin with SELECT")

    # Rule 3: must contain a FROM clause.
    if not re.search(r"(?i)\bfrom\b", text):
        reasons.append("missing FROM clause")

    # Rules 1 & 8: exactly one statement (at most a single trailing semicolon).
    if text.count(";") > 1:
        reasons.append("contains more than one semicolon (multiple statements)")
    elif ";" in text and not text.endswith(";"):
        reasons.append("contains an embedded semicolon (multiple statements)")

    # Rule 4: reject destructive / non-read-only keywords.
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"(?i)\b{keyword}\b", text):
            reasons.append(f"contains forbidden keyword '{keyword}'")

    # Only documented tables may be referenced.
    unknown_tables = sorted(_referenced_tables(text) - VALID_TABLES)
    if unknown_tables:
        reasons.append(f"references unknown table(s): {', '.join(unknown_tables)}")

    if reasons:
        raise ValidationError("Invalid SQL — " + "; ".join(reasons) + ".", sql=text)


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
            base_url=GEMINI_BASE_URL,
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
        """Return exactly one validated SQLite SELECT query for ``user_query``.

        The generated SQL is validated before it is returned, so malformed or
        unsafe SQL never reaches the executor.

        Raises ``EmptyQueryError`` for a blank query, ``LlmApiError`` on API
        failure or an empty/unusable response, and ``ValidationError`` if the
        generated SQL fails validation.
        """
        if not user_query or not user_query.strip():
            raise EmptyQueryError("User query is empty.")

        prompt = self._build_prompt(user_query)
        raw = self._call_llm(prompt)
        sql = clean_sql(raw)
        if not sql:
            raise LlmApiError("LLM response contained no SQL after cleaning.")
        validate_sql(sql)
        return sql


# --------------------------------------------------------------------------- #
# Minimal smoke test (invoked only when this module is run directly)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generator = SqlGenerator()
    print(generator.generate("What is the groundwater level in Salem?"))
