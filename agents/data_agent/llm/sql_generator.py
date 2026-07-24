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

import hashlib
import logging
import os
import re
import time
from pathlib import Path

import openai
from openai import OpenAI
from dotenv import load_dotenv

from sql_intent_validator import IntentValidationError, SqlIntentValidator

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

#: Total SQL-generation attempts (1 initial + regenerations). When the generated
#: SQL fails intent/safety validation, the generator regenerates with corrective
#: feedback appended to the prompt, up to this many attempts.
MAX_GENERATION_ATTEMPTS: int = 3

#: Empty / unusable LLM responses are sometimes transient at the provider. The
#: SQL Generator retries the model call this many times total before giving up.
#: Verification contract: three empty responses in a row stop the pipeline.
EMPTY_RESPONSE_MAX_ATTEMPTS: int = 3
#: Backoff (seconds) applied BEFORE each retry after an empty response
#: (index 0 -> before the 2nd attempt, index 1 -> before the 3rd, ...).
EMPTY_RESPONSE_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0)

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


class EmptyLlmResponseError(LlmApiError):
    """The LLM returned an empty / unusable response.

    Empty means any of: ``None``, an empty or whitespace-only string, or the
    degenerate literals ``{}``, ``[]`` or ``null``. Raised only after the
    empty-response retries are exhausted. Subclasses ``LlmApiError`` so existing
    ``except LlmApiError`` / ``except SqlGeneratorError`` handling still applies,
    while the distinct class name lets the pipeline identify this exact cause.
    """


class ValidationError(SqlGeneratorError):
    """The generated SQL failed validation (malformed, unsafe, or not minimal).

    The offending SQL is attached as ``.sql`` for diagnostics.
    """

    def __init__(self, message: str, sql: str | None = None) -> None:
        super().__init__(message)
        self.sql = sql


class ClarificationNeededError(SqlGeneratorError):
    """The question is under-specified for the Data Agent (no location).

    Raised BEFORE any LLM call when the user names no district or firka and the
    question is not an aggregate/ranking/"all" query, so no minimal SQL can be
    written. ``clarification`` carries a user-facing prompt for a location.
    """

    def __init__(self, clarification: str) -> None:
        super().__init__(clarification)
        self.clarification = clarification


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


def _is_nonempty_llm_text(text: str | None) -> bool:
    """True only when ``text`` is a usable, non-empty model response.

    Treats as EMPTY / invalid: ``None``, empty string, whitespace-only, and the
    degenerate literals ``{}``, ``[]``, ``null`` and ``None``.
    """
    if text is None:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped in ("{}", "[]", "null", "None"):
        return False
    return True


def _provider_label(base_url: str) -> str:
    """Human-readable provider name derived from the client base URL (for logs)."""
    url = (base_url or "").lower()
    if "opencode" in url:
        return "OpenCode Zen"
    if "groq" in url:
        return "Groq"
    if "googleapis" in url or "gemini" in url:
        return "Gemini"
    if "openai.com" in url:
        return "OpenAI"
    return base_url or "unknown"


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
        max_generation_attempts: int = MAX_GENERATION_ATTEMPTS,
    ) -> None:
        self._system_prompt = _load_text_file(SYSTEM_PROMPT_PATH)
        self._schema = _load_text_file(SCHEMA_PATH)

        # Intent-aware minimality validation (Data-Agent-only, no LLM/no exec).
        self._intent_validator = SqlIntentValidator()
        self._max_generation_attempts = max(1, max_generation_attempts)

        key = api_key or _resolve_api_key()
        self._model_name = model_name
        # Base URL is read from the module global at construction time (the
        # production wiring overrides it to the OpenCode Zen endpoint). Captured
        # here for diagnostic logging (provider label).
        self._base_url = GEMINI_BASE_URL
        self._provider_label = _provider_label(GEMINI_BASE_URL)
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

    def _build_prompt(self, user_query: str, feedback: str | None = None) -> str:
        prompt = (
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
        if feedback:
            # A prior attempt was rejected; steer the regeneration.
            prompt += (
                "\n\n==========================================================\n"
                "REGENERATION FEEDBACK (the previous SQL was rejected)\n"
                "==========================================================\n"
                f"{feedback}\n"
                "Return only the corrected SQLite SELECT query."
            )
        return prompt

    # -- rate limiting ---------------------------------------------------- #

    def _throttle(self) -> None:
        """Space sequential requests to respect the requests-per-minute cap."""
        elapsed = time.monotonic() - self._last_call_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.monotonic()

    # -- LLM call --------------------------------------------------------- #

    def _invoke_model(self, prompt: str) -> tuple[str | None, str | None]:
        """One logical model call with rate-limit / transient-error retries.

        Returns ``(text, finish_reason)`` verbatim from the provider (the text
        may be empty/None -- emptiness is validated by the caller). Raises
        ``LlmApiError`` only on a non-transient API failure or after exhausting
        transient retries.
        """
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
                self.last_generation_seconds = time.monotonic() - start
                if completion.choices:
                    choice = completion.choices[0]
                    return choice.message.content, getattr(choice, "finish_reason", None)
                return None, "no_choices"
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

    def _call_llm(self, prompt: str) -> str:
        """Return a validated, non-empty model response.

        The response is validated immediately: ``None``, empty, whitespace-only,
        or the degenerate literals ``{}`` / ``[]`` / ``null`` are treated as
        empty. Empty responses are retried (with backoff) up to
        ``EMPTY_RESPONSE_MAX_ATTEMPTS`` times; if every attempt is empty,
        ``EmptyLlmResponseError`` is raised and NO invalid text is returned.
        """
        prompt_len = len(prompt)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        for attempt in range(1, EMPTY_RESPONSE_MAX_ATTEMPTS + 1):
            text, finish_reason = self._invoke_model(prompt)
            response_len = len(text) if text else 0
            logger.info(
                "SQL LLM call | provider=%s model=%s attempt=%d/%d prompt_len=%d "
                "prompt_hash=%s response_len=%d finish_reason=%s",
                self._provider_label, self._model_name, attempt,
                EMPTY_RESPONSE_MAX_ATTEMPTS, prompt_len, prompt_hash,
                response_len, finish_reason,
            )

            if _is_nonempty_llm_text(text):
                return text

            logger.error(
                "SQL Generator received an EMPTY LLM response | provider=%s model=%s "
                "attempt=%d/%d prompt_len=%d prompt_hash=%s response_len=%d "
                "finish_reason=%s",
                self._provider_label, self._model_name, attempt,
                EMPTY_RESPONSE_MAX_ATTEMPTS, prompt_len, prompt_hash,
                response_len, finish_reason,
            )
            if attempt < EMPTY_RESPONSE_MAX_ATTEMPTS:
                index = min(attempt - 1, len(EMPTY_RESPONSE_BACKOFF_SECONDS) - 1)
                time.sleep(EMPTY_RESPONSE_BACKOFF_SECONDS[index])

        raise EmptyLlmResponseError(
            f"SQL Generator returned an empty response after "
            f"{EMPTY_RESPONSE_MAX_ATTEMPTS} attempts "
            f"(provider={self._provider_label}, model={self._model_name})."
        )

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        return min(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SECONDS)

    # -- public API ------------------------------------------------------- #

    def generate(self, user_query: str) -> str:
        """Return exactly one validated, minimal SQLite SELECT for ``user_query``.

        The generated SQL passes both structural/safety validation
        (``validate_sql``) and intent/minimality validation
        (``SqlIntentValidator``) before it is returned, so malformed, unsafe, or
        needlessly broad SQL never reaches the executor. If the first attempt is
        rejected, the generator regenerates with corrective feedback appended to
        the prompt, up to ``max_generation_attempts`` times.

        Raises:
            ``EmptyQueryError`` for a blank query;
            ``ClarificationNeededError`` when the question names no location
            (district/firka) and is not an aggregate/ranking/"all" query, in
            which case NO LLM call is made;
            ``EmptyLlmResponseError`` when the model returns an empty/unusable
            response on every attempt (a subclass of ``LlmApiError``);
            ``LlmApiError`` on other API failures;
            ``ValidationError`` if, after all attempts, the SQL is still unsafe
            or not minimal for the question.
        """
        if not user_query or not user_query.strip():
            raise EmptyQueryError("User query is empty.")

        # Rule 13 -- under-specified location: do NOT generate SQL (no LLM call).
        # Return a clarification signal; the pipeline surfaces it to the user.
        if self._intent_validator.needs_clarification(user_query):
            raise ClarificationNeededError(
                "This groundwater data question needs a location. Please specify a "
                "district or firka (for example: 'groundwater level in Coimbatore')."
            )

        feedback: str | None = None
        last_error: Exception | None = None
        for _attempt in range(1, self._max_generation_attempts + 1):
            prompt = self._build_prompt(user_query, feedback=feedback)
            raw = self._call_llm(prompt)
            # --- TEMP DEBUG (remove after investigation): raw LLM response --- #
            logger.info("=" * 80)
            logger.info("RAW SQL LLM RESPONSE START")
            logger.info("=" * 80)
            logger.info("%r", raw)
            logger.info("=" * 80)
            logger.info("RAW SQL LLM RESPONSE END")
            logger.info("=" * 80)
            # --- END TEMP DEBUG --- #
            sql = clean_sql(raw)
            # --- TEMP DEBUG (remove after investigation): cleaned SQL --- #
            logger.info("=" * 80)
            logger.info("CLEANED SQL")
            logger.info("=" * 80)
            logger.info("%r", sql)
            logger.info("=" * 80)
            # --- END TEMP DEBUG --- #
            if not sql:
                last_error = LlmApiError("LLM response contained no SQL after cleaning.")
                feedback = "Return exactly one SQLite SELECT statement and nothing else."
                continue
            try:
                validate_sql(sql)  # structural / safety (single SELECT, safe, known tables)
                self._intent_validator.validate(sql, user_query)  # minimality vs. intent
                return sql
            except ValidationError as error:
                last_error = error
                feedback = (
                    f"The previous SQL was invalid: {error} "
                    "Generate one correct, safe SQLite SELECT."
                )
            except IntentValidationError as error:
                last_error = error
                feedback = error.feedback

        # Attempts exhausted -- surface a validation error (caught upstream and
        # recorded as a failed agent response; never reaches the executor).
        if isinstance(last_error, IntentValidationError):
            raise ValidationError(
                f"Generated SQL failed intent validation after "
                f"{self._max_generation_attempts} attempts: {last_error}",
                sql=getattr(last_error, "sql", None),
            )
        if isinstance(last_error, ValidationError):
            raise last_error
        raise LlmApiError(
            f"Failed to generate usable SQL after {self._max_generation_attempts} "
            f"attempts: {last_error}"
        )


# --------------------------------------------------------------------------- #
# Minimal smoke test (invoked only when this module is run directly)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generator = SqlGenerator()
    print(generator.generate("What is the groundwater level in Salem?"))
