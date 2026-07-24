"""Intent-aware SQL validation for the AquaMind AI Data Agent.

Single responsibility: given a generated SQLite ``SELECT`` and the ORIGINAL user
question, verify the SQL returns only the MINIMAL data required to answer that
question. This layer runs AFTER SQL generation and BEFORE execution, entirely
inside the Data Agent. It never executes SQL, never calls the LLM, and never
rewrites SQL -- it only accepts or rejects, so the SQL Generator can regenerate.

It complements (does not replace) the structural/safety checks in
``sql_generator.validate_sql`` (single statement, SELECT-only, no destructive
keywords, documented tables). Here the focus is *minimality vs. intent*:

    * mandatory filters the user stated (district / firka / year) are present,
    * "latest / current" resolves to a single latest row,
    * "top N" carries a matching LIMIT,
    * ``SELECT *`` is rejected,
    * an unfiltered, non-aggregated query on a huge table is rejected.

The module is self-contained: it defines its own ``IntentValidationError`` and
imports nothing from ``sql_generator`` (no import cycle).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger("aquamind.sql_intent_validator")

# --------------------------------------------------------------------------- #
# Paths / configuration
# --------------------------------------------------------------------------- #

_LLM_DIR: Path = Path(__file__).resolve().parent
#: Read-only source for the district gazetteer (Data Agent's own database).
_DB_PATH: Path = _LLM_DIR.parent / "database" / "groundwater.db"

#: Per-observation tables where an unfiltered, non-aggregated query is unsafe
#: (millions / hundreds of thousands of rows).
_LARGE_TIMESERIES_TABLES: frozenset[str] = frozenset(
    {"groundwater_level", "rainfall", "river_water_level", "river_discharge"}
)

# --------------------------------------------------------------------------- #
# Intent-recognition patterns (applied to the natural-language user question)
# --------------------------------------------------------------------------- #

_LATEST_WORDS = re.compile(
    r"(?i)\b(latest|current|present|recent|newest|most\s+recent|as\s+of\s+now|"
    r"right\s+now|today|now)\b"
)
_TOPN_RE = re.compile(r"(?i)\b(?:top|bottom|highest|lowest|largest|smallest)\s+(\d+)\b")
_RANK_WORDS = re.compile(
    r"(?i)\b(top|bottom|highest|lowest|largest|smallest|most|least|maximum|minimum|"
    r"max|min|rank|ranking)\b"
)
_AGG_WORDS = re.compile(
    r"(?i)\b(average|avg|mean|total|sum|count|how\s+many|number\s+of|maximum|minimum|"
    r"max|min)\b"
)
_ALL_WORDS = re.compile(
    r"(?i)\b(all|every|entire|complete\s+dataset|list\s+all|export)\b"
)
#: Location-type hints other than district/firka that still scope a query
#: (so a station/village query is NOT treated as under-specified).
_LOCATION_HINT = re.compile(
    r"(?i)\b(station|well|piezometer|village|block|taluk|tehsil|river|basin|"
    r"watershed|tributary)\b"
)
_FIRKA_WORD = re.compile(r"(?i)\bfirka\b")
_YEAR_RE = re.compile(r"\b(?:19\d{2}|20\d{2})\b")

# --------------------------------------------------------------------------- #
# Patterns applied to the generated SQL
# --------------------------------------------------------------------------- #

_SQL_AGG_RE = re.compile(r"(?i)\b(count|sum|avg|min|max)\s*\(|\bgroup\s+by\b")
_LIMIT_RE = re.compile(r"(?i)\blimit\s+(\d+)\b")
_SELECT_STAR_RE = re.compile(r"(?i)\bselect\s+\*\s+from\b")
_WHERE_RE = re.compile(r"(?i)\bwhere\b")
_DISTRICT_RE = re.compile(r"(?i)\bdistrict\b")
_FROM_JOIN_RE = re.compile(r"(?i)\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class IntentValidationError(Exception):
    """The generated SQL does not minimally satisfy the user's question.

    ``feedback`` is a short corrective instruction appended to the prompt on
    regeneration; ``sql`` is the rejected SQL (for diagnostics).
    """

    def __init__(self, message: str, feedback: str | None = None, sql: str | None = None) -> None:
        super().__init__(message)
        self.feedback = feedback or message
        self.sql = sql


# --------------------------------------------------------------------------- #
# Gazetteer
# --------------------------------------------------------------------------- #

def load_district_gazetteer(db_path: Path = _DB_PATH) -> frozenset[str]:
    """Return the lower-cased set of district names, read-only and best-effort.

    Sourced from the small ``district`` GEC table (~189 rows). If the database
    is unavailable, an empty set is returned and district detection is simply
    skipped (the broad-query guard still protects against oversized results).
    """
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT DISTINCT district FROM district WHERE district IS NOT NULL"
            ).fetchall()
        finally:
            connection.close()
        return frozenset(
            str(value[0]).strip().lower() for value in rows if str(value[0]).strip()
        )
    except sqlite3.Error as error:
        logger.warning("Could not load district gazetteer (%s); skipping district checks.", error)
        return frozenset()


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #

class SqlIntentValidator:
    """Validates that generated SQL minimally satisfies the user's question."""

    def __init__(self, district_gazetteer: frozenset[str] | None = None) -> None:
        self._districts = (
            district_gazetteer if district_gazetteer is not None else load_district_gazetteer()
        )

    # -- intent helpers ---------------------------------------------------- #

    def mentioned_districts(self, user_query: str) -> list[str]:
        """District names from the gazetteer that appear as whole words in the query."""
        text = (user_query or "").lower()
        found: list[str] = []
        for name in self._districts:
            if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", text):
                found.append(name)
        return found

    def needs_clarification(self, user_query: str) -> bool:
        """True when the query is under-specified for the Data Agent (Rule 13).

        A question is under-specified when it names no district and no firka, and
        is not an aggregate, a ranking, an explicit "all" request, or scoped by
        another location hint (station/village/block/river/...). Such a question
        cannot be answered with a minimal, filtered query, so the Data Agent
        should ask for a location instead of generating a broad query.
        """
        query = user_query or ""
        if self.mentioned_districts(query):
            return False
        if _FIRKA_WORD.search(query):
            return False
        if _LOCATION_HINT.search(query):
            return False
        if _RANK_WORDS.search(query) or _AGG_WORDS.search(query) or _ALL_WORDS.search(query):
            return False
        return True

    # -- SQL validation ---------------------------------------------------- #

    def validate(self, sql: str, user_query: str) -> None:
        """Raise :class:`IntentValidationError` if ``sql`` is not minimal for the query.

        Returns ``None`` when the SQL is acceptable. Never rewrites SQL.
        """
        text = (sql or "").strip()
        query = user_query or ""

        has_aggregate = bool(_SQL_AGG_RE.search(text))
        has_where = bool(_WHERE_RE.search(text))
        limit_match = _LIMIT_RE.search(text)
        years_in_query = _YEAR_RE.findall(query)

        # Validation 6 -- reject SELECT * (COUNT(*) etc. are unaffected).
        if _SELECT_STAR_RE.search(text):
            raise IntentValidationError(
                "SQL uses SELECT *.",
                feedback="Do not use SELECT *. Select only the specific columns needed to answer the question.",
                sql=text,
            )

        # Validation 3 -- every year the user stated must be filtered in the SQL.
        for year in years_in_query:
            if year not in text:
                raise IntentValidationError(
                    f"User specified year {year} but the SQL does not filter by it.",
                    feedback=(
                        f"The user asked about the year {year}. Add the matching year filter "
                        f"(time-series: year = {year}; GEC tables: the matching assessment_year) "
                        f"and do not return other years."
                    ),
                    sql=text,
                )

        # Validation 1 -- a named district must be filtered in the SQL.
        districts = self.mentioned_districts(query)
        if districts and not _DISTRICT_RE.search(text):
            raise IntentValidationError(
                f"User named district(s) {districts} but the SQL has no district filter.",
                feedback=(
                    "The user named a district. Add a WHERE filter on the district column "
                    "(case-insensitive), e.g. WHERE district = '<name>' COLLATE NOCASE."
                ),
                sql=text,
            )

        # Validation 2 -- a firka question must reference the firka column/table.
        if _FIRKA_WORD.search(query) and not _FIRKA_WORD.search(text):
            raise IntentValidationError(
                "User mentioned a firka but the SQL does not reference the firka column/table.",
                feedback="The user asked about a firka. Query the firka table and filter WHERE firka = '<name>'.",
                sql=text,
            )

        # Validation 4 -- "latest / current" must resolve to a single latest row
        # (unless the answer is an aggregate, which already returns one row).
        if _LATEST_WORDS.search(query) and not years_in_query and not has_aggregate:
            if not (limit_match and int(limit_match.group(1)) == 1):
                raise IntentValidationError(
                    "User asked for the latest/current value but the SQL is not limited to one latest row.",
                    feedback=(
                        "Return only the most recent record: order by the latest date "
                        "(year DESC, then a reformatted observation_time DESC) and use LIMIT 1."
                    ),
                    sql=text,
                )

        # Validation 5 -- an explicit "top/bottom N" must carry a matching LIMIT N.
        topn = _TOPN_RE.search(query)
        if topn:
            requested = topn.group(1)
            if not limit_match:
                raise IntentValidationError(
                    f"User asked for {requested} rows but the SQL has no LIMIT.",
                    feedback=f"Add LIMIT {requested} and ORDER BY the ranked metric.",
                    sql=text,
                )
            if limit_match.group(1) != requested:
                raise IntentValidationError(
                    f"User asked for {requested} rows but the SQL LIMIT is {limit_match.group(1)}.",
                    feedback=f"Use LIMIT {requested} to match the number of rows the user requested.",
                    sql=text,
                )

        # Validation 7 -- broad-query guard: an unfiltered, non-aggregated,
        # unlimited query on a huge time-series table would return far too many
        # rows (usually because a mandatory filter was dropped). Reject it.
        tables = {match.group(1).lower() for match in _FROM_JOIN_RE.finditer(text)}
        if tables & _LARGE_TIMESERIES_TABLES and not has_where and not has_aggregate and not limit_match:
            raise IntentValidationError(
                "SQL is unfiltered and would return an excessive number of rows.",
                feedback=(
                    "This query is too broad. Add the mandatory filters (district/firka and/or "
                    "year), aggregate (AVG/SUM/MIN/MAX/COUNT), or restrict to the latest record "
                    "with LIMIT 1 so only the required rows are returned."
                ),
                sql=text,
            )
