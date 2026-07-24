"""SQLite Executor for the AquaMind AI Data Agent.

Single responsibility: execute an already-validated SQL query against
``groundwater.db`` and return the raw result rows.

This component does NOT generate or modify SQL, call the LLM, answer questions,
or format responses. It only runs SQL and returns rows.

The database is opened in **read-only** mode, which is both the correct access
level for the Data Agent and a hard safety guarantee: any accidental write
statement fails at the SQLite layer.

Public interface:
    SQLiteExecutor().execute(sql_query) -> list[sqlite3.Row]
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("aquamind.sqlite_executor")

DATABASE_DIR: Path = Path(__file__).resolve().parent
DB_PATH: Path = DATABASE_DIR / "groundwater.db"

#: Diagnostic threshold only. When a query returns more rows than this, the
#: executor logs a WARNING so oversized result sets are visible in the logs.
#: Results are NEVER truncated and the SQL is NEVER modified -- producing a
#: minimal result set is the SQL Generator's responsibility.
LARGE_RESULT_WARNING_THRESHOLD: int = 50

#: Hard safety cap. A query that returns MORE than this many rows is rejected
#: (not truncated) to protect the downstream LLM context window. The Data Agent
#: raises rather than passing an oversized result to the Response Generator.
#: Kept deliberately small: a minimal, well-scoped answer is far below this, and
#: even 200 rows is large for an LLM prompt.
MAX_RESULT_ROWS: int = 200


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class SQLiteExecutorError(Exception):
    """Base error for the SQLite Executor."""


class DatabaseNotFoundError(SQLiteExecutorError):
    """The SQLite database file does not exist."""


class SQLExecutionError(SQLiteExecutorError):
    """The SQL query failed to execute."""


class RowLimitExceededError(SQLiteExecutorError):
    """The query returned more rows than the Data Agent will pass downstream.

    Signals that the generated SQL was too broad. Results are NOT truncated and
    NOT forwarded; the Data Agent fails this query instead of overflowing the
    LLM context window.
    """


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #

class SQLiteExecutor:
    """Executes validated SQL against the groundwater database (read-only)."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path

    def execute(self, sql_query: str) -> list[sqlite3.Row]:
        """Execute ``sql_query`` and return the result rows.

        Rows are returned as ``sqlite3.Row`` objects, preserving column names.
        An empty result set returns an empty list (not an error).

        Raises:
            DatabaseNotFoundError: if the database file is missing.
            SQLExecutionError: if the query fails to execute.
        """
        if not self._db_path.exists():
            raise DatabaseNotFoundError(f"Database not found: {self._db_path}")

        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(sql_query)
            # Fetch at most one row beyond the cap: this detects an oversized
            # result WITHOUT materializing a huge set, and never truncates what
            # is actually returned (a legitimate <= cap result is returned whole).
            rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
        except sqlite3.Error as error:
            raise SQLExecutionError(f"SQL execution failed: {error}") from error
        finally:
            if connection is not None:
                connection.close()

        if len(rows) > MAX_RESULT_ROWS:
            logger.warning(
                "Query returned more than %d rows; rejecting as too broad "
                "(results NOT truncated and NOT sent downstream).",
                MAX_RESULT_ROWS,
            )
            raise RowLimitExceededError(
                f"Generated SQL was too broad: it returns more than {MAX_RESULT_ROWS} "
                "rows. The Data Agent will not forward an oversized result. Regenerate "
                "with mandatory filters (district/firka/year), an aggregate "
                "(AVG/SUM/MIN/MAX/COUNT), or the latest record (LIMIT 1)."
            )

        row_count = len(rows)
        logger.info("Executed query; %d row(s) returned.", row_count)
        if row_count > LARGE_RESULT_WARNING_THRESHOLD:
            logger.warning(
                "Query returned %d rows (> %d). Results are NOT truncated; the "
                "SQL Generator should return only the minimum rows required.",
                row_count, LARGE_RESULT_WARNING_THRESHOLD,
            )
        return rows
