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


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class SQLiteExecutorError(Exception):
    """Base error for the SQLite Executor."""


class DatabaseNotFoundError(SQLiteExecutorError):
    """The SQLite database file does not exist."""


class SQLExecutionError(SQLiteExecutorError):
    """The SQL query failed to execute."""


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
            rows = connection.execute(sql_query).fetchall()
            logger.info("Executed query; %d row(s) returned.", len(rows))
            return rows
        except sqlite3.Error as error:
            raise SQLExecutionError(f"SQL execution failed: {error}") from error
        finally:
            if connection is not None:
                connection.close()
