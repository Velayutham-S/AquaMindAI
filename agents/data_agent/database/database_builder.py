"""SQLite database layer for the AquaMind AI Data Agent.

This module is responsible for exactly one thing: building and validating the
``groundwater.db`` SQLite database from the six preprocessed master datasets.
It performs no SQL generation, answers no user questions, and contains no LLM
logic -- it is pure database management.

Running this module (``python database_builder.py``) performs a full,
deterministic rebuild:

1. Drop and recreate each of the six tables (idempotent rebuild).
2. Infer an appropriate SQLite column type per column from the master data.
3. Import every row, preserving all columns, values, and NULLs.
4. Create only the indexes that provide real query benefit.
5. Validate row counts, column counts, table/index existence and NULL
   preservation, printing a summary to the terminal.

Design notes
------------
* **Primary keys.** These government datasets have no column (or small column
  set) that is reliably unique and non-null across every year, so each table
  uses a surrogate ``id INTEGER PRIMARY KEY AUTOINCREMENT``. The raw row
  counters (``_id``/``SlNo``) were already dropped during preprocessing and are
  never used as keys.
* **Types.** Types are inferred from the data (INTEGER / REAL / TEXT). Because
  SQLite uses type affinity, a value that does not fit the inferred type is
  still stored losslessly, so inference is a safe best-effort declaration.
* **NULLs.** Master CSVs represent missing data as empty fields; these are read
  as NaN and stored as SQL NULL, never as empty strings or zeros.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger("aquamind.database")

# --------------------------------------------------------------------------- #
# Paths and configuration
# --------------------------------------------------------------------------- #

DATABASE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = DATABASE_DIR.parents[2]  # database -> data_agent -> agents -> root
MASTER_DIR: Path = PROJECT_ROOT / "structured_data" / "master_datasets"
DB_PATH: Path = DATABASE_DIR / "groundwater.db"

#: Each table maps directly to exactly one master dataset.
TABLES: dict[str, str] = {
    "district": "master_district.csv",
    "firka": "master_firka.csv",
    "groundwater_level": "master_groundwater_level.csv",
    "rainfall": "master_rainfall.csv",
    "river_discharge": "master_river_discharge.csv",
    "river_water_level": "master_river_water_level.csv",
}

#: Surrogate primary key added to every table.
PRIMARY_KEY = "id"

#: Rows read per chunk during import (keeps the largest table off the heap).
CHUNK_SIZE = 200_000

#: Rows sampled when inferring column types (affinity makes this safe).
TYPE_SAMPLE_ROWS = 200_000

_INT_PATTERN = r"^-?\d+$"
_REAL_PATTERN = r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def configure_logging() -> None:
    """Configure console logging once for direct execution."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #

def infer_column_types(csv_path: Path) -> dict[str, str]:
    """Infer a SQLite type (INTEGER / REAL / TEXT) for each column.

    A column is INTEGER if every non-empty sampled value is an integer, REAL if
    every value is numeric, and TEXT otherwise. Empty-only columns default to
    TEXT (they hold only NULLs, so affinity is irrelevant).
    """
    sample = pd.read_csv(csv_path, dtype=str, nrows=TYPE_SAMPLE_ROWS, na_filter=False)
    types: dict[str, str] = {}
    for column in sample.columns:
        values = sample[column].str.strip()
        values = values[values != ""]
        if values.empty:
            types[column] = "TEXT"
        elif values.str.match(_INT_PATTERN).all():
            types[column] = "INTEGER"
        elif values.str.match(_REAL_PATTERN).all():
            types[column] = "REAL"
        else:
            types[column] = "TEXT"
    return types


# --------------------------------------------------------------------------- #
# Index planning
# --------------------------------------------------------------------------- #

def plan_indexes(table: str, columns: list[str]) -> list[tuple[str, list[str], str]]:
    """Return the beneficial indexes for a table as (name, columns, rationale).

    Deliberately skipped: ``state`` (single value -- all data is Tamil Nadu, so
    no selectivity), ``measurement_type`` (only 2-3 distinct values),
    ``observation_time`` (free-text timestamp, not chronologically sortable),
    and sparse identifier columns (``assessment_unit``, ``village``).
    """
    available = set(columns)
    plans: list[tuple[str, list[str], str]] = []

    def add(index_columns: list[str], rationale: str) -> None:
        if all(column in available for column in index_columns):
            name = f"idx_{table}_" + "_".join(index_columns)
            plans.append((name, index_columns, rationale))

    if table in ("district", "firka"):
        add(["district", "assessment_year"],
            "Per-unit, per-year lookups; leading 'district' also serves district-only filters.")
        add(["assessment_year"],
            "Retrieve all assessment units for a given assessment year.")
        if table == "firka":
            add(["firka"], "Look up a firka by name across years.")
    else:  # time-series tables
        add(["station"], "Station-level access (all readings for a monitoring station).")
        add(["district", "year"],
            "Dominant geographic + temporal filter; leading 'district' also serves district-only filters.")
        add(["year"], "State-wide temporal grouping and multi-year trend analysis.")

    return plans


# --------------------------------------------------------------------------- #
# Table creation and import
# --------------------------------------------------------------------------- #

def create_table(connection: sqlite3.Connection, table: str, column_types: dict[str, str]) -> None:
    """Drop and recreate a table with a surrogate primary key and typed columns."""
    column_defs = ", ".join(f'"{column}" {sql_type}' for column, sql_type in column_types.items())
    connection.execute(f'DROP TABLE IF EXISTS "{table}"')
    connection.execute(
        f'CREATE TABLE "{table}" ("{PRIMARY_KEY}" INTEGER PRIMARY KEY AUTOINCREMENT, {column_defs})'
    )


def import_csv(connection: sqlite3.Connection, table: str, csv_path: Path) -> tuple[int, int]:
    """Import a master CSV into an existing table in chunks.

    Returns ``(rows_inserted, null_cells)``. Empty fields become NaN and are
    stored as SQL NULL; other values are stored under the column's type affinity.
    """
    rows_inserted = 0
    null_cells = 0
    for chunk in pd.read_csv(csv_path, dtype=str, chunksize=CHUNK_SIZE):
        null_cells += int(chunk.isna().sum().sum())
        rows_inserted += len(chunk)
        chunk.to_sql(table, connection, if_exists="append", index=False)
    return rows_inserted, null_cells


def create_indexes(connection: sqlite3.Connection, table: str, plans: list[tuple[str, list[str], str]]) -> None:
    """Create the planned indexes, logging the rationale for each."""
    for name, index_columns, rationale in plans:
        column_list = ", ".join(f'"{column}"' for column in index_columns)
        connection.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}" ({column_list})')
        logger.info("  index %-42s (%s) -- %s", name, ", ".join(index_columns), rationale)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def count_csv_rows(csv_path: Path) -> int:
    """Count data rows (excluding the header) in a CSV file cheaply."""
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def count_db_nulls(connection: sqlite3.Connection, table: str, columns: list[str]) -> int:
    """Count total NULL cells across all data columns of a table."""
    terms = " + ".join(f'SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END)' for column in columns)
    result = connection.execute(f'SELECT {terms} FROM "{table}"').fetchone()[0]
    return int(result or 0)


def validate_table(
    connection: sqlite3.Connection,
    table: str,
    csv_path: Path,
    data_columns: list[str],
    rows_inserted: int,
    csv_null_cells: int,
    expected_indexes: list[str],
) -> bool:
    """Validate one imported table and log the outcome. Returns True if all checks pass."""
    db_rows = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    csv_rows = count_csv_rows(csv_path)

    table_info = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    db_data_columns = [row[1] for row in table_info if row[1] != PRIMARY_KEY]

    db_indexes = {
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,)
        ).fetchall()
        if row[0].startswith("idx_")
    }
    db_nulls = count_db_nulls(connection, table, db_data_columns)

    checks = {
        "rows match CSV": db_rows == csv_rows == rows_inserted,
        "columns preserved": db_data_columns == data_columns,
        "indexes created": set(expected_indexes).issubset(db_indexes),
        "NULLs preserved": db_nulls == csv_null_cells,
    }
    passed = all(checks.values())

    logger.info(
        "VALIDATION %-20s rows=%d (csv=%d) | data_columns=%d | indexes=%d | nulls db=%d csv=%d | %s",
        table, db_rows, csv_rows, len(db_data_columns), len(db_indexes),
        db_nulls, csv_null_cells, "PASS" if passed else "FAIL",
    )
    for label, ok in checks.items():
        if not ok:
            logger.error("  FAILED CHECK [%s] on table '%s'", label, table)
    return passed


# --------------------------------------------------------------------------- #
# Build orchestration
# --------------------------------------------------------------------------- #

def build_table(connection: sqlite3.Connection, table: str, filename: str) -> bool:
    """Build, import, index and validate a single table. Returns success."""
    csv_path = MASTER_DIR / filename
    if not csv_path.exists():
        logger.error("Master dataset missing for '%s': %s", table, csv_path)
        return False

    try:
        logger.info("=== Building table '%s' from %s ===", table, filename)
        column_types = infer_column_types(csv_path)
        data_columns = list(column_types.keys())

        create_table(connection, table, column_types)
        rows_inserted, csv_null_cells = import_csv(connection, table, csv_path)

        plans = plan_indexes(table, data_columns)
        create_indexes(connection, table, plans)
        connection.commit()

        logger.info("  imported rows=%d, columns=%d", rows_inserted, len(data_columns))
        return validate_table(
            connection, table, csv_path, data_columns,
            rows_inserted, csv_null_cells, [name for name, _, _ in plans],
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        logger.error("Corrupted or empty CSV for '%s': %s", table, error)
    except sqlite3.Error as error:
        logger.error("SQLite error while building '%s': %s", table, error)
    except Exception as error:  # noqa: BLE001 - report and continue with other tables
        logger.error("Unexpected error while building '%s': %s", table, error)
    return False


def build_database() -> None:
    """Rebuild the entire groundwater database from the master datasets."""
    configure_logging()
    logger.info("Building groundwater database at %s", DB_PATH)

    if not MASTER_DIR.is_dir():
        logger.error("Master datasets directory not found: %s", MASTER_DIR)
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    results: dict[str, bool] = {}
    try:
        # Bulk-load performance pragmas (safe: the DB is rebuilt from source and
        # used read-only by the Data Agent).
        connection.execute("PRAGMA journal_mode = MEMORY")
        connection.execute("PRAGMA synchronous = OFF")
        for table, filename in TABLES.items():
            results[table] = build_table(connection, table, filename)
    finally:
        connection.close()

    succeeded = sum(results.values())
    logger.info("=" * 78)
    logger.info("BUILD SUMMARY: %d/%d tables built successfully", succeeded, len(TABLES))
    for table, ok in results.items():
        logger.info("  %-20s %s", table, "OK" if ok else "FAILED")
    if succeeded != len(TABLES):
        logger.error("One or more tables failed to build; see errors above.")


if __name__ == "__main__":
    build_database()
