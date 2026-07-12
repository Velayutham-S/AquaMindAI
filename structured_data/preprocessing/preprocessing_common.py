"""Shared helpers for the AquaMind AI structured-data preprocessing pipeline.

This module centralizes the logic common to every category preprocessing script
so that the six ``preprocess_*.py`` scripts stay thin, declarative, and free of
duplicated code (DRY). It contains no AI logic -- its sole concern is turning
raw government datasets into standardized master datasets.

Two families of raw data are supported:

* **Time-series CSV categories** (groundwater level, rainfall, river discharge,
  river water level). These share one station/location schema plus a single
  measurement column whose name varies between files. They are normalized into
  a tall/tidy schema: one value column plus a ``measurement_type`` column, so a
  new measurement cadence becomes a new row value rather than a new column.

* **GEC assessment workbooks** (district, firka). These are Excel reports with a
  three-level hierarchical (merged-cell) header. The header is flattened into a
  single row of meaningful column names, and yearly reports are unioned.

Design choices worth noting:

* Time-series masters are written **incrementally** (append per file) because a
  single category can exceed six million rows; the pipeline never holds all
  categories in memory at once.
* The only columns ever dropped are non-analytical row counters (``_id``,
  ``SlNo``, ``S.No``). Every meaningful column is preserved; where a column is
  absent for a given year it is left NULL rather than invented.
* GEC grand-total rows (state aggregates) are excluded so the table keeps a
  single grain -- one row per assessment unit -- which avoids double counting in
  downstream SQL. This exclusion is logged, never silent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger("aquamind.preprocessing")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PREPROCESSING_DIR: Path = Path(__file__).resolve().parent
STRUCTURED_DATA_DIR: Path = PREPROCESSING_DIR.parent
MASTER_DATASETS_DIR: Path = STRUCTURED_DATA_DIR / "master_datasets"

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #

CSV_READ_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1")

#: Row-counter columns that carry no analytical value and are always dropped.
NON_ANALYTICAL_COLUMNS: tuple[str, ...] = ("_id", "SlNo", "S.No")

#: Standardized station/location schema shared by every time-series CSV file.
#: Maps the raw government column name to its standardized snake_case name.
BASE_STATION_RENAME: dict[str, str] = {
    "Station": "station",
    "Agency": "agency",
    "State LGD Code": "state_lgd_code",
    "State": "state",
    "District LGD Code": "district_lgd_code",
    "District": "district",
    "Tehsil": "tehsil",
    "Block": "block",
    "Village": "village",
    "River": "river",
    "Basin": "basin",
    "Tributary": "tributary",
    "Subtributary": "subtributary",
    "SubSubtributary": "subsubtributary",
    "Local River": "local_river",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Data Acquisition Time": "observation_time",
}

#: Canonical location column order used at the front of every time-series master.
_STATION_ORDER: tuple[str, ...] = (
    "station", "agency", "state_lgd_code", "state", "district_lgd_code", "district",
    "tehsil", "block", "village", "river", "basin", "tributary", "subtributary",
    "subsubtributary", "local_river", "latitude", "longitude",
)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def configure_logging() -> None:
    """Configure console logging once for direct script execution."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )


# --------------------------------------------------------------------------- #
# Generic text / column helpers
# --------------------------------------------------------------------------- #

def to_snake_case(label: str) -> str:
    """Convert an arbitrary header label into a SQL-safe snake_case identifier.

    Parentheses are unwrapped (so units like ``(mm)`` are kept as ``mm``), ``%``
    becomes ``percent``, and every run of non-alphanumeric characters collapses
    to a single underscore.
    """
    text = str(label).replace("%", " percent ")
    text = text.replace("(", " ").replace(")", " ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def make_unique(names: list[str]) -> list[str]:
    """Return ``names`` with duplicates disambiguated by a numeric suffix."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    return unique


def strip_whitespace(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim leading/trailing whitespace from every text cell, preserving NULLs."""
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    return frame


def extract_year(observation_time: pd.Series) -> pd.Series:
    """Extract the four-digit year from a ``DD-MM-YYYY HH:MM`` timestamp column.

    The day and month are two-digit fields, so the first run of four consecutive
    digits is always the year. Rows that do not match are left NULL.
    """
    return observation_time.str.extract(r"(\d{4})", expand=False)


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Read a CSV as strings, trying supported encodings.

    Returns ``None`` if the file cannot be read with any supported encoding, so
    the caller can skip a corrupted file without aborting the whole category.
    All values are read as strings to preserve exact source representations
    (coordinate precision, LGD codes, large resource figures) and to avoid
    dtype-guessing across millions of rows.
    """
    last_error: Exception | None = None
    for encoding in CSV_READ_ENCODINGS:
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError as error:
            last_error = error
        except Exception as error:  # noqa: BLE001 - report and skip, never abort
            logger.warning("Could not read '%s': %s", path.name, error)
            return None
    logger.warning("Could not decode '%s' with supported encodings: %s", path.name, last_error)
    return None


def save_dataframe(frame: pd.DataFrame, master_filename: str) -> Path:
    """Write a completed master dataset to the master_datasets directory."""
    MASTER_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MASTER_DATASETS_DIR / master_filename
    frame.to_csv(output_path, index=False)
    return output_path


def count_csv_rows(path: Path) -> int:
    """Count data rows (excluding the header) in a CSV file cheaply."""
    for encoding in CSV_READ_ENCODINGS:
        try:
            with path.open("r", encoding=encoding) as handle:
                return max(sum(1 for _ in handle) - 1, 0)
        except UnicodeDecodeError:
            continue
    return -1


# --------------------------------------------------------------------------- #
# Time-series (tall/tidy) category pipeline
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TimeSeriesConfig:
    """Declarative configuration for one time-series CSV category."""

    category_folder: str
    master_filename: str
    value_column: str
    measurement_subject: str
    extra_rename: dict[str, str] = field(default_factory=dict)

    @property
    def raw_dir(self) -> Path:
        return STRUCTURED_DATA_DIR / self.category_folder

    @property
    def rename_map(self) -> dict[str, str]:
        return {**BASE_STATION_RENAME, **self.extra_rename}

    @property
    def canonical_columns(self) -> list[str]:
        """Fixed output column order, guaranteeing a stable master schema."""
        return [
            *_STATION_ORDER,
            *self.extra_rename.values(),
            "observation_time",
            self.value_column,
            "measurement_type",
            "year",
            "source_file",
        ]


def derive_measurement_type(measurement_column: str, subject: str) -> str:
    """Derive a ``measurement_type`` label from the raw measurement column name.

    The parenthetical unit and the category subject phrase are removed, leaving
    only the cadence/method (e.g. ``Groundwater Level Telemetry 6 Hourly
    (meter)`` -> ``Telemetry 6 Hourly``).
    """
    without_unit = re.sub(r"\(.*?\)", "", measurement_column)
    without_subject = without_unit.replace(subject, "")
    cleaned = " ".join(without_subject.split())
    return cleaned or measurement_column


def _transform_timeseries_file(frame: pd.DataFrame, config: TimeSeriesConfig, source_file: str) -> pd.DataFrame | None:
    """Standardize a single raw time-series file into the tall master schema."""
    known_columns = set(config.rename_map) | set(NON_ANALYTICAL_COLUMNS)
    measurement_columns = [column for column in frame.columns if column not in known_columns]

    if len(measurement_columns) != 1:
        logger.error(
            "Expected exactly one measurement column in '%s' but found %d (%s); skipping.",
            source_file, len(measurement_columns), measurement_columns,
        )
        return None

    measurement_column = measurement_columns[0]
    measurement_type = derive_measurement_type(measurement_column, config.measurement_subject)

    frame = frame.drop(columns=[c for c in NON_ANALYTICAL_COLUMNS if c in frame.columns])
    frame = frame.rename(columns={**config.rename_map, measurement_column: config.value_column})
    frame = strip_whitespace(frame)

    year = extract_year(frame["observation_time"]) if "observation_time" in frame else pd.NA
    frame = frame.assign(measurement_type=measurement_type, year=year, source_file=source_file)

    # Enforce the canonical schema: consistent order and NULLs for absent columns.
    return frame.reindex(columns=config.canonical_columns)


def build_timeseries_master(config: TimeSeriesConfig) -> None:
    """Build one tall master dataset for a time-series CSV category.

    Files are processed one at a time and appended to the output, so the whole
    category is never held in memory at once. Duplicate rows are removed per
    file; because ``source_file`` is part of every row, cross-file rows can
    never be exact duplicates, so per-file de-duplication is complete.
    """
    configure_logging()
    logger.info("=== Building %s from %s ===", config.master_filename, config.raw_dir)

    if not config.raw_dir.is_dir():
        logger.error("Raw directory not found: %s", config.raw_dir)
        return

    csv_files = sorted(config.raw_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in %s", config.raw_dir)
        return

    MASTER_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MASTER_DATASETS_DIR / config.master_filename
    if output_path.exists():
        output_path.unlink()  # start fresh so re-runs do not append to stale data

    total_written = 0
    total_duplicates = 0
    processed = 0
    header_written = False

    for path in csv_files:
        frame = read_csv_safe(path)
        if frame is None:
            logger.warning("Skipping unreadable file: %s", path.name)
            continue

        processed += 1
        if frame.empty:
            logger.info("File '%s' has no data rows; nothing to merge.", path.name)
            continue

        transformed = _transform_timeseries_file(frame, config, path.name)
        if transformed is None:
            continue

        before = len(transformed)
        transformed = transformed.drop_duplicates()
        duplicates = before - len(transformed)
        total_duplicates += duplicates

        transformed.to_csv(output_path, mode="a", header=not header_written, index=False)
        header_written = True
        total_written += len(transformed)
        logger.info(
            "  %-70s rows=%8d  duplicates_removed=%d", path.name, len(transformed), duplicates
        )

    logger.info(
        "Done: %d/%d files merged -> %s | total_rows=%d duplicates_removed=%d",
        processed, len(csv_files), output_path.name, total_written, total_duplicates,
    )
    _summarize_master(output_path, expected_files=len(csv_files), processed_files=processed)


# --------------------------------------------------------------------------- #
# GEC assessment workbook pipeline (district, firka)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GecConfig:
    """Declarative configuration for one GEC workbook category."""

    category_folder: str
    master_filename: str
    sheet_name: str = "GEC"
    identifier_columns: tuple[str, ...] = ()

    @property
    def raw_dir(self) -> Path:
        return STRUCTURED_DATA_DIR / self.category_folder


def _ffill_within_segments(values: list[str | None], reset_positions: set[int]) -> list[str | None]:
    """Forward-fill ``values`` left-to-right, restarting at each reset position.

    A reset position marks where a higher header level begins a new group, so a
    lower-level label never bleeds across its parent's boundary.
    """
    filled: list[str | None] = []
    current: str | None = None
    for index, value in enumerate(values):
        if index in reset_positions:
            current = None
        if value is not None:
            current = value
        filled.append(current)
    return filled


def _flatten_gec_header(header_band: pd.DataFrame) -> list[str]:
    """Flatten a multi-row GEC header band into one meaningful name per column.

    Merged-cell labels are propagated across the columns they span, but each
    level is reset at every boundary introduced by a higher level so a child
    label stays within its parent category. The surviving levels are then joined
    top-to-bottom, skipping blanks and consecutive repeats.
    """
    originals: list[list[str | None]] = [
        [
            " ".join(str(value).split()) if (pd.notna(value) and str(value).strip()) else None
            for value in row.tolist()
        ]
        for _, row in header_band.iterrows()
    ]

    column_count = header_band.shape[1]
    filled_rows: list[list[str | None]] = []
    for level, values in enumerate(originals):
        reset_positions = {
            column
            for column in range(column_count)
            if any(originals[higher][column] is not None for higher in range(level))
        }
        filled_rows.append(_ffill_within_segments(values, reset_positions))

    names: list[str] = []
    for column_index in range(column_count):
        parts: list[str] = []
        for row_values in filled_rows:
            value = row_values[column_index]
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        names.append(" | ".join(parts) if parts else f"column_{column_index}")
    return names


def _read_gec_file(path: Path, config: GecConfig) -> pd.DataFrame | None:
    """Read and standardize a single GEC workbook into a flat DataFrame."""
    try:
        raw = pd.read_excel(path, sheet_name=config.sheet_name, header=None, dtype=str)
    except Exception as error:  # noqa: BLE001 - report and skip, never abort
        logger.warning("Could not read workbook '%s': %s", path.name, error)
        return None

    first_column = raw.iloc[:, 0].astype("string")
    normalized = first_column.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

    header_positions = normalized.index[normalized == "sno"].tolist()
    if not header_positions:
        logger.error("Could not locate the 'S.No' header row in '%s'; skipping.", path.name)
        return None
    header_start = header_positions[0]

    numeric_mask = first_column.str.fullmatch(r"\s*\d+\s*").fillna(False)
    data_positions = [pos for pos in numeric_mask.index[numeric_mask] if pos > header_start]
    if not data_positions:
        logger.error("Could not locate data rows in '%s'; skipping.", path.name)
        return None
    data_start = data_positions[0]

    header_band = raw.iloc[header_start:data_start].dropna(how="all")
    columns = make_unique([to_snake_case(name) for name in _flatten_gec_header(header_band)])

    data = raw.iloc[data_start:].copy()
    data.columns = columns

    # Keep only genuine unit rows (numeric S.No); this drops the state grand-total
    # aggregate row and any blank trailing rows.
    row_counter = data["s_no"].astype("string").str.fullmatch(r"\s*\d+\s*").fillna(False)
    dropped = int((~row_counter).sum())
    data = data[row_counter]
    if dropped:
        logger.info("  %-30s dropped %d aggregate/blank row(s)", path.name, dropped)

    data = data.drop(columns=["s_no"]).copy()  # row counter, no analytical value
    data = strip_whitespace(data)
    return data.assign(assessment_year=path.stem, source_file=path.name)


def _order_gec_columns(frame: pd.DataFrame, config: GecConfig) -> pd.DataFrame:
    """Move identifiers to the front and bookkeeping columns to the end."""
    front = [c for c in (*config.identifier_columns, "assessment_year") if c in frame.columns]
    tail = ["source_file"]
    middle = [c for c in frame.columns if c not in front and c not in tail]
    return frame[[*front, *middle, *[c for c in tail if c in frame.columns]]]


def build_gec_master(config: GecConfig) -> None:
    """Build one master dataset for a GEC workbook category by unioning years."""
    configure_logging()
    logger.info("=== Building %s from %s ===", config.master_filename, config.raw_dir)

    if not config.raw_dir.is_dir():
        logger.error("Raw directory not found: %s", config.raw_dir)
        return

    workbook_files = sorted(config.raw_dir.glob("*.xlsx"))
    if not workbook_files:
        logger.error("No .xlsx files found in %s", config.raw_dir)
        return

    frames: list[pd.DataFrame] = []
    processed = 0
    for path in workbook_files:
        frame = _read_gec_file(path, config)
        if frame is None:
            continue
        processed += 1
        frames.append(frame)
        logger.info("  %-20s rows=%5d columns=%d", path.name, len(frame), frame.shape[1])

    if not frames:
        logger.error("No readable workbooks in %s; master not created.", config.raw_dir)
        return

    # Union of all meaningful columns across years; absent columns become NULL.
    merged = pd.concat(frames, ignore_index=True, sort=False)
    before = len(merged)
    merged = merged.drop_duplicates()
    duplicates = before - len(merged)
    merged = _order_gec_columns(merged, config)

    output_path = save_dataframe(merged, config.master_filename)
    logger.info(
        "Done: %d/%d workbooks merged -> %s | rows=%d columns=%d duplicates_removed=%d",
        processed, len(workbook_files), output_path.name, len(merged), merged.shape[1], duplicates,
    )
    _summarize_master(output_path, expected_files=len(workbook_files), processed_files=processed)


# --------------------------------------------------------------------------- #
# Validation summary
# --------------------------------------------------------------------------- #

def _summarize_master(output_path: Path, expected_files: int, processed_files: int) -> None:
    """Print a concise validation summary for a generated master dataset."""
    if not output_path.exists():
        logger.error("VALIDATION: master file was not created: %s", output_path)
        return

    row_count = count_csv_rows(output_path)
    try:
        head = pd.read_csv(output_path, nrows=5, dtype=str)
        readable = True
        column_count = head.shape[1]
    except Exception as error:  # noqa: BLE001
        readable = False
        column_count = -1
        logger.error("VALIDATION: master file is not readable: %s", error)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "VALIDATION %s | files %d/%d | rows=%d | columns=%d | readable=%s | size=%.2f MB",
        output_path.name, processed_files, expected_files, row_count, column_count, readable, size_mb,
    )
