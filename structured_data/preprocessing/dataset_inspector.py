"""Reusable dataset inspection utility for the AquaMind AI preprocessing pipeline.

This module is the schema-discovery and validation layer that every category
preprocessing script relies on before standardizing raw government groundwater
datasets. It is deliberately independent of any AI agent: its only concern is
understanding and validating the *shape* of the raw data.

Responsibilities
----------------
1. Inspect workbook sheets (all sheets of an Excel file, or the single logical
   sheet of a CSV file).
2. Detect the actual header row, since government Excel reports frequently place
   title/notes rows above the true column header.
3. Identify the column names for each sheet.
4. Report schema differences across the files of a single category (union,
   common, per-file missing, per-file extra, and whitespace-only variants).
5. Validate datasets (readability, emptiness, and per-file error reporting)
   without ever raising for a single bad file, so one corrupt file cannot abort
   inspection of an entire category.

The module exposes a small, typed API (`inspect_file`, `inspect_folder`,
`compare_schemas`, `render_report`) that the preprocessing scripts import, and a
`main()` entry point so a category folder can be inspected directly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --- Configuration constants (no magic numbers scattered through the logic) ---

#: Number of leading rows scanned when locating the header row of a sheet.
HEADER_SCAN_ROWS: int = 25

#: A candidate header row must contain at least this fraction of the maximum
#: non-null cell count seen in the scanned window.
HEADER_FILL_TOLERANCE: float = 0.9

#: A candidate header row must be at least this fraction textual (string cells),
#: which distinguishes a label row from a numeric data row.
MIN_HEADER_STRING_RATIO: float = 0.6

#: Encodings attempted, in order, when reading CSV files.
CSV_READ_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1")

#: Logical sheet name used to represent a CSV file (which has no sheets).
CSV_SHEET_NAME: str = "csv"

#: Recognized file extensions per family.
CSV_SUFFIXES: tuple[str, ...] = (".csv",)
EXCEL_SUFFIXES: tuple[str, ...] = (".xlsx", ".xls")


@dataclass(frozen=True)
class SheetSchema:
    """Schema of a single sheet within a dataset file."""

    sheet_name: str
    header_row_index: int
    columns: list[str]
    n_data_rows: int


@dataclass(frozen=True)
class FileInspection:
    """Result of inspecting one raw dataset file."""

    path: Path
    file_type: str  # "csv" | "excel" | "unknown"
    is_readable: bool
    is_empty: bool
    sheets: list[SheetSchema] = field(default_factory=list)
    error: str | None = None

    @property
    def primary_columns(self) -> list[str]:
        """Columns of the first sheet, or an empty list if none were read."""
        return self.sheets[0].columns if self.sheets else []


@dataclass(frozen=True)
class SchemaComparison:
    """Cross-file schema comparison for a single category folder."""

    union_columns: list[str]
    common_columns: list[str]
    per_file_missing: dict[str, list[str]]
    per_file_extra: dict[str, list[str]]
    whitespace_variants: dict[str, list[str]]


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #

def _clean_label(value: object) -> str:
    """Return a trimmed, whitespace-collapsed string form of a cell label."""
    text = "" if value is None else str(value)
    return " ".join(text.split())


def _normalization_key(label: str) -> str:
    """Return a case/whitespace-insensitive key used to match column names."""
    return _clean_label(label).lower()


def _count_non_null(row: pd.Series) -> int:
    """Count non-null cells in a row."""
    return int(row.notna().sum())


def _string_ratio(row: pd.Series) -> float:
    """Return the fraction of non-null cells in a row that are non-empty text."""
    non_null = row.dropna()
    if non_null.empty:
        return 0.0
    string_cells = sum(
        1 for value in non_null if isinstance(value, str) and value.strip()
    )
    return string_cells / len(non_null)


def _detect_header_row(raw_window: pd.DataFrame) -> int:
    """Detect the header row index within a header-less scan window.

    Strategy: the header is the earliest row that is both well populated
    (close to the maximum non-null count in the window) and predominantly
    textual. This handles government reports that prepend title/notes rows
    while still resolving to row 0 for clean CSV headers.
    """
    if raw_window.empty:
        return 0

    non_null_counts = raw_window.apply(_count_non_null, axis=1)
    max_non_null = int(non_null_counts.max())
    if max_non_null == 0:
        return 0

    fill_threshold = max_non_null * HEADER_FILL_TOLERANCE

    for position in range(len(raw_window)):
        row = raw_window.iloc[position]
        well_populated = non_null_counts.iloc[position] >= fill_threshold
        mostly_text = _string_ratio(row) >= MIN_HEADER_STRING_RATIO
        if well_populated and mostly_text:
            return position

    # Fallback: earliest sufficiently populated row, even if not clearly textual.
    for position in range(len(raw_window)):
        if non_null_counts.iloc[position] >= fill_threshold:
            return position

    return 0


def _extract_columns(raw_window: pd.DataFrame, header_row_index: int) -> list[str]:
    """Extract cleaned, de-duplicated column labels from the detected header row."""
    if raw_window.empty or header_row_index >= len(raw_window):
        return []

    header_cells = raw_window.iloc[header_row_index].tolist()
    columns: list[str] = []
    seen: dict[str, int] = {}
    for cell in header_cells:
        label = _clean_label(cell)
        if not label:
            label = "unnamed"
        if label in seen:
            seen[label] += 1
            label = f"{label}.{seen[label]}"
        else:
            seen[label] = 0
        columns.append(label)
    return columns


def classify_file(path: Path) -> str:
    """Classify a path as ``"csv"``, ``"excel"`` or ``"unknown"`` by suffix."""
    suffix = path.suffix.lower()
    if suffix in CSV_SUFFIXES:
        return "csv"
    if suffix in EXCEL_SUFFIXES:
        return "excel"
    return "unknown"


# --------------------------------------------------------------------------- #
# CSV inspection
# --------------------------------------------------------------------------- #

def _read_csv_window(path: Path) -> pd.DataFrame:
    """Read the leading scan window of a CSV, trying supported encodings."""
    last_error: Exception | None = None
    for encoding in CSV_READ_ENCODINGS:
        try:
            return pd.read_csv(
                path,
                header=None,
                nrows=HEADER_SCAN_ROWS,
                dtype=str,
                encoding=encoding,
                on_bad_lines="skip",
            )
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise last_error if last_error else RuntimeError("Unable to read CSV window")


def _count_csv_data_rows(path: Path, header_row_index: int) -> int:
    """Count data rows in a CSV cheaply by counting physical lines."""
    for encoding in CSV_READ_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, errors="strict") as handle:
                total_lines = sum(1 for _ in handle)
            data_rows = total_lines - (header_row_index + 1)
            return max(data_rows, 0)
        except UnicodeDecodeError:
            continue
    return -1


def inspect_csv(path: Path) -> FileInspection:
    """Inspect a single CSV file."""
    try:
        window = _read_csv_window(path)
    except Exception as exc:  # noqa: BLE001 - report, never abort the batch
        logger.warning("Failed to read CSV '%s': %s", path.name, exc)
        return FileInspection(path, "csv", is_readable=False, is_empty=True, error=str(exc))

    header_index = _detect_header_row(window)
    columns = _extract_columns(window, header_index)
    data_rows = _count_csv_data_rows(path, header_index)
    is_empty = data_rows == 0

    schema = SheetSchema(CSV_SHEET_NAME, header_index, columns, data_rows)
    if is_empty:
        logger.info("CSV '%s' has a header but no data rows.", path.name)
    return FileInspection(path, "csv", is_readable=True, is_empty=is_empty, sheets=[schema])


# --------------------------------------------------------------------------- #
# Excel inspection
# --------------------------------------------------------------------------- #

def _inspect_excel_sheet(workbook: pd.ExcelFile, sheet_name: str) -> SheetSchema:
    """Inspect a single sheet of an already-opened Excel workbook."""
    window = workbook.parse(sheet_name=sheet_name, header=None, nrows=HEADER_SCAN_ROWS, dtype=str)
    header_index = _detect_header_row(window)
    columns = _extract_columns(window, header_index)

    # Determine data-row count without loading numeric conversions.
    full = workbook.parse(sheet_name=sheet_name, header=None, usecols=[0])
    data_rows = max(len(full) - (header_index + 1), 0)

    return SheetSchema(sheet_name, header_index, columns, data_rows)


def inspect_excel(path: Path) -> FileInspection:
    """Inspect every sheet of a single Excel workbook."""
    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:  # noqa: BLE001 - report, never abort the batch
        logger.warning("Failed to open workbook '%s': %s", path.name, exc)
        return FileInspection(path, "excel", is_readable=False, is_empty=True, error=str(exc))

    sheets: list[SheetSchema] = []
    try:
        for sheet_name in workbook.sheet_names:
            try:
                sheets.append(_inspect_excel_sheet(workbook, sheet_name))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse sheet '%s' in '%s': %s", sheet_name, path.name, exc)
    finally:
        workbook.close()

    is_empty = all(sheet.n_data_rows == 0 for sheet in sheets) if sheets else True
    return FileInspection(path, "excel", is_readable=bool(sheets), is_empty=is_empty, sheets=sheets)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def inspect_file(path: Path) -> FileInspection:
    """Inspect a single dataset file, dispatching by file type."""
    file_type = classify_file(path)
    if file_type == "csv":
        return inspect_csv(path)
    if file_type == "excel":
        return inspect_excel(path)

    logger.warning("Unsupported file type skipped: '%s'", path.name)
    return FileInspection(path, "unknown", is_readable=False, is_empty=True, error="unsupported file type")


def inspect_folder(folder: Path) -> list[FileInspection]:
    """Inspect every supported dataset file within a category folder."""
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    files = sorted(
        entry for entry in folder.iterdir()
        if entry.is_file() and classify_file(entry) != "unknown"
    )
    if not files:
        logger.warning("No supported dataset files found in '%s'.", folder)

    return [inspect_file(path) for path in files]


def compare_schemas(inspections: list[FileInspection]) -> SchemaComparison:
    """Compare primary-sheet schemas across the files of one category."""
    readable = [item for item in inspections if item.is_readable and item.primary_columns]

    # Map each normalized key to a representative original label.
    key_to_label: dict[str, str] = {}
    per_file_keys: dict[str, set[str]] = {}
    whitespace_variants: dict[str, list[str]] = {}

    for item in readable:
        keys: set[str] = set()
        for label in item.primary_columns:
            key = _normalization_key(label)
            keys.add(key)
            existing = key_to_label.setdefault(key, label)
            if existing != label:
                variants = whitespace_variants.setdefault(key, [existing])
                if label not in variants:
                    variants.append(label)
        per_file_keys[item.path.name] = keys

    if not per_file_keys:
        return SchemaComparison([], [], {}, {}, {})

    union_keys: set[str] = set().union(*per_file_keys.values())
    common_keys: set[str] = set(union_keys)
    for keys in per_file_keys.values():
        common_keys &= keys

    def labels_for(keys: set[str]) -> list[str]:
        return sorted(key_to_label[key] for key in keys)

    per_file_missing = {
        name: labels_for(union_keys - keys)
        for name, keys in per_file_keys.items()
        if union_keys - keys
    }
    per_file_extra = {
        name: labels_for(keys - common_keys)
        for name, keys in per_file_keys.items()
        if keys - common_keys
    }

    return SchemaComparison(
        union_columns=labels_for(union_keys),
        common_columns=labels_for(common_keys),
        per_file_missing=per_file_missing,
        per_file_extra=per_file_extra,
        whitespace_variants={key_to_label[key]: variants for key, variants in whitespace_variants.items()},
    )


def render_report(folder: Path, inspections: list[FileInspection], comparison: SchemaComparison) -> str:
    """Render a human-readable inspection report for a category folder."""
    lines: list[str] = []
    lines.append(f"Dataset inspection report: {folder}")
    lines.append("=" * 72)
    lines.append(f"Files inspected: {len(inspections)}")
    lines.append("")

    for item in inspections:
        status = "OK" if item.is_readable else "UNREADABLE"
        empty = " (EMPTY)" if item.is_empty else ""
        lines.append(f"- {item.path.name} [{item.file_type}] {status}{empty}")
        if item.error:
            lines.append(f"    error: {item.error}")
        for sheet in item.sheets:
            lines.append(
                f"    sheet '{sheet.sheet_name}': header_row={sheet.header_row_index}, "
                f"columns={len(sheet.columns)}, data_rows={sheet.n_data_rows}"
            )
            lines.append(f"      columns: {sheet.columns}")
    lines.append("")

    lines.append("Schema comparison (primary sheet of each file)")
    lines.append("-" * 72)
    lines.append(f"Common columns ({len(comparison.common_columns)}): {comparison.common_columns}")
    lines.append(f"Union columns ({len(comparison.union_columns)}): {comparison.union_columns}")

    if comparison.per_file_missing:
        lines.append("")
        lines.append("Columns missing per file (relative to union):")
        for name, missing in comparison.per_file_missing.items():
            lines.append(f"  {name}: {missing}")

    if comparison.per_file_extra:
        lines.append("")
        lines.append("Columns not shared by all files (per file):")
        for name, extra in comparison.per_file_extra.items():
            lines.append(f"  {name}: {extra}")

    if comparison.whitespace_variants:
        lines.append("")
        lines.append("Whitespace/case variants of the same logical column:")
        for label, variants in comparison.whitespace_variants.items():
            lines.append(f"  {label}: {variants}")

    return "\n".join(lines)


def _configure_logging() -> None:
    """Configure module-level logging for direct execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: inspect one or more category folders and print a report."""
    parser = argparse.ArgumentParser(description="Inspect AquaMind AI raw dataset folders.")
    parser.add_argument("folders", nargs="+", help="One or more category folders to inspect.")
    args = parser.parse_args(argv)

    _configure_logging()

    for raw_folder in args.folders:
        folder = Path(raw_folder)
        try:
            inspections = inspect_folder(folder)
        except NotADirectoryError as exc:
            logger.error("%s", exc)
            continue
        comparison = compare_schemas(inspections)
        print(render_report(folder, inspections, comparison))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
