# Data Agent — SQLite Database Layer

This directory contains the SQLite database layer used exclusively by the
AquaMind AI **Data Agent**. Its single responsibility is to build, populate,
validate, and maintain `groundwater.db` from the preprocessed master datasets.

It performs **no** SQL generation, answers **no** user questions, and contains
**no** LLM logic. Those concerns belong to the Data Agent, not to this layer.

## Contents

| File | Purpose |
|------|---------|
| `database_builder.py` | Builds and validates `groundwater.db` from the master datasets. |
| `groundwater.db` | The generated SQLite database (build artifact, not source). |
| `README.md` | This document. |

## Source data

The builder reads only the six master datasets produced by the preprocessing
stage (it never touches the raw datasets):

```
structured_data/master_datasets/
    master_district.csv
    master_firka.csv
    master_groundwater_level.csv
    master_rainfall.csv
    master_river_discharge.csv
    master_river_water_level.csv
```

## How to rebuild `groundwater.db`

From the project root:

```bash
python agents/data_agent/database/database_builder.py
```

The build is deterministic and idempotent: each run drops and recreates the six
tables, so it can be run any time the master datasets change. `groundwater.db`
is a generated artifact and is safe to delete and regenerate.

## Tables

One table per master dataset — no splitting, no duplication.

| Table | Rows | Data columns | Description |
|-------|-----:|-------------:|-------------|
| `district` | 189 | 154 | GEC district-level assessment (union across assessment years). |
| `firka` | 7,127 | 163 | GEC firka-level assessment (union across assessment years). |
| `groundwater_level` | 6,281,586 | 23 | Groundwater level observations (tall schema). |
| `rainfall` | 196,442 | 22 | Rainfall observations (tall schema). |
| `river_discharge` | 20 | 22 | River discharge observations (tall schema). |
| `river_water_level` | 78,394 | 25 | River water level observations (tall schema). |

Every table also has a surrogate `id INTEGER PRIMARY KEY AUTOINCREMENT` column
in addition to the data columns above.

## Column types

Types are **inferred** from the data rather than forced to `TEXT`:

- A column is `INTEGER` if every non-empty value is an integer,
- `REAL` if every value is numeric,
- `TEXT` otherwise.

Because SQLite uses type affinity, any value that does not match the inferred
type is still stored losslessly, so inference is a safe best-effort declaration.
Coordinates, measurement values, and resource figures are stored as `REAL`;
`year` and LGD codes as `INTEGER`; names, timestamps, and `measurement_type` as
`TEXT`.

## Primary keys

These government datasets have no column (or small combination) that is
reliably unique and non-null across every year, so each table uses a surrogate
`id INTEGER PRIMARY KEY AUTOINCREMENT`. The raw row counters (`_id`, `SlNo`)
were dropped during preprocessing and are never used as keys.

## Indexes

Only indexes with real query benefit are created. Each is chosen for a concrete
Data Agent access pattern:

**GEC tables (`district`, `firka`)**
- `(district, assessment_year)` — per-unit, per-year lookups; the leading
  `district` column also serves district-only filters.
- `(assessment_year)` — retrieve all units for a given assessment year.
- `(firka)` *(firka table only)* — look up a firka by name across years.

**Time-series tables (`groundwater_level`, `rainfall`, `river_discharge`, `river_water_level`)**
- `(station)` — station-level access (all readings for a monitoring station).
- `(district, year)` — the dominant geographic + temporal filter; the leading
  `district` column also serves district-only filters.
- `(year)` — state-wide temporal grouping and multi-year trend analysis.

**Deliberately not indexed** (documented to justify the absence):
- `state` — single value (all data is Tamil Nadu), so no selectivity.
- `measurement_type` — only 2–3 distinct values; low selectivity.
- `observation_time` — free-text `DD-MM-YYYY HH:MM` timestamp, not
  chronologically sortable as text; `year` is indexed for temporal queries.
- `assessment_unit`, `village` — sparse / mostly empty in these datasets.

## Validation

After importing each table the builder verifies and prints a summary for:

- **Row count** — database rows equal the master CSV rows and the number of
  rows imported.
- **Column count / names** — all data columns preserved, in order.
- **Table existence** — the table was created.
- **Index existence** — every planned index was created.
- **NULL preservation** — the total number of NULL cells in the database equals
  the number of empty cells in the master CSV (missing values remain NULL and
  are never turned into empty strings or zeros).

A table is reported as `PASS` only when all checks succeed, and the run ends
with a `BUILD SUMMARY` line reporting how many of the six tables built
successfully.

## Error handling

The builder logs (rather than crashes) on missing master datasets, corrupted or
empty CSVs, failed imports, and SQLite errors. A failure in one table does not
abort the others; the final summary reports per-table success.

## Notes

- `groundwater.db` is roughly 1.7 GB (data plus indexes). Like the master
  datasets, it is a generated artifact and should not be committed to version
  control — regenerate it from source with the command above.
- The database is intended to be used read-only by the Data Agent.
