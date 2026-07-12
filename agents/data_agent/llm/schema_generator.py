"""Database schema documentation generator for the AquaMind AI Data Agent.

This is a **build-time** tool. It opens the SQLite database once, reads all of
its schema metadata (tables, columns, types, primary keys, indexes, row counts)
plus a bounded set of observed low-cardinality values, and writes a single
authoritative Markdown document:

    agents/data_agent/llm/llm_inputs/database_schema.md

That document is the schema reference consumed by the SQL Generation LLM (along
with the system prompt and the user's query). The LLM therefore never inspects
``groundwater.db`` at runtime -- schema knowledge is generated once, here.

Running ``python schema_generator.py`` regenerates and overwrites the document,
then validates that every table, column, primary key, index, and row count has
been documented.

Nothing in the schema is invented. Structure comes directly from the database;
human-readable descriptions are inferred only from table names, column names,
indexes, and observed values.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("aquamind.schema")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

LLM_DIR: Path = Path(__file__).resolve().parent
DATA_AGENT_DIR: Path = LLM_DIR.parent
DB_PATH: Path = DATA_AGENT_DIR / "database" / "groundwater.db"
OUTPUT_PATH: Path = LLM_DIR / "llm_inputs" / "database_schema.md"

PRIMARY_KEY = "id"

# --------------------------------------------------------------------------- #
# Grounded description knowledge (token -> meaning).
# Every key below is a literal token that appears in the database's table or
# column names, so these are descriptions of existing elements, not inventions.
# --------------------------------------------------------------------------- #

TABLE_PURPOSES: dict[str, str] = {
    "district": "District-level GEC groundwater resource assessment for Tamil Nadu. One row per district per assessment year.",
    "firka": "Firka-level (revenue sub-division) GEC groundwater resource assessment. One row per firka per assessment year.",
    "groundwater_level": "Groundwater level observations (depth to water, in metres) recorded at monitoring stations over time.",
    "rainfall": "Rainfall observations (in millimetres) recorded at monitoring stations over time.",
    "river_discharge": "River discharge / flow observations (in cubic metres per second) recorded at gauging stations over time.",
    "river_water_level": "River water level / stage observations (in metres) recorded at gauging stations over time.",
}

GEC_TABLES = ("district", "firka")
TIME_SERIES_TABLES = ("groundwater_level", "rainfall", "river_discharge", "river_water_level")

#: Fixed value column per time-series table (each column exists in the schema).
VALUE_COLUMNS: dict[str, str] = {
    "groundwater_level": "groundwater_level_m",
    "rainfall": "rainfall_mm",
    "river_discharge": "river_discharge_m3s",
    "river_water_level": "river_water_level_m",
}

#: Location/identifier columns shared across the time-series tables.
TIME_SERIES_LOCATION = (
    "station", "agency", "district", "tehsil", "block", "village",
    "river", "basin", "local_river", "latitude", "longitude",
)

#: Unit tokens that appear as column-name suffixes, with their expansions.
UNIT_GLOSSARY: dict[str, str] = {
    "ham": "hectare-metres (a volume: 1 ham = 10,000 cubic metres)",
    "ha": "hectares (an area)",
    "ha_m": "hectare-metres (a volume of ground water extraction)",
    "mm": "millimetres (rainfall depth)",
    "m3s": "cubic metres per second (river discharge / flow rate)",
    "m": "metres (water level / depth)",
    "percent": "percent (%)",
}

#: Category tokens that appear as column-name suffixes in the GEC tables.
CATEGORY_GLOSSARY: dict[str, str] = {
    "c": "Command area (canal-irrigated area)",
    "nc": "Non-Command area (area outside canal command)",
    "pq": "Poor Quality area (area with poor groundwater quality)",
    "total": "total across all sub-categories",
    "fresh": "fresh (non-saline) groundwater zone",
    "saline": "saline groundwater zone",
}

#: Broader domain terms, keyed by tokens present in table/column names.
DOMAIN_TERMS: dict[str, str] = {
    "GEC": "Groundwater Estimation Committee methodology used for resource assessment.",
    "firka": "A firka is a revenue sub-division (a group of villages) below the district level.",
    "assessment_unit": "The unit being assessed (a district in the district table; a firka in the firka table).",
    "recharge": "Water that replenishes the aquifer (from rainfall, canals, tanks, etc.).",
    "extraction": "Groundwater withdrawn/drafted for domestic, industrial, or irrigation use.",
    "stage_of_ground_water_extraction": "Extraction as a percentage of the annual extractable resource; the key stress indicator.",
    "annual_extractable_ground_water_resource": "The volume of groundwater that can be sustainably extracted per year.",
    "environmental_flows": "Groundwater volume reserved for ecological/base-flow needs.",
    "categorization_of_assessment_unit": "The stress category assigned to the unit (e.g. safe, semi-critical, critical, over-exploited, saline).",
    "quality_tagging": "Flags indicating groundwater quality parameters present in the unit.",
    "rl_msl": "Reduced Level above Mean Sea Level (ground elevation reference).",
    "observation_time": "Timestamp of the reading, stored as text in DD-MM-YYYY HH:MM format.",
    "measurement_type": "How the reading was collected (manual vs telemetry, and its cadence).",
}

#: User vocabulary -> schema targets (grounded in real columns/tables).
USER_VOCABULARY: list[tuple[str, str]] = [
    ("water level, groundwater level, water table, depth to water", "groundwater_level.groundwater_level_m"),
    ("rainfall, rain, precipitation", "rainfall.rainfall_mm"),
    ("river discharge, river flow, flow rate", "river_discharge.river_discharge_m3s"),
    ("river level, river water level, gauge, stage", "river_water_level.river_water_level_m"),
    ("recharge", "district/firka: ground_water_recharge_ham_* columns"),
    ("extraction, draft, withdrawal, pumping", "district/firka: ground_water_extraction_for_all_uses_ha_m_* columns"),
    ("stage of extraction, exploitation, stress", "district/firka: stage_of_ground_water_extraction_percent_* columns"),
    ("over-exploited, critical, semi-critical, safe, saline", "firka: categorization_of_assessment_unit_* columns"),
    ("assessment, resource assessment, GEC", "district or firka tables"),
    ("station, well, observation well, piezometer", "time-series tables: station column"),
    ("year, annual, trend, over time", "time-series: year column; GEC: assessment_year column"),
    ("district", "district column (all tables) / district table"),
    ("firka", "firka column / firka table"),
]

#: Columns to sample distinct values for (value semantics). Base names only.
LOW_CARDINALITY_COLUMNS = ("measurement_type", "agency", "is_discharge_data_available")
GEC_CATEGORICAL_PREFIXES = (
    "categorization_of_assessment_unit_", "pre_monsoon_of_gw_trend_",
    "post_monsoon_of_gw_trend_", "quality_tagging_",
)
MAX_DISTINCT = 25


# --------------------------------------------------------------------------- #
# Schema reflection (metadata read directly from the database)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    not_null: bool
    primary_key: bool


@dataclass(frozen=True)
class IndexInfo:
    name: str
    columns: list[str]
    unique: bool


@dataclass(frozen=True)
class TableSchema:
    name: str
    row_count: int
    columns: list[ColumnInfo]
    indexes: list[IndexInfo]
    create_sql: str

    @property
    def data_columns(self) -> list[ColumnInfo]:
        return [c for c in self.columns if c.name != PRIMARY_KEY]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Build it with database_builder.py first.")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def list_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def describe_table(connection: sqlite3.Connection, table: str) -> TableSchema:
    info = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    columns = [ColumnInfo(r[1], r[2], bool(r[3]), bool(r[5])) for r in info]

    indexes: list[IndexInfo] = []
    for row in connection.execute(f'PRAGMA index_list("{table}")').fetchall():
        index_name, unique = row[1], bool(row[2])
        if index_name.startswith("sqlite_autoindex"):
            continue
        cols = [ir[2] for ir in connection.execute(f'PRAGMA index_info("{index_name}")').fetchall()]
        indexes.append(IndexInfo(index_name, cols, unique))

    row_count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    sql_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    create_sql = (sql_row[0] if sql_row and sql_row[0] else "").strip()
    return TableSchema(table, row_count, columns, indexes, create_sql)


def distinct_values(connection: sqlite3.Connection, table: str, column: str) -> list[str]:
    rows = connection.execute(
        f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL '
        f'ORDER BY "{column}" LIMIT ?', (MAX_DISTINCT + 1,)
    ).fetchall()
    return [str(r[0]) for r in rows]


def year_range(connection: sqlite3.Connection, table: str) -> tuple[str, str] | None:
    row = connection.execute(f'SELECT MIN(year), MAX(year) FROM "{table}"').fetchone()
    if row and row[0] is not None and row[1] is not None:
        return str(row[0]), str(row[1])
    return None


def sample_value_semantics(connection: sqlite3.Connection, schema: TableSchema) -> dict[str, list[str]]:
    """Return observed distinct values for the table's low-cardinality columns."""
    semantics: dict[str, list[str]] = {}
    for column in schema.column_names:
        is_low_card = column in LOW_CARDINALITY_COLUMNS
        is_gec_cat = schema.name in GEC_TABLES and column.startswith(GEC_CATEGORICAL_PREFIXES)
        if not (is_low_card or is_gec_cat):
            continue
        values = distinct_values(connection, schema.name, column)
        if values and len(values) <= MAX_DISTINCT:
            semantics[column] = values
    return semantics


# --------------------------------------------------------------------------- #
# Description inference (from names only)
# --------------------------------------------------------------------------- #

def describe_column(table: str, column: ColumnInfo) -> str:
    """Infer a short human description of a column from its name."""
    name = column.name
    fixed = {
        "id": "Surrogate primary key (auto-increment row id).",
        "station": "Monitoring/gauging station name.",
        "agency": "Agency that recorded the observation.",
        "state": "State name (all rows are Tamil Nadu).",
        "state_lgd_code": "State LGD (Local Government Directory) code.",
        "district": "District name.",
        "district_lgd_code": "District LGD code.",
        "tehsil": "Tehsil / taluk name.",
        "block": "Block name.",
        "village": "Village name.",
        "river": "River name.",
        "basin": "River basin name.",
        "tributary": "Tributary name.",
        "subtributary": "Sub-tributary name.",
        "subsubtributary": "Sub-sub-tributary name.",
        "local_river": "Local river name.",
        "latitude": "Latitude (decimal degrees).",
        "longitude": "Longitude (decimal degrees).",
        "observation_time": "Reading timestamp, text in DD-MM-YYYY HH:MM format.",
        "measurement_type": "Collection method and cadence (see Column Value Semantics).",
        "year": "Year of the observation (derived from observation_time).",
        "source_file": "Original master dataset file the row came from (provenance).",
        "firka": "Firka (revenue sub-division) name.",
        "watershed_district": "Watershed / district grouping for the firka.",
        "assessment_unit": "Name of the assessed unit (often blank in the district table).",
        "assessment_year": "GEC assessment year, e.g. '2024-2025'.",
        "rl_of_zero_gauge": "Reduced level of the zero gauge point.",
        "mean_sea_level": "Mean sea level reference for the gauge.",
        "is_discharge_data_available": "Whether discharge data is available at this gauge.",
        "rl_msl": "Reduced Level above Mean Sea Level (ground elevation).",
    }
    if name in fixed:
        return fixed[name]
    if name in VALUE_COLUMNS.values():
        return "The measured value for this table (see Aggregation Columns)."

    # GEC hierarchical metric column: <category path> ... <unit> <sub-category>.
    tokens = name.split("_")
    category = CATEGORY_GLOSSARY.get(tokens[-1], "")
    unit = next((UNIT_GLOSSARY[u] for u in ("ham", "ha_m", "ha", "percent", "mm") if f"_{u}_" in f"_{name}_"), "")
    readable = name.replace("_", " ")
    description = f"GEC metric: {readable}."
    if unit:
        description += f" Unit: {unit}."
    if category:
        description += f" Sub-category: {category}."
    return description


def present_units(all_columns: set[str]) -> dict[str, str]:
    return {token: meaning for token, meaning in UNIT_GLOSSARY.items()
            if any(f"_{token}_" in f"_{c}_" or c.endswith(f"_{token}") for c in all_columns)}


# --------------------------------------------------------------------------- #
# Markdown section builders
# --------------------------------------------------------------------------- #

def _bar(char: str = "-") -> str:
    return char * 3


def section_overview(schemas: list[TableSchema], connection: sqlite3.Connection) -> list[str]:
    total_rows = sum(s.row_count for s in schemas)
    lines = [
        "## Database Overview",
        "",
        f"- **Database file:** `{DB_PATH.name}` (SQLite).",
        f"- **Tables:** {len(schemas)}.",
        f"- **Total rows:** {total_rows:,}.",
        "- **Geographic scope:** Tamil Nadu, India (the `state` column is single-valued).",
        "- **Two data families:** GEC resource-assessment tables (`district`, `firka`) and "
        "time-series observation tables (`groundwater_level`, `rainfall`, `river_discharge`, `river_water_level`).",
        "",
        "| Table | Rows | Data columns | Grain |",
        "|-------|-----:|-------------:|-------|",
    ]
    for schema in schemas:
        if schema.name in GEC_TABLES:
            grain = f"one row per {schema.name} per assessment year"
        else:
            yr = year_range(connection, schema.name)
            span = f", years {yr[0]}-{yr[1]}" if yr else ""
            grain = f"one row per observation{span}"
        lines.append(f"| `{schema.name}` | {schema.row_count:,} | {len(schema.data_columns)} | {grain} |")
    lines.append("")
    return lines


def section_table_documentation(schemas: list[TableSchema]) -> list[str]:
    lines = ["## Table Documentation", "",
             "Every column of every table, with its SQLite type and constraints.", ""]
    for schema in schemas:
        lines.append(f"### `{schema.name}` ({schema.row_count:,} rows)")
        lines.append("")
        lines.append(f"_{TABLE_PURPOSES.get(schema.name, '')}_")
        lines.append("")
        lines.append("| Column | Type | Constraints | Description |")
        lines.append("|--------|------|-------------|-------------|")
        for column in schema.columns:
            constraints = []
            if column.primary_key:
                constraints.append("PRIMARY KEY")
            if column.not_null:
                constraints.append("NOT NULL")
            lines.append(
                f"| `{column.name}` | {column.type or 'TEXT'} | {', '.join(constraints) or '-'} "
                f"| {describe_column(schema.name, column)} |"
            )
        lines.append("")
    return lines


def section_table_purpose(schemas: list[TableSchema]) -> list[str]:
    lines = ["## Table Purpose", ""]
    for schema in schemas:
        lines.append(f"- **`{schema.name}`** — {TABLE_PURPOSES.get(schema.name, '')}")
    lines.append("")
    return lines


def section_query_routing() -> list[str]:
    return [
        "## Query Routing Guide",
        "",
        "Pick the table from the user's intent:",
        "",
        "- Groundwater **level / depth to water / water table** → `groundwater_level`.",
        "- **Rainfall / precipitation** → `rainfall`.",
        "- **River discharge / flow** → `river_discharge`.",
        "- **River water level / stage / gauge** → `river_water_level`.",
        "- **District-level assessment** (recharge, extraction, stage of extraction, availability, "
        "resource categorization at district scale) → `district`.",
        "- **Firka-level assessment** or **over-exploited / critical / safe categorization**, "
        "or any firka-specific question → `firka`.",
        "",
        "Rules of thumb:",
        "- 'How much water is there / recharge / extraction / how stressed' → GEC tables (`district`, `firka`).",
        "- 'What was the reading / value over time / at a station / by year' → time-series tables.",
        "- Firka questions use `firka`; district questions use `district`. They are not joined.",
        "",
    ]


def section_relationships(schemas: list[TableSchema]) -> list[str]:
    by_name = {s.name: set(s.column_names) for s in schemas}
    shared = set.intersection(*by_name.values()) - {PRIMARY_KEY, "source_file"}
    lines = [
        "## Database Relationships",
        "",
        "There are **no foreign keys**; the tables are independent and are not designed to be JOINed. "
        "They relate only through shared descriptive columns (logical, value-based links):",
        "",
        f"- Columns common to every table: {', '.join(f'`{c}`' for c in sorted(shared)) or '(none)'}.",
        "- `district` is present in all tables and is the primary way to align data geographically "
        "(e.g. compare a district's rainfall with its assessment).",
        "- Time-series tables share the full station/location block "
        "(`station`, `district`, `block`, `village`, `river`, `basin`, `latitude`, `longitude`, ...).",
        "- `district` and `firka` share the GEC metric columns but at different spatial grains.",
        "",
        "Prefer single-table queries. If a cross-table comparison is unavoidable, align on `district` "
        "(and `year`/`assessment_year`) rather than assuming row-level correspondence.",
        "",
    ]
    return lines


def section_indexes(schemas: list[TableSchema]) -> list[str]:
    lines = [
        "## Index Documentation",
        "",
        "Indexes present in the database (use these columns in `WHERE`/`ORDER BY` for fast queries):",
        "",
    ]
    for schema in schemas:
        lines.append(f"**`{schema.name}`**")
        if not schema.indexes:
            lines.append("- (no user indexes)")
        for index in schema.indexes:
            lines.append(f"- `{index.name}` on ({', '.join(f'`{c}`' for c in index.columns)})")
        lines.append("")
    lines += [
        "Deliberately **not** indexed (do not assume fast filtering on these): `state` (single value), "
        "`measurement_type` (2-3 values), `observation_time` (free text). Use `year` for temporal filters.",
        "",
    ]
    return lines


def section_sql_notes() -> list[str]:
    return [
        "## SQL Generation Notes",
        "",
        "- Dialect is **SQLite**. Use SQLite-compatible syntax only.",
        "- Generate a single **read-only `SELECT`**. Never write, update, delete, or alter.",
        "- Always quote identifiers with double quotes if unsure; column names are snake_case.",
        "- `year` is an INTEGER column on time-series tables; filter with `year = 2023`.",
        "- `assessment_year` is TEXT like `'2024-2025'` on GEC tables; filter with the full string.",
        "- `observation_time` is TEXT in `DD-MM-YYYY HH:MM` format and is **not** chronologically "
        "sortable as text; use `year` for time filtering/grouping and trends.",
        "- Match names case-insensitively and loosely: `WHERE district LIKE '%salem%'` "
        "(names are stored in mixed/upper case; GEC tables store district in upper case, e.g. `SALEM`).",
        "- Time-series tables mix measurement cadences in one column; when a specific cadence is meant, "
        "filter `measurement_type` (see Column Value Semantics).",
        "- Missing data is stored as `NULL`; guard aggregates (e.g. `AVG(col)` ignores NULLs; use "
        "`WHERE col IS NOT NULL` when counting).",
        "- Some GEC metric columns are stored with TEXT affinity because the source contained "
        "non-numeric cells; wrap them with `CAST(col AS REAL)` before aggregating if needed.",
        "- All data is Tamil Nadu; a `state` filter is unnecessary.",
        "- Add `LIMIT` for browse-style questions; use aggregates for 'how much/average/total' questions.",
        "",
    ]


def section_domain_terminology(all_columns: set[str]) -> list[str]:
    lines = ["## Domain Terminology", ""]
    for term, meaning in DOMAIN_TERMS.items():
        lines.append(f"- **{term}** — {meaning}")
    lines.append("")
    lines.append("**Category suffixes** on GEC metric columns:")
    for token, meaning in CATEGORY_GLOSSARY.items():
        lines.append(f"- `_{token}` — {meaning}")
    lines.append("")
    units = present_units(all_columns)
    if units:
        lines.append("**Units** encoded in column names:")
        for token, meaning in units.items():
            lines.append(f"- `{token}` — {meaning}")
        lines.append("")
    return lines


def section_example_questions() -> list[str]:
    return [
        "## Example User Questions",
        "",
        "- \"What is the average groundwater level in Salem in 2023?\" → `groundwater_level`.",
        "- \"Total rainfall recorded in Chennai district in 2022.\" → `rainfall`.",
        "- \"Which firkas are over-exploited in 2024-2025?\" → `firka` (categorization column).",
        "- \"Stage of groundwater extraction for Coimbatore district.\" → `district`.",
        "- \"List monitoring stations in Madurai.\" → `groundwater_level` (distinct station).",
        "- \"Annual extractable groundwater resource for each district in 2023-2024.\" → `district`.",
        "- \"Trend of groundwater level over the years for a station.\" → `groundwater_level` grouped by `year`.",
        "- \"River discharge readings for a river.\" → `river_discharge`.",
        "",
    ]


def section_column_usage() -> list[str]:
    return [
        "## Column Usage",
        "",
        "- **Filter columns** (WHERE): `district`, `firka`, `station`, `block`, `village`, `year`, "
        "`assessment_year`, `measurement_type`.",
        "- **Grouping columns** (GROUP BY): `district`, `firka`, `station`, `year`, `assessment_year`, "
        "`measurement_type`.",
        "- **Aggregation targets** (AVG/SUM/MIN/MAX): the value column of each time-series table and the "
        "numeric GEC metric columns (see Aggregation Columns).",
        "- **Ordering** (ORDER BY): `year`, or a value column; avoid ordering by `observation_time` (text).",
        "- **Provenance / non-analytical:** `id`, `source_file` (do not aggregate or expose unless asked).",
        "",
    ]


def section_table_selection() -> list[str]:
    return [
        "## Table Selection Guide",
        "",
        "| If the question mentions... | Use table |",
        "|-----------------------------|-----------|",
        "| water level, water table, depth to water | `groundwater_level` |",
        "| rainfall, rain, precipitation | `rainfall` |",
        "| discharge, flow rate | `river_discharge` |",
        "| river level, stage, gauge | `river_water_level` |",
        "| recharge, extraction, availability, stage of extraction (district) | `district` |",
        "| firka, over-exploited/critical/safe, firka-level assessment | `firka` |",
        "",
    ]


def section_important_columns(schemas: list[TableSchema]) -> list[str]:
    lines = ["## Important Columns", "",
             "The most useful columns per table for answering questions:", ""]
    for schema in schemas:
        cols = set(schema.column_names)
        if schema.name in TIME_SERIES_TABLES:
            key = [c for c in ("station", "district", "block", "village", "year",
                               "measurement_type", VALUE_COLUMNS[schema.name]) if c in cols]
        else:
            identifiers = [c for c in ("district", "firka", "assessment_year") if c in cols]
            headline = [c for c in schema.column_names if c.endswith("_total") and (
                c.startswith("ground_water_recharge_ham")
                or c.startswith("annual_extractable_ground_water_resource_ham")
                or c.startswith("ground_water_extraction_for_all_uses_ha_m")
                or c.startswith("stage_of_ground_water_extraction_percent")
                or c.startswith("total_ground_water_availability_in_the_area_ham"))]
            categorization = [c for c in schema.column_names if c.startswith("categorization_of_assessment_unit_")]
            key = identifiers + headline + categorization
        lines.append(f"- **`{schema.name}`**: {', '.join(f'`{c}`' for c in key)}")
    lines.append("")
    return lines


def section_aggregation_columns(schemas: list[TableSchema]) -> list[str]:
    lines = ["## Aggregation Columns", "",
             "Numeric columns suitable for AVG / SUM / MIN / MAX:", ""]
    for schema in schemas:
        if schema.name in TIME_SERIES_TABLES:
            targets = [VALUE_COLUMNS[schema.name]]
            lines.append(f"- **`{schema.name}`**: `{targets[0]}` (the measured value).")
        else:
            numeric = [c.name for c in schema.data_columns
                       if c.type in ("REAL", "INTEGER") and c.name.endswith("_total")]
            preview = ", ".join(f"`{c}`" for c in numeric[:8])
            more = f" ... (+{len(numeric) - 8} more `_total` metrics)" if len(numeric) > 8 else ""
            lines.append(f"- **`{schema.name}`**: the numeric GEC metric columns, e.g. {preview}{more}. "
                         "Prefer the `_total` columns for headline figures; `_c`/`_nc`/`_pq` break totals "
                         "into Command / Non-Command / Poor-Quality areas.")
    lines.append("")
    return lines


def section_search_keywords(schemas: list[TableSchema]) -> list[str]:
    lines = ["## Search Keywords", "",
             "Text columns to match user-provided place/entity names (use `LIKE`):", ""]
    text_identifier_candidates = (
        "station", "district", "firka", "block", "village", "tehsil",
        "river", "basin", "local_river", "watershed_district", "assessment_unit",
    )
    for schema in schemas:
        cols = set(schema.column_names)
        present = [c for c in text_identifier_candidates if c in cols]
        lines.append(f"- **`{schema.name}`**: {', '.join(f'`{c}`' for c in present)}")
    lines.append("")
    return lines


def section_common_patterns() -> list[str]:
    return [
        "## Common SQL Patterns",
        "",
        "```sql",
        "-- Average value for a district in a year (time-series)",
        "SELECT AVG(groundwater_level_m) FROM groundwater_level",
        "WHERE district LIKE '%salem%' AND year = 2023;",
        "",
        "-- Yearly trend for a station",
        "SELECT year, AVG(groundwater_level_m) AS avg_level",
        "FROM groundwater_level WHERE station LIKE '%<name>%'",
        "GROUP BY year ORDER BY year;",
        "",
        "-- Filter a specific measurement cadence",
        "SELECT * FROM rainfall",
        "WHERE district LIKE '%chennai%' AND measurement_type = 'Manual Daily';",
        "",
        "-- District assessment headline figure for a year (GEC)",
        "SELECT district, stage_of_ground_water_extraction_percent_total",
        "FROM district WHERE assessment_year = '2024-2025' ORDER BY district;",
        "",
        "-- Firkas in a stress category",
        "SELECT firka, district FROM firka",
        "WHERE assessment_year = '2024-2025'",
        "  AND categorization_of_assessment_unit_total = 'over_exploited';",
        "```",
        "",
    ]


def section_value_semantics(value_semantics: dict[str, dict[str, list[str]]]) -> list[str]:
    lines = ["## Column Value Semantics", "",
             "Observed distinct values for low-cardinality columns (use these exact strings in filters):", ""]
    for table, columns in value_semantics.items():
        if not columns:
            continue
        lines.append(f"**`{table}`**")
        for column, values in columns.items():
            shown = ", ".join(f"`{v}`" for v in values)
            lines.append(f"- `{column}`: {shown}")
        lines.append("")
    lines += [
        "Notes:",
        "- `measurement_type` distinguishes manual vs telemetry readings and their cadence.",
        "- GEC categorization values (e.g. `safe`, `semi_critical`, `critical`, `over_exploited`, "
        "`saline`) classify each unit's groundwater stress.",
        "- Trend columns record `Rising` / `Falling` / `Neither Rising Nor Falling`.",
        "",
    ]
    return lines


def section_user_vocabulary() -> list[str]:
    lines = ["## Column Aliases / User Vocabulary", "",
             "Everyday phrasing mapped to schema targets:", "",
             "| User says | Maps to |", "|-----------|---------|"]
    for phrase, target in USER_VOCABULARY:
        lines.append(f"| {phrase} | {target} |")
    lines.append("")
    return lines


def section_query_examples_per_table(schemas: list[TableSchema]) -> list[str]:
    lines = ["## Query Examples Per Table", ""]
    examples: dict[str, list[str]] = {
        "groundwater_level": [
            "SELECT AVG(groundwater_level_m) FROM groundwater_level WHERE district LIKE '%salem%' AND year = 2023;",
            "SELECT year, MIN(groundwater_level_m), MAX(groundwater_level_m) FROM groundwater_level GROUP BY year ORDER BY year;",
        ],
        "rainfall": [
            "SELECT SUM(rainfall_mm) FROM rainfall WHERE district LIKE '%chennai%' AND year = 2022;",
            "SELECT district, AVG(rainfall_mm) FROM rainfall WHERE year = 2023 GROUP BY district;",
        ],
        "river_discharge": [
            "SELECT station, AVG(river_discharge_m3s) FROM river_discharge GROUP BY station;",
            "SELECT year, AVG(river_discharge_m3s) FROM river_discharge GROUP BY year ORDER BY year;",
        ],
        "river_water_level": [
            "SELECT AVG(river_water_level_m) FROM river_water_level WHERE river LIKE '%<river>%' AND year = 2023;",
            "SELECT station, MAX(river_water_level_m) FROM river_water_level GROUP BY station;",
        ],
        "district": [
            "SELECT district, annual_extractable_ground_water_resource_ham_total FROM district WHERE assessment_year = '2024-2025';",
            "SELECT district, stage_of_ground_water_extraction_percent_total FROM district WHERE assessment_year = '2024-2025' ORDER BY stage_of_ground_water_extraction_percent_total DESC;",
        ],
        "firka": [
            "SELECT firka, district FROM firka WHERE assessment_year = '2024-2025' AND categorization_of_assessment_unit_total = 'over_exploited';",
            "SELECT district, COUNT(*) FROM firka WHERE assessment_year = '2024-2025' GROUP BY district;",
        ],
    }
    for schema in schemas:
        lines.append(f"### `{schema.name}`")
        lines.append("")
        lines.append("```sql")
        for query in examples.get(schema.name, []):
            lines.append(query)
        lines.append("```")
        lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# LLM-oriented guidance sections (added to improve SQL generation accuracy)
# --------------------------------------------------------------------------- #

#: Columns that make sense as multi-table alignment keys, in display order.
_JOIN_KEY_ORDER = ("district", "station", "year", "assessment_year")


def section_canonical_table_selection(schemas: list[TableSchema]) -> list[str]:
    names = {s.name for s in schemas}
    columns = {s.name: set(s.column_names) for s in schemas}
    lines = [
        "## Canonical Table Selection Rules",
        "",
        "Choose the table directly from the user's intent (rules use the discovered table names):",
        "",
    ]

    def rule(intent: str, tables: list[str]) -> None:
        present = [t for t in tables if t in names]
        if not present:
            return
        lines.append(f"- If the user asks about **{intent}**:")
        for table in present:
            lines.append(f"  - → `{table}`")

    recharge_tables = [s.name for s in schemas if any("recharge" in c for c in columns[s.name])]
    extraction_tables = [s.name for s in schemas if any("extraction" in c for c in columns[s.name])]

    rule("groundwater monitoring observations (level / depth to water)", ["groundwater_level"])
    rule("groundwater recharge", recharge_tables)
    rule("groundwater extraction / draft / stage of extraction", extraction_tables)
    rule("rainfall", ["rainfall"])
    rule("river discharge / flow", ["river_discharge"])
    rule("river water level / stage", ["river_water_level"])
    rule("district groundwater assessment", ["district"])
    rule("firka groundwater assessment", ["firka"])
    lines.append("")
    return lines


def section_cross_table_rules(schemas: list[TableSchema]) -> list[str]:
    columns = {s.name: set(s.column_names) for s in schemas}
    names = [s.name for s in schemas]
    lines = [
        "## Cross Table Query Rules",
        "",
        "There are **no foreign keys** in this database. The pairs below can only be aligned "
        "*logically* on shared descriptive columns; there is no guaranteed row-level correspondence. "
        "Mind differing grains (time-series `year` is INTEGER; GEC `assessment_year` is a TEXT range "
        "such as `'2024-2025'`, so they are not directly equal).",
        "",
        "| Table A | Table B | Align on |",
        "|---------|---------|----------|",
    ]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            table_a, table_b = names[i], names[j]
            shared = [k for k in _JOIN_KEY_ORDER if k in columns[table_a] and k in columns[table_b]]
            if shared:
                keys = ", ".join(f"`{k}`" for k in shared)
                lines.append(f"| `{table_a}` | `{table_b}` | {keys} |")
    lines.append("")
    return lines


def section_preferred_filter_columns(schemas: list[TableSchema]) -> list[str]:
    lines = ["## Preferred Filter Columns", "",
             "Columns most commonly suitable for `WHERE` clauses, per table:", ""]
    for schema in schemas:
        cols = set(schema.column_names)
        if schema.name in TIME_SERIES_TABLES:
            candidates = ("district", "station", "block", "village", "year", "measurement_type")
        else:
            candidates = ("district", "firka", "assessment_year")
        preferred = [c for c in candidates if c in cols]
        lines.append(f"- **`{schema.name}`**: " + ", ".join(f"`{c}`" for c in preferred))
    lines.append("")
    return lines


def section_preferred_sorting_columns(schemas: list[TableSchema]) -> list[str]:
    lines = ["## Preferred Sorting Columns", "",
             "Columns commonly suitable for `ORDER BY`, per table:", ""]
    for schema in schemas:
        cols = set(schema.column_names)
        if schema.name in TIME_SERIES_TABLES:
            ordering = [VALUE_COLUMNS[schema.name]] + [c for c in ("year", "observation_time") if c in cols]
        else:
            ordering = [c for c in ("assessment_year", "district", "firka") if c in cols]
            if "stage_of_ground_water_extraction_percent_total" in cols:
                ordering.append("stage_of_ground_water_extraction_percent_total")
        lines.append(f"- **`{schema.name}`**: " + ", ".join(f"`{c}`" for c in ordering))
    lines.append("")
    return lines


def section_synonym_dictionary(schemas: list[TableSchema]) -> list[str]:
    all_columns = {c for s in schemas for c in s.column_names}

    def has_col(name: str) -> bool:
        return name in all_columns

    def has_prefix(prefix: str) -> bool:
        return any(c.startswith(prefix) for c in all_columns)

    candidates = [
        (["water table", "water level", "groundwater level", "groundwater depth", "depth to water"],
         "groundwater_level.groundwater_level_m", has_col("groundwater_level_m")),
        (["rainfall", "rain", "precipitation"],
         "rainfall.rainfall_mm", has_col("rainfall_mm")),
        (["water discharge", "river discharge", "flow", "flow rate"],
         "river_discharge.river_discharge_m3s", has_col("river_discharge_m3s")),
        (["river level", "river water level", "stage", "gauge level"],
         "river_water_level.river_water_level_m", has_col("river_water_level_m")),
        (["groundwater extraction", "draft", "withdrawal", "pumping"],
         "district/firka: ground_water_extraction_for_all_uses_ha_m_* columns",
         has_prefix("ground_water_extraction")),
        (["availability", "available groundwater", "extractable resource"],
         "district/firka: annual_extractable_ground_water_resource_ham_* columns",
         has_prefix("annual_extractable_ground_water_resource")),
        (["recharge"],
         "district/firka: ground_water_recharge_ham_* columns",
         has_prefix("ground_water_recharge")),
        (["stage of extraction", "exploitation", "stress level"],
         "district/firka: stage_of_ground_water_extraction_percent_* columns",
         has_prefix("stage_of_ground_water_extraction")),
        (["over-exploited", "critical", "semi-critical", "safe", "category"],
         "firka: categorization_of_assessment_unit_* columns",
         has_prefix("categorization_of_assessment_unit")),
    ]
    lines = ["## Synonym Dictionary", "",
             "Natural-language terms mapped to schema targets (only mappings inferable from the schema):", "",
             "| User term(s) | Maps to |", "|--------------|---------|"]
    for synonyms, target, present in candidates:
        if present:
            lines.append(f"| {', '.join(synonyms)} | `{target}` |")
    lines.append("")
    return lines


def section_forbidden_sql() -> list[str]:
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "PRAGMA", "VACUUM"]
    lines = [
        "## Forbidden SQL",
        "",
        "The SQL Generation LLM must produce **only read-only SQLite `SELECT` statements**. "
        "It must never generate any of the following keywords, which modify data or schema or change "
        "engine state:",
        "",
    ]
    lines += [f"- `{keyword}`" for keyword in forbidden]
    lines += [
        "",
        "Additional rules:",
        "- Emit a **single** statement only (no semicolon-separated multiple statements).",
        "- No transactions (`BEGIN`/`COMMIT`/`ROLLBACK`) and no SQL comments used to smuggle commands.",
        "- A single `SELECT` (optionally with CTEs and subqueries) is the only permitted output.",
        "",
    ]
    return lines


def section_case_sensitivity(connection: sqlite3.Connection, schemas: list[TableSchema]) -> list[str]:
    lines = [
        "## Case Sensitivity Notes",
        "",
        "SQLite `=` comparisons on TEXT are **case-sensitive** by default. Place/entity names are stored "
        "with the casing observed below:",
        "",
    ]
    for schema in schemas:
        if "district" not in schema.column_names:
            continue
        samples = distinct_values(connection, schema.name, "district")[:3]
        if not samples:
            continue
        casing = "UPPERCASE" if all(v == v.upper() for v in samples) else "Mixed / Title case"
        examples = ", ".join(f"`{v}`" for v in samples)
        lines.append(f"- **`{schema.name}`.district** — {casing}; e.g. {examples}")
    lines += [
        "",
        "Guidance:",
        "- Match the exact casing shown above, or make matching case-insensitive using "
        "`COLLATE NOCASE` (e.g. `WHERE district = 'salem' COLLATE NOCASE`), `UPPER(col) = UPPER('...')`, "
        "or `WHERE district LIKE '%salem%'`.",
        "- GEC tables (`district`, `firka`) and the time-series tables may use different casing for the "
        "same place, so prefer case-insensitive matching when the target table is uncertain.",
        "",
    ]
    return lines


def section_frequently_used_columns(schemas: list[TableSchema]) -> list[str]:
    lines = ["## Frequently Used Columns", "",
             "Columns expected to be queried most frequently, per table:", ""]
    for schema in schemas:
        cols = set(schema.column_names)
        if schema.name in TIME_SERIES_TABLES:
            frequent = [c for c in ("district", "station", "year", VALUE_COLUMNS[schema.name],
                                    "measurement_type") if c in cols]
        else:
            frequent = [c for c in ("district", "firka", "assessment_year") if c in cols]
            for metric in ("stage_of_ground_water_extraction_percent_total",
                           "annual_extractable_ground_water_resource_ham_total",
                           "categorization_of_assessment_unit_total"):
                if metric in cols:
                    frequent.append(metric)
        lines.append(f"- **`{schema.name}`**: " + ", ".join(f"`{c}`" for c in frequent))
    lines.append("")
    return lines


def section_time_columns(schemas: list[TableSchema]) -> list[str]:
    lines = ["## Time Columns", "", "Temporal columns per table:", ""]
    temporal_names = {"year", "observation_time", "assessment_year"}
    for schema in schemas:
        temporal = [c for c in schema.column_names
                    if c in temporal_names or c.endswith("_time") or c.endswith("_year")]
        value = ", ".join(f"`{c}`" for c in temporal) if temporal else "(none)"
        lines.append(f"- **`{schema.name}`**: {value}")
    lines.append("")
    return lines


def section_example_sql_queries(connection: sqlite3.Connection, schemas: list[TableSchema]) -> list[str]:
    lines = ["## Example SQL Queries", "",
             "Documentation only (do not execute). SQLite-compatible `SELECT` examples per table, "
             "using real values discovered in the database:", ""]
    for schema in schemas:
        cols = set(schema.column_names)
        district_val = None
        if "district" in cols:
            sampled = distinct_values(connection, schema.name, "district")[:1]
            district_val = sampled[0] if sampled else None

        lines.append(f"### `{schema.name}`")
        lines.append("")
        lines.append("```sql")
        if schema.name in TIME_SERIES_TABLES:
            value = VALUE_COLUMNS[schema.name]
            if district_val is not None:
                lines.append(f"SELECT * FROM {schema.name} WHERE district = '{district_val}' LIMIT 10;")
                lines.append(f"SELECT AVG({value}) FROM {schema.name} WHERE district = '{district_val}';")
            lines.append(f"SELECT year, AVG({value}) FROM {schema.name} GROUP BY year ORDER BY year;")
        else:
            years = distinct_values(connection, schema.name, "assessment_year")
            year_val = years[-1] if years else "2024-2025"
            lines.append(f"SELECT * FROM {schema.name} WHERE assessment_year = '{year_val}' LIMIT 10;")
            if district_val is not None:
                lines.append(f"SELECT * FROM {schema.name} WHERE district = '{district_val}';")
            if "stage_of_ground_water_extraction_percent_total" in cols:
                lines.append(
                    f"SELECT district, stage_of_ground_water_extraction_percent_total FROM {schema.name} "
                    f"WHERE assessment_year = '{year_val}' "
                    "ORDER BY stage_of_ground_water_extraction_percent_total DESC LIMIT 10;"
                )
            if "categorization_of_assessment_unit_total" in cols and "firka" in cols:
                lines.append(
                    f"SELECT firka, district FROM {schema.name} "
                    f"WHERE assessment_year = '{year_val}' "
                    "AND categorization_of_assessment_unit_total = 'over_exploited';"
                )
        lines.append("```")
        lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Document assembly + validation
# --------------------------------------------------------------------------- #

def build_markdown(
    schemas: list[TableSchema],
    connection: sqlite3.Connection,
    value_semantics: dict[str, dict[str, list[str]]],
) -> str:
    all_columns = {c for schema in schemas for c in schema.column_names}
    parts: list[str] = [
        "# AquaMind AI — Groundwater Database Schema",
        "",
        "> Auto-generated by `schema_generator.py`. Do not edit by hand; rerun the generator instead.",
        "> This is the authoritative schema reference for the SQL Generation LLM. The LLM must rely on "
        "this document and must not inspect the database directly.",
        "",
    ]
    parts += section_overview(schemas, connection)
    parts += section_table_documentation(schemas)
    parts += section_table_purpose(schemas)
    parts += section_query_routing()
    parts += section_relationships(schemas)
    parts += section_indexes(schemas)
    parts += section_sql_notes()
    parts += section_domain_terminology(all_columns)
    parts += section_example_questions()
    parts += section_column_usage()
    parts += section_table_selection()
    parts += section_important_columns(schemas)
    parts += section_aggregation_columns(schemas)
    parts += section_search_keywords(schemas)
    parts += section_common_patterns()
    parts += section_value_semantics(value_semantics)
    parts += section_user_vocabulary()
    parts += section_query_examples_per_table(schemas)

    # LLM-oriented guidance sections (improve SQL generation accuracy).
    parts += section_canonical_table_selection(schemas)
    parts += section_cross_table_rules(schemas)
    parts += section_preferred_filter_columns(schemas)
    parts += section_preferred_sorting_columns(schemas)
    parts += section_synonym_dictionary(schemas)
    parts += section_forbidden_sql()
    parts += section_case_sensitivity(connection, schemas)
    parts += section_frequently_used_columns(schemas)
    parts += section_time_columns(schemas)
    parts += section_example_sql_queries(connection, schemas)
    return "\n".join(parts).rstrip() + "\n"


#: Every section that must appear exactly once in the generated document.
EXPECTED_SECTIONS: tuple[str, ...] = (
    "Database Overview", "Table Documentation", "Table Purpose", "Query Routing Guide",
    "Database Relationships", "Index Documentation", "SQL Generation Notes", "Domain Terminology",
    "Example User Questions", "Column Usage", "Table Selection Guide", "Important Columns",
    "Aggregation Columns", "Search Keywords", "Common SQL Patterns", "Column Value Semantics",
    "Column Aliases / User Vocabulary", "Query Examples Per Table",
    # New LLM-oriented sections:
    "Canonical Table Selection Rules", "Cross Table Query Rules", "Preferred Filter Columns",
    "Preferred Sorting Columns", "Synonym Dictionary", "Forbidden SQL", "Case Sensitivity Notes",
    "Frequently Used Columns", "Time Columns", "Example SQL Queries",
)


def validate_documentation(document: str, schemas: list[TableSchema]) -> bool:
    """Verify every table, column, PK, index, row count, and section is present exactly once."""
    ok = True

    heading_lines = [line.strip() for line in document.splitlines()]
    for title in EXPECTED_SECTIONS:
        count = heading_lines.count(f"## {title}")
        if count == 0:
            logger.error("Missing section: %s", title)
            ok = False
        elif count > 1:
            logger.error("Duplicate section (%d occurrences): %s", count, title)
            ok = False

    for schema in schemas:
        if f"`{schema.name}`" not in document:
            logger.error("Table '%s' not documented.", schema.name)
            ok = False
        if f"{schema.row_count:,}" not in document and str(schema.row_count) not in document:
            logger.error("Row count for '%s' not documented.", schema.name)
            ok = False
        for column in schema.columns:
            if f"`{column.name}`" not in document:
                logger.error("Column '%s.%s' not documented.", schema.name, column.name)
                ok = False
        if not any(c.primary_key for c in schema.columns):
            logger.error("No primary key found for '%s'.", schema.name)
            ok = False
        elif "PRIMARY KEY" not in document:
            logger.error("Primary key not documented for '%s'.", schema.name)
            ok = False
        for index in schema.indexes:
            if f"`{index.name}`" not in document:
                logger.error("Index '%s' not documented.", index.name)
                ok = False
    return ok


def generate() -> bool:
    """Regenerate database_schema.md from the database and validate coverage."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Reading schema from %s", DB_PATH)
    try:
        connection = connect_readonly(DB_PATH)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return False

    try:
        schemas = [describe_table(connection, table) for table in list_tables(connection)]
        if not schemas:
            logger.error("No tables found in the database.")
            return False

        value_semantics = {s.name: sample_value_semantics(connection, s) for s in schemas}
        document = build_markdown(schemas, connection, value_semantics)
    finally:
        connection.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(document, encoding="utf-8")
    logger.info("Wrote %s (%d lines)", OUTPUT_PATH, document.count("\n") + 1)

    total_columns = sum(len(s.columns) for s in schemas)
    total_indexes = sum(len(s.indexes) for s in schemas)
    passed = validate_documentation(document, schemas)
    logger.info(
        "VALIDATION: %d tables, %d columns, %d indexes, %d sections documented -> %s",
        len(schemas), total_columns, total_indexes, len(EXPECTED_SECTIONS), "PASS" if passed else "FAIL",
    )
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if generate() else 1)
