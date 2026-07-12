"""Preprocess the groundwater level datasets into master_groundwater_level.csv.

Merges every groundwater level CSV in ``structured_data/groundwater_level`` into
one tall master dataset. Each source file carries a differently named
measurement column (Quarterly Manual, Telemetry Quadridaily, Telemetry 6
Hourly); these are normalized into a single ``groundwater_level_m`` value column
plus a ``measurement_type`` column.
"""

from __future__ import annotations

from preprocessing_common import TimeSeriesConfig, build_timeseries_master

CONFIG = TimeSeriesConfig(
    category_folder="groundwater_level",
    master_filename="master_groundwater_level.csv",
    value_column="groundwater_level_m",
    measurement_subject="Groundwater Level",
    extra_rename={"RL_MSL": "rl_msl"},
)


def main() -> None:
    build_timeseries_master(CONFIG)


if __name__ == "__main__":
    main()
