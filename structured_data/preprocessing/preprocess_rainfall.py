"""Preprocess the rainfall datasets into master_rainfall.csv.

Merges every rainfall CSV in ``structured_data/rainfall`` into one tall master
dataset. The differently named measurement columns (Manual Daily, Telemetry
Hourly) are normalized into a single ``rainfall_mm`` value column plus a
``measurement_type`` column.
"""

from __future__ import annotations

from preprocessing_common import TimeSeriesConfig, build_timeseries_master

CONFIG = TimeSeriesConfig(
    category_folder="rainfall",
    master_filename="master_rainfall.csv",
    value_column="rainfall_mm",
    measurement_subject="Rainfall",
)


def main() -> None:
    build_timeseries_master(CONFIG)


if __name__ == "__main__":
    main()
