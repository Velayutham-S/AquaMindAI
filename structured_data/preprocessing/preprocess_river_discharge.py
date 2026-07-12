"""Preprocess the river discharge datasets into master_river_discharge.csv.

Merges every river discharge CSV in ``structured_data/river_discharge`` into one
tall master dataset. The measurement column is normalized into a single
``river_discharge_m3s`` value column plus a ``measurement_type`` column. Empty
source files (header only) are skipped with logging.
"""

from __future__ import annotations

from preprocessing_common import TimeSeriesConfig, build_timeseries_master

CONFIG = TimeSeriesConfig(
    category_folder="river_discharge",
    master_filename="master_river_discharge.csv",
    value_column="river_discharge_m3s",
    measurement_subject="River Water Discharge",
)


def main() -> None:
    build_timeseries_master(CONFIG)


if __name__ == "__main__":
    main()
