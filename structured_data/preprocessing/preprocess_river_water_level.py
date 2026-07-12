"""Preprocess the river water level datasets into master_river_water_level.csv.

Merges every river water level CSV in ``structured_data/river_water_level`` into
one tall master dataset. The measurement column is normalized into a single
``river_water_level_m`` value column plus a ``measurement_type`` column. The
category-specific gauge attributes (discharge availability, zero-gauge reduced
level, mean sea level) are preserved.
"""

from __future__ import annotations

from preprocessing_common import TimeSeriesConfig, build_timeseries_master

CONFIG = TimeSeriesConfig(
    category_folder="river_water_level",
    master_filename="master_river_water_level.csv",
    value_column="river_water_level_m",
    measurement_subject="River Water Level",
    extra_rename={
        "Is_DischargeDataAvailable": "is_discharge_data_available",
        "RL_of_zeroGauge": "rl_of_zero_gauge",
        "MeanSeaLevel": "mean_sea_level",
    },
)


def main() -> None:
    build_timeseries_master(CONFIG)


if __name__ == "__main__":
    main()
