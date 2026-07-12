"""Preprocess the firka GEC assessment workbooks into master_firka.csv.

Merges every yearly firka assessment workbook in ``structured_data/firka`` into
one master dataset. The three-level hierarchical (merged-cell) header is
flattened into single meaningful column names, columns are unioned across years
(absent columns left NULL), and the state grand-total row is excluded to keep
one row per firka.
"""

from __future__ import annotations

from preprocessing_common import GecConfig, build_gec_master

CONFIG = GecConfig(
    category_folder="firka",
    master_filename="master_firka.csv",
    identifier_columns=("state", "district", "firka", "watershed_district"),
)


def main() -> None:
    build_gec_master(CONFIG)


if __name__ == "__main__":
    main()
