"""Preprocess the district GEC assessment workbooks into master_district.csv.

Merges every yearly district assessment workbook in
``structured_data/district`` into one master dataset. The three-level
hierarchical (merged-cell) header is flattened into single meaningful column
names, columns are unioned across years (absent columns left NULL), and the
state grand-total row is excluded to keep one row per district.
"""

from __future__ import annotations

from preprocessing_common import GecConfig, build_gec_master

CONFIG = GecConfig(
    category_folder="district",
    master_filename="master_district.csv",
    identifier_columns=("state", "district", "assessment_unit"),
)


def main() -> None:
    build_gec_master(CONFIG)


if __name__ == "__main__":
    main()
