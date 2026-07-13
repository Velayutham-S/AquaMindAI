"""Dataset loading for the AquaMind AI Prediction Agent (offline training).

Single responsibility: read exactly one master dataset for a prediction task
into a clean :class:`pandas.DataFrame` ready for feature engineering.

It does no feature engineering, no encoding and no model logic. It only:

* reads **only** the columns the task needs (selective load -> small memory
  footprint even for multi-million-row datasets),
* drops rows with a missing target or missing required columns (unusable rows),
* optionally draws a **deterministic** random sample so offline training stays
  fast and reproducible.

The loader never writes to disk and never mutates the source datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("aquamind.prediction.dataset_loader")


class DatasetLoadError(RuntimeError):
    """Raised when a task's master dataset cannot be loaded."""


class DatasetLoader:
    """Loads and lightly cleans a task's master dataset (dependency-injected)."""

    def __init__(self, master_dir: Path, random_state: int = 42) -> None:
        self._master_dir = Path(master_dir)
        self._random_state = random_state

    def load(self, task, max_rows: int | None = None) -> pd.DataFrame:
        """Return the cleaned (optionally sampled) DataFrame for ``task``.

        Parameters
        ----------
        task:
            A ``PredictionTaskConfig`` describing the dataset, target and the
            raw columns to read.
        max_rows:
            If set and the cleaned dataset is larger, a deterministic random
            sample of this many rows is returned. ``None`` uses all rows.
        """
        path = self._master_dir / task.dataset_filename
        if not path.exists():
            raise DatasetLoadError(f"Master dataset not found for '{task.name}': {path}")

        columns = task.raw_columns_to_load
        dtype = self._build_dtype_map(task)

        try:
            frame = pd.read_csv(path, usecols=columns, dtype=dtype)
        except ValueError as error:
            raise DatasetLoadError(
                f"Failed to read required columns {columns} from {path}: {error}"
            ) from error

        total_rows = len(frame)
        frame = self._drop_unusable_rows(frame, task)
        usable_rows = len(frame)

        sampled = False
        if max_rows is not None and usable_rows > max_rows:
            frame = frame.sample(n=max_rows, random_state=self._random_state).reset_index(drop=True)
            sampled = True
        else:
            frame = frame.reset_index(drop=True)

        logger.info(
            "Loaded '%s': %d rows read, %d usable, %d used%s",
            task.name, total_rows, usable_rows, len(frame),
            " (deterministic sample)" if sampled else "",
        )
        return frame

    @staticmethod
    def _build_dtype_map(task) -> dict[str, str]:
        """Choose memory-efficient dtypes: categories for text, float32 numerics."""
        dtype: dict[str, str] = {column: "category" for column in task.categorical_features}
        if task.datetime_column:
            dtype[task.datetime_column] = "object"  # parsed later, kept as text here
        numeric_raw = [c for c in task.numeric_features if c not in task.derived_features]
        for column in [*numeric_raw, task.target_column]:
            dtype[column] = "float32"
        return dtype

    @staticmethod
    def _drop_unusable_rows(frame: pd.DataFrame, task) -> pd.DataFrame:
        """Drop rows missing the target or any required raw column."""
        subset = [task.target_column, *task.required_columns]
        present = [column for column in subset if column in frame.columns]
        return frame.dropna(subset=present)
