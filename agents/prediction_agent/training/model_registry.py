"""Model registry for the AquaMind AI Prediction Agent.

Single responsibility: persist and reload the trained artifacts for a task.

For each task two files are written under ``models/``:

* ``<task>_model.joblib``   -- the complete fitted scikit-learn pipeline
  (preprocessor + estimator). The runtime Prediction Agent loads ONLY this and
  calls ``predict``; it never retrains.
* ``<task>_metadata.json``  -- feature metadata, preprocessing metadata,
  training configuration and evaluation metrics (human- and machine-readable).

The registry performs no training, evaluation or feature engineering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger("aquamind.prediction.model_registry")


class ModelRegistry:
    """Saves and loads trained pipelines and their metadata (dependency-injected)."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = Path(models_dir)

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    def model_path(self, task_name: str) -> Path:
        return self._models_dir / f"{task_name}_model.joblib"

    def metadata_path(self, task_name: str) -> Path:
        return self._models_dir / f"{task_name}_metadata.json"

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, task_name: str, pipeline: Any, metadata: dict) -> tuple[Path, Path]:
        """Persist the fitted pipeline and its metadata; returns both paths."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        model_file = self.model_path(task_name)
        metadata_file = self.metadata_path(task_name)

        joblib.dump(pipeline, model_file)
        metadata_file.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Saved model -> %s (%.2f KB)", model_file,
                    self.model_file_size_bytes(task_name) / 1024)
        logger.info("Saved metadata -> %s", metadata_file)
        return model_file, metadata_file

    def load(self, task_name: str) -> tuple[Any, dict]:
        """Load and return ``(pipeline, metadata)`` for a task."""
        model_file = self.model_path(task_name)
        metadata_file = self.metadata_path(task_name)
        if not model_file.exists():
            raise FileNotFoundError(f"No saved model for task '{task_name}': {model_file}")

        pipeline = joblib.load(model_file)
        metadata: dict = {}
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        return pipeline, metadata

    def model_file_size_bytes(self, task_name: str) -> int:
        """Size of the saved model file in bytes (0 if not yet saved)."""
        model_file = self.model_path(task_name)
        return model_file.stat().st_size if model_file.exists() else 0
