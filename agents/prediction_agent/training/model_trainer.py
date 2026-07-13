"""Model training for the AquaMind AI Prediction Agent (offline).

Single responsibility: fit each candidate regressor as a self-contained
scikit-learn ``Pipeline`` of ``preprocessor -> estimator`` on the training
split, recording how long each fit took.

Bundling the preprocessor with the estimator is deliberate: the saved artifact
is a complete pipeline, so at inference the runtime agent feeds raw feature
rows straight in and the fitted imputation/scaling/encoding are applied
automatically -- there is no way for training and inference preprocessing to
drift apart.

This module selects nothing and evaluates nothing; it only fits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger("aquamind.prediction.model_trainer")


@dataclass
class TrainedModel:
    """A fitted candidate pipeline plus its training wall-clock time."""

    name: str
    pipeline: Any
    fit_seconds: float


class ModelTrainer:
    """Fits candidate models as ``preprocessor + estimator`` pipelines."""

    def __init__(self, preprocessor_factory: Callable[[], Any]) -> None:
        #: A zero-arg factory returning a FRESH, unfitted preprocessor per model
        #: (so no fitted state is shared between candidates).
        self._preprocessor_factory = preprocessor_factory

    def train(
        self,
        candidate_models: "dict[str, Any]",
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "dict[str, TrainedModel]":
        """Fit every candidate and return the fitted pipelines by name."""
        from sklearn.pipeline import Pipeline

        trained: "dict[str, TrainedModel]" = {}
        for name, estimator in candidate_models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self._preprocessor_factory()),
                    ("model", estimator),
                ]
            )
            logger.info("Training candidate '%s'...", name)
            start = perf_counter()
            pipeline.fit(x_train, y_train)
            elapsed = perf_counter() - start
            trained[name] = TrainedModel(name=name, pipeline=pipeline, fit_seconds=elapsed)
            logger.info("  '%s' trained in %.2fs", name, elapsed)
        return trained
