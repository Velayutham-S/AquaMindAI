"""Model evaluation and selection for the AquaMind AI Prediction Agent.

Single responsibility: score fitted candidate pipelines on the validation split
with standard regression metrics, and objectively select the best model.

Metrics
-------
* MAE  -- mean absolute error (same units as the target, metres).
* RMSE -- root mean squared error (penalises large misses).
* R2   -- coefficient of determination (variance explained; higher is better).
* MAPE -- mean absolute percentage error (scale-free; noisy when targets are
  near zero, so reported for reference).

Selection is driven by the configured metric ("rmse"/"mae"/"mape" -> lower is
better; "r2" -> higher is better), so no algorithm is favoured a priori.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("aquamind.prediction.model_evaluator")

#: Metrics where a smaller value is better (used to decide selection direction).
_LOWER_IS_BETTER = {"mae", "rmse", "mape"}

#: Targets whose absolute value is below this (metres) are excluded from MAPE.
#: MAPE divides by the target, so values near zero make it explode and become
#: meaningless -- this makes MAPE reported "when appropriate", as required.
_MAPE_MIN_ABS_TARGET = 1.0


@dataclass
class EvaluationResult:
    """Validation metrics for one candidate model."""

    name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    fit_seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class ModelEvaluator:
    """Scores candidates and selects the best by the configured metric."""

    def evaluate(self, name: str, pipeline, x_val: pd.DataFrame, y_val: pd.Series,
                 fit_seconds: float = 0.0) -> EvaluationResult:
        """Compute validation metrics for a single fitted pipeline."""
        from sklearn.metrics import (
            mean_absolute_error,
            mean_absolute_percentage_error,
            r2_score,
            root_mean_squared_error,
        )

        predictions = pipeline.predict(x_val)
        return EvaluationResult(
            name=name,
            mae=float(mean_absolute_error(y_val, predictions)),
            rmse=float(root_mean_squared_error(y_val, predictions)),
            r2=float(r2_score(y_val, predictions)),
            mape=self._robust_mape(y_val, predictions, mean_absolute_percentage_error),
            fit_seconds=float(fit_seconds),
        )

    @staticmethod
    def _robust_mape(y_true, y_pred, mape_fn) -> float:
        """MAPE over targets with |value| >= threshold (``nan`` if none qualify).

        Excluding near-zero targets keeps MAPE finite and interpretable for a
        target that can legitimately be close to zero.
        """
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        mask = np.abs(y_true_arr) >= _MAPE_MIN_ABS_TARGET
        if not mask.any():
            return float("nan")
        return float(mape_fn(y_true_arr[mask], y_pred_arr[mask]))

    def compare(self, trained_models: dict, x_val: pd.DataFrame,
                y_val: pd.Series) -> list[EvaluationResult]:
        """Evaluate all candidates; returns results in candidate order."""
        results: list[EvaluationResult] = []
        for name, trained in trained_models.items():
            result = self.evaluate(name, trained.pipeline, x_val, y_val, trained.fit_seconds)
            logger.info(
                "  %-16s MAE=%.4f RMSE=%.4f R2=%.4f MAPE=%.4f",
                name, result.mae, result.rmse, result.r2, result.mape,
            )
            results.append(result)
        return results

    def select_best(self, results: list[EvaluationResult],
                    metric: str = "rmse") -> EvaluationResult:
        """Return the best result by ``metric`` (direction inferred from metric)."""
        if not results:
            raise ValueError("Cannot select a best model from an empty result list.")
        metric = metric.lower()
        if metric not in {"mae", "rmse", "r2", "mape"}:
            raise ValueError(f"Unsupported selection metric: {metric!r}")
        reverse = metric not in _LOWER_IS_BETTER  # r2 -> higher is better
        ranked = sorted(results, key=lambda r: getattr(r, metric), reverse=reverse)
        best = ranked[0]
        logger.info("Selected best model '%s' by %s.", best.name, metric)
        return best
