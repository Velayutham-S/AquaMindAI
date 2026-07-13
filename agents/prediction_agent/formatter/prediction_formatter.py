"""Prediction Formatter for the AquaMind AI Prediction Agent.

Single responsibility: convert a runtime prediction result into the Prediction
Agent's structured evidence response -- the prediction counterpart of the Data
Agent's evidence formatter and the Knowledge Agent's knowledge formatter.

This is a **pure formatting layer**. It performs NO inference, NO feature
engineering, NO model loading, NO LLM calls, NO summarization, NO natural-language
generation, NO logging and NO I/O. It only reshapes the prediction result into a
deterministic response envelope, with no side effects.

Public interface:
    PredictionFormatter().format(prediction_result) -> dict
"""

from __future__ import annotations

import math
from typing import Any


class PredictionFormatter:
    """Transforms a runtime prediction result into a structured response."""

    AGENT_NAME: str = "prediction_agent"
    QUERY_TYPE: str = "prediction"
    PREDICTION_METHOD: str = "machine_learning"
    STATUS_SUCCESS: str = "SUCCESS"
    STATUS_NO_PREDICTION: str = "NO_PREDICTION"

    #: Target column -> human-readable unit. The Prediction Agent currently
    #: predicts groundwater level only; extend this map when new targets are added.
    _UNITS: dict[str, str] = {"groundwater_level_m": "metres below ground level"}
    _DEFAULT_UNIT: str = "metres below ground level"

    def format(self, prediction_result: "dict[str, Any] | None") -> dict[str, Any]:
        """Return a structured response envelope for one prediction result.

        A missing result or a missing / non-finite ``predicted_value`` yields a
        ``NO_PREDICTION`` response. This method never raises.
        """
        if not self._has_prediction(prediction_result):
            return {
                "agent_name": self.AGENT_NAME,
                "status": self.STATUS_NO_PREDICTION,
                "query_type": self.QUERY_TYPE,
                "prediction": None,
            }

        target = prediction_result.get("target")
        return {
            "agent_name": self.AGENT_NAME,
            "status": self.STATUS_SUCCESS,
            "query_type": self.QUERY_TYPE,
            "prediction_method": self.PREDICTION_METHOD,
            "model_name": prediction_result.get("model_name"),
            "prediction": {
                "district": prediction_result.get("district"),
                "prediction_year": prediction_result.get("prediction_year"),
                "prediction_month": prediction_result.get("prediction_month"),
                "target": target,
                "predicted_value": float(prediction_result["predicted_value"]),
                "unit": self._resolve_unit(target),
            },
        }

    @staticmethod
    def _has_prediction(prediction_result: "dict[str, Any] | None") -> bool:
        """True only when a finite ``predicted_value`` is present."""
        if not prediction_result:
            return False
        value = prediction_result.get("predicted_value")
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _resolve_unit(cls, target: Any) -> str:
        """Map a target to its unit (falls back to the groundwater-level unit)."""
        return cls._UNITS.get(target, cls._DEFAULT_UNIT)
