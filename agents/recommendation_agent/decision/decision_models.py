"""Data model + validation for the Recommendation Decision.

Pure data. The Recommendation Decision emits exactly one
:class:`RecommendationDecision`: a boolean plus a short justification. It never
contains recommendations, evidence, or the user query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class DecisionValidationError(Exception):
    """The raw decision JSON was missing fields or had the wrong types."""


@dataclass(frozen=True)
class RecommendationDecision:
    """Whether additional recommendations would improve the final answer."""

    recommendation_required: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_required": self.recommendation_required,
            "reason": self.reason,
        }

    @classmethod
    def parse(cls, payload: Any) -> "RecommendationDecision":
        """Validate a raw JSON object into a ``RecommendationDecision``."""
        if not isinstance(payload, dict):
            raise DecisionValidationError(
                f"Decision must be a JSON object, got {type(payload).__name__}."
            )
        if "recommendation_required" not in payload:
            raise DecisionValidationError("Missing 'recommendation_required'.")
        required = payload["recommendation_required"]
        if not isinstance(required, bool):
            raise DecisionValidationError("'recommendation_required' must be a boolean.")
        reason = payload.get("reason", "")
        if not isinstance(reason, str) or not reason.strip():
            raise DecisionValidationError("'reason' must be a non-empty string.")
        return cls(recommendation_required=required, reason=reason.strip())
