"""Data models + validation for the Recommendation Generator output.

Pure data. A :class:`RecommendationResponse` is a status plus an ordered list of
:class:`Recommendation` items (title + description). It carries no evidence and
no final natural-language answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RecommendationValidationError(Exception):
    """The raw recommendation JSON was malformed."""


STATUS_SUCCESS = "SUCCESS"
STATUS_NO_RECOMMENDATIONS = "NO_RECOMMENDATIONS"


@dataclass(frozen=True)
class Recommendation:
    """One actionable recommendation grounded in the aggregated evidence."""

    title: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "description": self.description}


@dataclass(frozen=True)
class RecommendationResponse:
    """The structured output of the Recommendation Generator."""

    status: str
    recommendations: tuple[Recommendation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "recommendations": [item.to_dict() for item in self.recommendations],
        }

    @classmethod
    def parse(cls, payload: Any) -> "RecommendationResponse":
        """Validate a raw JSON object into a ``RecommendationResponse``."""
        if not isinstance(payload, dict):
            raise RecommendationValidationError(
                f"Recommendation response must be a JSON object, got {type(payload).__name__}."
            )
        raw_items = payload.get("recommendations")
        if not isinstance(raw_items, list) or not raw_items:
            raise RecommendationValidationError(
                "'recommendations' must be a non-empty list."
            )
        items: list[Recommendation] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise RecommendationValidationError(
                    f"recommendation[{index}] must be an object."
                )
            title = item.get("title")
            description = item.get("description")
            if not isinstance(title, str) or not title.strip():
                raise RecommendationValidationError(
                    f"recommendation[{index}].title must be a non-empty string."
                )
            if not isinstance(description, str) or not description.strip():
                raise RecommendationValidationError(
                    f"recommendation[{index}].description must be a non-empty string."
                )
            items.append(Recommendation(title=title.strip(), description=description.strip()))
        status = payload.get("status") or STATUS_SUCCESS
        if not isinstance(status, str):
            raise RecommendationValidationError("'status' must be a string.")
        return cls(status=status, recommendations=tuple(items))
