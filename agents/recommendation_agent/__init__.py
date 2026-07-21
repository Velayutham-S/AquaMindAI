"""Recommendation Layer for AquaMind AI.

Two independent components that run AFTER the Response Aggregator and BEFORE the
(future) Response Generator:

* :class:`RecommendationDecider` -- decides whether recommendations are required,
  looking only at the aggregated specialist-agent evidence.
* :class:`RecommendationGenerator` -- when required, produces a structured
  :class:`RecommendationResponse` grounded solely in that evidence.

Neither component generates the final user response.
"""

from .config import LlmClient, build_evidence_view
from .decision.decision_models import DecisionValidationError, RecommendationDecision
from .decision.recommendation_decision import RecommendationDecider
from .generator.recommendation_generator import RecommendationGenerator
from .generator.recommendation_models import (
    Recommendation,
    RecommendationResponse,
    RecommendationValidationError,
)

__all__ = [
    "LlmClient",
    "build_evidence_view",
    "RecommendationDecider",
    "RecommendationDecision",
    "DecisionValidationError",
    "RecommendationGenerator",
    "RecommendationResponse",
    "Recommendation",
    "RecommendationValidationError",
]
