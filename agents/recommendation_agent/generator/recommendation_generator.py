"""Recommendation Generator component for AquaMind AI.

Single responsibility: given the aggregated specialist-agent evidence AND a
positive :class:`RecommendationDecision`, produce a structured
:class:`RecommendationResponse` (a list of actionable recommendations).

It uses ONLY the information contained in the aggregated evidence. It never
invents facts, never modifies/summarizes evidence, never contradicts the
specialist agents, and never produces the final natural-language answer.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import (
    GENERATOR_PROMPT_PATH,
    LlmClient,
    build_evidence_view,
    build_generator_client,
    extract_json_object,
    load_prompt,
)
from ..decision.decision_models import RecommendationDecision
from .recommendation_models import RecommendationResponse


class RecommendationGenerator:
    """Generates grounded recommendations from aggregated evidence + a decision."""

    def __init__(self, client: LlmClient | None = None, prompt: str | None = None) -> None:
        # Recommendation Generator LLM -> Groq (llama-3.3-70b-versatile).
        self._client = client or build_generator_client()
        self._prompt = prompt if prompt is not None else load_prompt(GENERATOR_PROMPT_PATH)

    def generate(
        self, aggregated: Any, decision: RecommendationDecision
    ) -> RecommendationResponse:
        """Return recommendations grounded in the aggregated evidence.

        Should be called ONLY when ``decision.recommendation_required`` is true;
        this is asserted defensively so the generator never runs unnecessarily.
        """
        if not decision.recommendation_required:
            raise ValueError(
                "RecommendationGenerator invoked when recommendation_required is false."
            )
        evidence = build_evidence_view(aggregated)
        prompt = self._build_prompt(evidence, decision)
        raw = self._client.complete(prompt)
        payload = extract_json_object(raw)
        return RecommendationResponse.parse(payload)

    def _build_prompt(
        self, evidence: dict[str, Any], decision: RecommendationDecision
    ) -> str:
        return (
            f"{self._prompt}\n\n"
            "==========================================================\n"
            "WHY RECOMMENDATIONS ARE REQUIRED (from the Recommendation Decision)\n"
            "==========================================================\n"
            f"{decision.reason}\n\n"
            "==========================================================\n"
            "AGGREGATED AGENT EVIDENCE (the ONLY facts you may use)\n"
            "==========================================================\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2, default=str)}\n\n"
            "Return ONLY the recommendation JSON object."
        )
