"""Recommendation Decision component for AquaMind AI.

Single responsibility: given ONLY the aggregated specialist-agent evidence,
decide whether additional recommendations would improve the final answer. It
returns a :class:`RecommendationDecision` (a boolean + reason) and NOTHING else.

It never generates recommendations, never rewrites/summarizes evidence, and
never sees the user query, conversation memory, planner decision or
AgentRequest -- only the agents' evidence (see ``build_evidence_view``).
"""

from __future__ import annotations

import json
from typing import Any

from ..config import (
    DECISION_PROMPT_PATH,
    LlmClient,
    build_evidence_view,
    extract_json_object,
    load_prompt,
)
from .decision_models import RecommendationDecision


class RecommendationDecider:
    """Decides whether recommendations are required for an aggregated response."""

    def __init__(self, client: LlmClient | None = None, prompt: str | None = None) -> None:
        # Dependency injection with production defaults.
        self._client = client or LlmClient()
        self._prompt = prompt if prompt is not None else load_prompt(DECISION_PROMPT_PATH)

    def decide(self, aggregated: Any) -> RecommendationDecision:
        """Return the recommendation decision for an aggregated response.

        ``aggregated`` may be an ``AggregatedUniversalAgentResponse`` or its dict.
        Only its agent evidence is used.
        """
        evidence = build_evidence_view(aggregated)
        prompt = self._build_prompt(evidence)
        raw = self._client.complete(prompt)
        payload = extract_json_object(raw)
        return RecommendationDecision.parse(payload)

    def _build_prompt(self, evidence: dict[str, Any]) -> str:
        return (
            f"{self._prompt}\n\n"
            "==========================================================\n"
            "AGGREGATED AGENT EVIDENCE (the ONLY input you may use)\n"
            "==========================================================\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2, default=str)}\n\n"
            "Return ONLY the decision JSON object."
        )
