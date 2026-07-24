"""Routing-decision validation for the AquaMind AI Supervisor Planner.

Single responsibility: turn the raw JSON object returned by the Planner LLM into
a strongly-typed, fully-consistent ``PlannerDecision`` — or reject it. It never
executes agents, never calls an LLM, and never repairs intent by guessing; it
only enforces that the decision is well-formed and internally consistent.

Out-of-domain (non-groundwater) intents (general_chat / out_of_scope) are a
terminal decision: they select no specialist agent, so they bypass the
groundwater agent-consistency checks and are validated only for a clean terminal
shape (confident, non-clarifying, no agent).

Enforced checks (groundwater decisions):
* required fields present and correctly typed,
* enum values valid (intent, confidence),
* agents are ONLY data_agent / knowledge_agent / prediction_agent
  (any other name — including general_llm — is rejected in this milestone),
* execution_order is a permutation of agents (same agents, no extras/missing),
* clarification consistency (clarifying => empty agents + a question; otherwise
  => a null question and at least one agent),
* confidence consistency (LOW <=> requires_clarification),
* intent consistency (single-intent => the matching single agent; mixed_query
  => two or more agents).

Any failure raises ``PlannerValidationError``.
"""

from __future__ import annotations

from typing import Any

from .planner_models import (
    OUT_OF_DOMAIN_INTENTS,
    AgentName,
    ConfidenceLevel,
    IntentType,
    PlannerDecision,
    PlannerValidationError,
)

_REQUIRED_FIELDS = (
    "intent",
    "confidence",
    "requires_clarification",
    "clarification_question",
    "agents",
    "execution_order",
    "reason",
)

#: Single-intent -> the exactly-one agent that must be selected.
_SINGLE_INTENT_AGENT = {
    IntentType.DATA_QUERY: AgentName.DATA_AGENT,
    IntentType.KNOWLEDGE_QUERY: AgentName.KNOWLEDGE_AGENT,
    IntentType.PREDICTION_QUERY: AgentName.PREDICTION_AGENT,
    IntentType.SYSTEM_INFORMATION: AgentName.KNOWLEDGE_AGENT,
}


class PlannerValidator:
    """Validates a raw routing dict into a consistent ``PlannerDecision``."""

    def validate(self, payload: Any) -> PlannerDecision:
        """Return a validated ``PlannerDecision`` or raise ``PlannerValidationError``."""
        if not isinstance(payload, dict):
            raise PlannerValidationError(
                f"Routing decision must be a JSON object, got {type(payload).__name__}.",
                payload=payload,
            )

        missing = [field for field in _REQUIRED_FIELDS if field not in payload]
        if missing:
            raise PlannerValidationError(
                f"Missing required field(s): {', '.join(missing)}.", payload=payload
            )

        intent = self._parse_enum(payload["intent"], IntentType, "intent", payload)
        confidence = self._parse_enum(payload["confidence"], ConfidenceLevel, "confidence", payload)

        requires_clarification = payload["requires_clarification"]
        if not isinstance(requires_clarification, bool):
            raise PlannerValidationError(
                "'requires_clarification' must be a boolean.", payload=payload
            )

        clarification_question = payload["clarification_question"]
        if clarification_question is not None and not isinstance(clarification_question, str):
            raise PlannerValidationError(
                "'clarification_question' must be a string or null.", payload=payload
            )

        reason = payload["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise PlannerValidationError("'reason' must be a non-empty string.", payload=payload)

        # Out-of-domain (non-groundwater) requests are a terminal decision: no
        # specialist agent runs. Handle them here -- before the strict agent
        # parsing and the groundwater consistency checks -- so the decision is
        # accepted gracefully instead of being rejected. Groundwater intents fall
        # through to the unchanged validation path below.
        if intent in OUT_OF_DOMAIN_INTENTS:
            return self._build_out_of_domain(
                intent, confidence, requires_clarification, clarification_question, reason, payload
            )

        agents = self._parse_agents(payload["agents"], "agents", payload)
        execution_order = self._parse_agents(payload["execution_order"], "execution_order", payload)

        self._check_execution_order(agents, execution_order, payload)
        self._check_clarification(requires_clarification, clarification_question, agents, payload)
        self._check_confidence(confidence, requires_clarification, payload)
        self._check_intent(intent, requires_clarification, agents, payload)

        district = self._parse_optional_str(payload.get("district"), "district", payload)
        firka = self._parse_optional_str(payload.get("firka"), "firka", payload)
        target_year = self._parse_optional_year(payload.get("target_year"), payload)

        return PlannerDecision(
            intent=intent,
            confidence=confidence,
            requires_clarification=requires_clarification,
            clarification_question=(clarification_question or None),
            agents=agents,
            execution_order=execution_order,
            reason=reason.strip(),
            district=district,
            firka=firka,
            target_year=target_year,
        )

    # ------------------------------------------------------------------ #
    # Out-of-domain (terminal) decision
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_out_of_domain(
        intent, confidence, requires_clarification, clarification_question, reason, payload
    ) -> PlannerDecision:
        """Validate and build a terminal out-of-domain decision (no agent runs).

        An out-of-domain query is not a groundwater request AquaMind AI handles,
        so no specialist agent is selected and no prediction slots apply. The
        decision must be confident and non-clarifying; the ``agents`` field is
        irrelevant here (nothing executes) and is normalized to empty.
        """
        if requires_clarification:
            raise PlannerValidationError(
                "An out-of-domain decision must not require clarification.", payload=payload
            )
        if clarification_question is not None:
            raise PlannerValidationError(
                "An out-of-domain decision must have a null clarification_question.",
                payload=payload,
            )
        if confidence is ConfidenceLevel.LOW:
            raise PlannerValidationError(
                "An out-of-domain decision must not use LOW confidence.", payload=payload
            )
        return PlannerDecision(
            intent=intent,
            confidence=confidence,
            requires_clarification=False,
            clarification_question=None,
            agents=(),
            execution_order=(),
            reason=reason.strip(),
            district=None,
            firka=None,
            target_year=None,
        )

    # ------------------------------------------------------------------ #
    # Field parsers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_enum(value: Any, enum_cls: type, field: str, payload: Any):
        try:
            return enum_cls(value)
        except ValueError as error:
            allowed = ", ".join(member.value for member in enum_cls)
            raise PlannerValidationError(
                f"Invalid '{field}' value {value!r}. Allowed: {allowed}.", payload=payload
            ) from error

    @staticmethod
    def _parse_optional_str(value: Any, field: str, payload: Any) -> str | None:
        """Optional structured slot: a non-empty string, or None (null/absent)."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise PlannerValidationError(
                f"'{field}' must be a string or null.", payload=payload
            )
        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _parse_optional_year(value: Any, payload: Any) -> int | None:
        """Optional forecast year: an int, a digit-string, or None (null/absent)."""
        if value is None:
            return None
        if isinstance(value, bool):  # bool is an int subclass; reject explicitly.
            raise PlannerValidationError("'target_year' must be an integer or null.", payload=payload)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            if cleaned.isdigit():
                return int(cleaned)
        raise PlannerValidationError(
            "'target_year' must be an integer or null.", payload=payload
        )

    @staticmethod
    def _parse_agents(value: Any, field: str, payload: Any) -> tuple[AgentName, ...]:
        if not isinstance(value, list):
            raise PlannerValidationError(f"'{field}' must be a list.", payload=payload)
        agents: list[AgentName] = []
        for item in value:
            try:
                agents.append(AgentName(item))
            except ValueError as error:
                allowed = ", ".join(member.value for member in AgentName)
                raise PlannerValidationError(
                    f"'{field}' contains disallowed agent {item!r}. "
                    f"Allowed agents in this milestone: {allowed}.",
                    payload=payload,
                ) from error
        return tuple(agents)

    # ------------------------------------------------------------------ #
    # Consistency checks
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_execution_order(agents, execution_order, payload) -> None:
        if sorted(a.value for a in agents) != sorted(a.value for a in execution_order):
            raise PlannerValidationError(
                "'execution_order' must contain exactly the same agents as 'agents'.",
                payload=payload,
            )
        if len(set(agents)) != len(agents):
            raise PlannerValidationError("'agents' must not contain duplicates.", payload=payload)

    @staticmethod
    def _check_clarification(requires_clarification, clarification_question, agents, payload) -> None:
        if requires_clarification:
            if agents:
                raise PlannerValidationError(
                    "When requires_clarification is true, 'agents' must be empty.", payload=payload
                )
            if not (clarification_question and clarification_question.strip()):
                raise PlannerValidationError(
                    "When requires_clarification is true, a clarification_question is required.",
                    payload=payload,
                )
        else:
            if clarification_question is not None:
                raise PlannerValidationError(
                    "When requires_clarification is false, clarification_question must be null.",
                    payload=payload,
                )
            if not agents:
                raise PlannerValidationError(
                    "When requires_clarification is false, at least one agent must be selected.",
                    payload=payload,
                )

    @staticmethod
    def _check_confidence(confidence, requires_clarification, payload) -> None:
        if confidence is ConfidenceLevel.LOW and not requires_clarification:
            raise PlannerValidationError(
                "LOW confidence must set requires_clarification to true.", payload=payload
            )
        if confidence is not ConfidenceLevel.LOW and requires_clarification:
            raise PlannerValidationError(
                "requires_clarification true must use LOW confidence.", payload=payload
            )

    @staticmethod
    def _check_intent(intent, requires_clarification, agents, payload) -> None:
        if requires_clarification:
            return  # agents are empty; nothing further to check against agents.

        if intent is IntentType.MIXED_QUERY:
            if len(agents) < 2:
                raise PlannerValidationError(
                    "'mixed_query' must select two or more agents.", payload=payload
                )
            return

        expected = _SINGLE_INTENT_AGENT.get(intent)
        if expected is None:
            # general_chat / out_of_scope -> General LLM, which is out of scope.
            raise PlannerValidationError(
                f"Intent '{intent.value}' routes to the General LLM, which is not available "
                "in this milestone.",
                payload=payload,
            )
        if agents != (expected,):
            raise PlannerValidationError(
                f"Intent '{intent.value}' must select exactly ['{expected.value}'], "
                f"got {[a.value for a in agents]}.",
                payload=payload,
            )
