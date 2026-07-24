"""PipelineService: the ONLY orchestration glue for the production AI pipeline.

This service invokes the already-existing, production AI components in order. It
contains NO AI logic, NO planner logic, and NO agent logic. It only:

    session (Conversation Memory)
      -> Supervisor Planner
      -> Supervisor Orchestrator (runs specialist agents + Response Aggregator)
      -> Recommendation Decision (+ optional Recommendation Generator)
      -> Response Generator
      -> FinalResponse text

All heavy pipeline imports and singletons are built lazily on first construction
so that importing this module (e.g. under test) stays cheap and does not require
API keys or model files. Real production classes are reused as-is; nothing is
re-implemented or duplicated.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

from dotenv import load_dotenv

from backend.config import OUT_OF_DOMAIN_MESSAGE, PROJECT_ROOT, bootstrap_pipeline_path

logger = logging.getLogger("aquamind.backend.pipeline")

_TIMEOUT_MARKERS = ("timeout", "timed out", "connection", "temporarily")
_TIMEOUT_EXCEPTIONS = {"APITimeoutError", "APIConnectionError"}
_VALIDATION_EXCEPTIONS = {"EmptyQueryError"}

#: Terminal Data Agent failures stop the pipeline before answer generation.
#: The set of terminal exception classes is built once in
#: ``PipelineService.__init__`` from the real classes (single source of truth):
#: ``EmptyLlmResponseError`` and ``LlmApiError`` (SQL Generator) and
#: ``RowLimitExceededError`` (SQLite Executor). All of them mean there is NO
#: valid structured data for downstream agents. The execution engine (which is
#: not modified here) records a failed agent as ``"<ClassName>: <message>"``, so
#: the live exception object is unavailable downstream; detection therefore
#: matches the recorded class name against that derived set.


# --------------------------------------------------------------------------- #
# Backend-level exceptions (translated from upstream failures for the API layer)
# --------------------------------------------------------------------------- #

class PipelineError(Exception):
    """A pipeline/LLM/agent failure that should surface as an upstream error."""


class PipelineTimeoutError(PipelineError):
    """The pipeline timed out talking to an upstream dependency."""


class PipelineValidationError(PipelineError):
    """The request was structurally invalid for the pipeline (e.g. empty query)."""


def _classify(error: Exception) -> PipelineError:
    """Translate any upstream exception into a safe backend exception."""
    name = type(error).__name__
    message = str(error)
    if name in _VALIDATION_EXCEPTIONS:
        return PipelineValidationError(message)
    if name in _TIMEOUT_EXCEPTIONS or any(m in message.lower() for m in _TIMEOUT_MARKERS):
        return PipelineTimeoutError(message)
    return PipelineError(message)


def _terminal_data_agent_presentation(cause: str) -> tuple[str, str, bool]:
    """Return ``(log_summary, user_message, recoverable)`` for a terminal cause.

    ``recoverable`` selects how the failure surfaces to the frontend:
      * True  -> ``PipelineValidationError`` (HTTP 400, ``user_message`` shown);
                 a user-fixable request (the query asked for too much raw data).
      * False -> ``PipelineError`` (HTTP 502, generic message); a provider/
                 transient failure the user cannot fix by rewording.
    """
    if cause == "RowLimitExceededError":
        return (
            "Pipeline stopped because Data Agent returned an oversized dataset.",
            "The requested query would return too much raw telemetry data. Please "
            "narrow your request or ask for a summary (average, latest, monthly "
            "trend, or station-wise results).",
            True,
        )
    return (
        "Pipeline stopped because Data Agent failed.",
        "The Data Agent could not retrieve valid data for this question "
        f"({cause}); the request was stopped before answer generation.",
        False,
    )


# --------------------------------------------------------------------------- #
# Specialist-agent adapters (thin SpecialistAgent glue over real components)
# --------------------------------------------------------------------------- #

class _ResponseFactory(Protocol):
    def __call__(self, agent_name: str, payload: dict[str, Any], execution_time: float): ...


class _DataAgentAdapter:
    """Real Data Agent: SqlGenerator -> SQLiteExecutor -> EvidenceFormatter."""

    AGENT_NAME = "data_agent"

    def __init__(self, generator: Any, executor: Any, formatter: Any, response_cls: Any) -> None:
        self._generator = generator
        self._executor = executor
        self._formatter = formatter
        self._response_cls = response_cls

    def execute(self, request: Any) -> Any:
        from time import perf_counter

        start = perf_counter()
        sql = self._generator.generate(request.user_query)
        rows = self._executor.execute(sql)
        evidence = self._formatter.format(rows)
        payload = {
            "agent_name": self.AGENT_NAME,
            "status": "SUCCESS" if evidence else "NO_RESULTS",
            "query_type": "data",
            "sql": sql,
            "row_count": len(evidence),
            "evidence": evidence,
        }
        return self._response_cls(self.AGENT_NAME, payload, perf_counter() - start)


class _KnowledgeAgentAdapter:
    """Real Knowledge Agent: RetrievalCoordinator (FAISS) -> KnowledgeFormatter."""

    AGENT_NAME = "knowledge_agent"

    def __init__(self, coordinator: Any, formatter: Any, response_cls: Any) -> None:
        self._coordinator = coordinator
        self._formatter = formatter
        self._response_cls = response_cls

    def execute(self, request: Any) -> Any:
        from time import perf_counter

        start = perf_counter()
        chunks = self._coordinator.retrieve(request.user_query)
        payload = self._formatter.format(chunks)
        return self._response_cls(self.AGENT_NAME, payload, perf_counter() - start)


class _PredictionAgentAdapter:
    """Real Prediction Agent (trained model) driven by AgentRequest.metadata.

    The Prediction Agent consumes STRUCTURED slots only (district / firka /
    prediction_year); it performs no natural-language understanding. The
    Supervisor Planner extracts these from the user query and the orchestrator
    forwards them here via ``request.metadata``.

    The trained model predicts at DISTRICT granularity, so a Firka request is
    resolved to its parent district through a deterministic lookup over
    ``master_firka.csv`` (a structured lookup, NOT NLU).

    Missing or unresolvable inputs produce a NO_PREDICTION response carrying a
    user-facing message -- never an exception -- so the pipeline degrades
    gracefully instead of surfacing an agent error.
    """

    AGENT_NAME = "prediction_agent"

    def __init__(self, runtime: Any, response_cls: Any) -> None:
        self._runtime = runtime
        self._response_cls = response_cls
        self._firka_to_district: dict[str, str] | None = None  # lazily loaded

    def execute(self, request: Any) -> Any:
        from time import perf_counter

        start = perf_counter()
        metadata = request.metadata or {}
        district = metadata.get("district")
        firka = metadata.get("firka")
        year = metadata.get("prediction_year")

        # Graceful validation (never raise): a NO_PREDICTION status is a
        # successful, non-error outcome that the Response Generator conveys.
        if not district and not firka:
            return self._no_prediction(
                "Please specify a district or Firka name to forecast the groundwater level.",
                perf_counter() - start,
            )
        if year is None:
            return self._no_prediction(
                "Please specify the prediction year.", perf_counter() - start
            )

        resolved_district = str(district).strip() if district else None
        resolved_firka = str(firka).strip() if firka else None
        if not resolved_district and resolved_firka:
            resolved_district = self._resolve_firka_to_district(resolved_firka)
            if not resolved_district:
                return self._no_prediction(
                    f"The requested Firka '{resolved_firka}' is not available in the "
                    "groundwater prediction dataset. Please provide a valid Firka or "
                    "District name.",
                    perf_counter() - start,
                )

        month = metadata.get("prediction_month")
        query = {
            "district": str(resolved_district),
            "year": int(year),
            "month": int(month) if month else 1,
        }
        outcome = self._runtime.run(query)
        return self._response_cls(self.AGENT_NAME, dict(outcome["response"]), perf_counter() - start)

    # -- graceful, structured helpers ------------------------------------- #

    def _no_prediction(self, message: str, execution_time: float) -> Any:
        """Build a NO_PREDICTION response (non-error) with a user-facing message."""
        payload = {
            "agent_name": self.AGENT_NAME,
            "status": "NO_PREDICTION",
            "query_type": "prediction",
            "message": message,
        }
        return self._response_cls(self.AGENT_NAME, payload, execution_time)

    def _resolve_firka_to_district(self, firka: str) -> str | None:
        """Map a Firka name to its parent district via a deterministic lookup.

        Returns the district that contains ``firka``, or None if the Firka is not
        present in ``master_firka.csv``.
        """
        if self._firka_to_district is None:
            self._firka_to_district = self._load_firka_district_map()
        return self._firka_to_district.get(firka.strip().upper())

    @staticmethod
    def _load_firka_district_map() -> dict[str, str]:
        """Build a normalized {FIRKA -> district} map from master_firka.csv (once)."""
        import pandas as pd
        import prediction_config as config

        path = config.MASTER_DATASETS_DIR / "master_firka.csv"
        mapping: dict[str, str] = {}
        try:
            frame = pd.read_csv(path, usecols=["district", "firka"], dtype=str)
        except Exception as error:  # noqa: BLE001 - resolution is best-effort
            logger.warning(
                "Could not load firka->district map (%s): %s", type(error).__name__, error
            )
            return mapping
        for district, firka in zip(frame["district"], frame["firka"]):
            if not isinstance(firka, str) or not isinstance(district, str):
                continue
            key = firka.strip().upper()
            if key and key not in mapping:
                mapping[key] = district.strip()
        return mapping


# --------------------------------------------------------------------------- #
# PipelineService
# --------------------------------------------------------------------------- #

class PipelineService:
    """Builds the real pipeline once and runs a message through it."""

    def __init__(self) -> None:
        bootstrap_pipeline_path()

        # Lazy imports: keep module import cheap; only touch the pipeline when a
        # real service is actually constructed (production / warmup).
        from orchestrator import SupervisorOrchestrator, UniversalAgentResponse
        from planner.planner import Planner
        from memory import ConversationMemory
        from sql_generator import SqlGenerator, EmptyLlmResponseError, LlmApiError
        from sqlite_executor import SQLiteExecutor, RowLimitExceededError
        from evidence_formatter import EvidenceFormatter
        from retrieval_coordinator import RetrievalCoordinator
        from knowledge_formatter import KnowledgeFormatter
        from end_to_end_pipeline_test import PredictionAgentRuntime
        from recommendation_agent import (
            RecommendationDecider,
            RecommendationGenerator,
        )
        from response_generator import (
            RESPONSE_SYSTEM_PROMPT_PATH,
            ResponseGenerator,
            load_prompt,
        )

        logger.info("Initializing AquaMind AI pipeline singletons…")

        # --- Production LLM configuration (confirmed deployment decision) ----- #
        # The production environment is .env.testing. Every LLM call site is
        # wired to the providers/models/keys defined there. ONLY the API layer
        # (provider, model, base URL, key) differs from the historical Gemini
        # setup; agent logic, prompts, the orchestrator, the parser and the
        # strict validation are all unchanged. Production retry defaults are kept
        # (unlike the deployment-test harness), and the Response Generator keeps
        # the 8192 output-token budget from the truncation fix.
        #   Planner                  -> OpenCode Zen (DeepSeek Flash 4)
        #   SQL Generator            -> OpenCode Zen (DeepSeek Flash 4)
        #   Recommendation Decision  -> OpenCode Zen (DeepSeek Flash 4)
        #   Recommendation Generator -> OpenCode Zen (DeepSeek Flash 4)
        #   Response Generator       -> OpenCode Zen (DeepSeek Flash 4)
        from recommendation_agent.config import (
            ConfigurationError,
            LlmClient,
            OPENCODE_BASE_URL,
            MODEL_NAME as DEEPSEEK_MODEL,
        )

        load_dotenv(PROJECT_ROOT / ".env.testing", override=True)

        def _key(var: str) -> str:
            value = os.environ.get(var, "").strip()
            if not value:
                raise ConfigurationError(f"{var} is not set in .env.testing.")
            return value

        response_max_output_tokens = 8192

        # Select the production provider endpoint for the Planner and SQL
        # Generator. Their OpenAI-client base URL is a module-level global read
        # at construction time; overriding it here selects the production
        # provider WITHOUT modifying the agent modules. This is environment /
        # provider selection only -- no agent, routing or pipeline logic changes.
        import planner.planner as planner_module
        import sql_generator as sql_module

        planner_module.GEMINI_BASE_URL = OPENCODE_BASE_URL
        sql_module.GEMINI_BASE_URL = OPENCODE_BASE_URL

        self._memory = ConversationMemory()
        self._planner = Planner(
            api_key=_key("PLANNER_TESTING_API_KEY"),
            model_name=DEEPSEEK_MODEL,
        )

        sql_generator = SqlGenerator(
            api_key=_key("SQL_GENERATOR_TESTING_API_KEY"),
            model_name=DEEPSEEK_MODEL,
        )
        data_agent = _DataAgentAdapter(
            sql_generator, SQLiteExecutor(), EvidenceFormatter(), UniversalAgentResponse
        )
        knowledge_agent = _KnowledgeAgentAdapter(
            RetrievalCoordinator(), KnowledgeFormatter(), UniversalAgentResponse
        )
        prediction_agent = _PredictionAgentAdapter(PredictionAgentRuntime(), UniversalAgentResponse)

        self._orchestrator = SupervisorOrchestrator(
            agents={
                data_agent.AGENT_NAME: data_agent,
                knowledge_agent.AGENT_NAME: knowledge_agent,
                prediction_agent.AGENT_NAME: prediction_agent,
            }
        )

        # Injected provider clients (production retry/backoff defaults preserved).
        self._decider = RecommendationDecider(
            client=LlmClient(
                api_key=_key("RECOMMENDATION_DECISION_TESTING_API_KEY"),
                model_name=DEEPSEEK_MODEL,
                base_url=OPENCODE_BASE_URL,
                # Constrain output to valid JSON at the provider. The decision's
                # free-text "reason" occasionally contained unescaped quotes,
                # producing malformed JSON (ResponseParsingError: Expecting ','
                # delimiter). JSON mode guarantees syntactically valid, correctly
                # escaped JSON. The prompt and parser are unchanged.
                response_format={"type": "json_object"},
            )
        )
        self._recommendation_generator = RecommendationGenerator(
            client=LlmClient(
                api_key=_key("RECOMMENDATION_GENERATOR_TESTING_API_KEY"),
                model_name=DEEPSEEK_MODEL,
                base_url=OPENCODE_BASE_URL,
            )
        )
        self._response_generator = ResponseGenerator(
            client=LlmClient(
                api_key=_key("RESPONSE_GENERATOR_TESTING_API_KEY"),
                model_name=DEEPSEEK_MODEL,
                base_url=OPENCODE_BASE_URL,
                max_output_tokens=response_max_output_tokens,
            )
        )
        self._system_prompt = load_prompt(RESPONSE_SYSTEM_PROMPT_PATH)

        # Terminal Data Agent failures (single source of truth: the real classes).
        # Any of these means there is no valid structured data for downstream
        # agents, so the pipeline must stop before the Recommendation layer and
        # Response Generator. Extend by adding a class here.
        self._terminal_data_agent_errors: tuple[type[BaseException], ...] = (
            EmptyLlmResponseError,
            LlmApiError,
            RowLimitExceededError,
        )
        self._terminal_data_agent_error_names = frozenset(
            exc.__name__ for exc in self._terminal_data_agent_errors
        )

        logger.info("Pipeline ready (production configuration: .env.testing).")

    # -- session helpers -------------------------------------------------- #

    def _ensure_session(self, session_id: str) -> None:
        if not self._memory.session_exists(session_id):
            self._memory.create_session(session_id)

    # -- prompt assembly (the orchestrator's contract; pure string glue) --- #

    def _assemble_response_prompt(
        self, user_query: str, aggregated: dict[str, Any], recommendation: dict[str, Any] | None
    ) -> str:
        sanitized: dict[str, Any] = {
            "status": aggregated.get("status"),
            "responses": aggregated.get("responses", []),
        }
        if recommendation is not None:
            sanitized["recommendation_response"] = recommendation
        aggregate_json = json.dumps(sanitized, indent=2, ensure_ascii=False)
        return (
            f"{self._system_prompt}\n\n"
            "=========================================================\n"
            "USER QUERY\n"
            "=========================================================\n"
            f"{user_query}\n\n"
            "=========================================================\n"
            "UPDATED AGGREGATED UNIVERSAL AGENT RESPONSE\n"
            "=========================================================\n"
            f"{aggregate_json}\n"
        )

    # -- Data Agent failure guard ----------------------------------------- #

    def _is_terminal_data_agent_error(self, error_text: str | None) -> str | None:
        """Return the terminal Data Agent exception class name in ``error_text``.

        A terminal Data Agent failure -- an empty LLM response
        (``EmptyLlmResponseError``), an LLM API failure (``LlmApiError``), or an
        oversized-result refusal (``RowLimitExceededError``) -- means there is no
        valid structured data for downstream agents.

        The execution engine (not modified here) records a failed agent as
        ``"<ExceptionClassName>: <message>"``, so the live exception object is
        unavailable at this point and a true ``isinstance`` check is not possible.
        Instead we compare the recorded class name against
        ``self._terminal_data_agent_error_names`` (derived from the real terminal
        exception classes). Returns the matched class name, or ``None`` when the
        failure is not terminal.
        """
        if not error_text:
            return None
        recorded_type = error_text.split(":", 1)[0].strip()
        if recorded_type in self._terminal_data_agent_error_names:
            return recorded_type
        return None

    def _halt_if_data_agent_failed(self, aggregated: Any) -> None:
        """Stop the pipeline before answer generation on a terminal Data Agent failure.

        The Supervisor's execution engine records a failing specialist agent as
        an ERROR response and continues. For a terminal Data Agent failure
        (empty response, LLM API error, or an oversized-result refusal), running
        the Recommendation layer and the Response Generator on absent/invalid
        evidence only produces a downstream ``FinalResponseValidationError``. We
        detect that cause here and stop immediately -- the Recommendation Agent
        and Response Generator are never invoked.
        """
        for response in aggregated.responses:
            if getattr(response, "agent_name", None) != "data_agent" or response.succeeded():
                continue
            cause = self._is_terminal_data_agent_error(response.error)
            if cause is None:
                continue
            summary, user_message, recoverable = _terminal_data_agent_presentation(cause)
            # Clear, cause-specific log (agent, exception type, message) instead
            # of a generic "Pipeline execution failed".
            logger.error(
                "%s Cause: %s. agent=%s exception_type=%s message=%s",
                summary, cause, response.agent_name, cause, response.error or "",
            )
            error_cls = PipelineValidationError if recoverable else PipelineError
            raise error_cls(user_message)

    # -- recommendation layer (optional, best-effort) --------------------- #

    def _maybe_generate_recommendation(self, aggregated: Any) -> dict[str, Any] | None:
        """Run the Recommendation Decision and (if required) Generator.

        Recommendations are an optional enhancement: any failure here is logged
        and swallowed so the user still receives a grounded answer.
        """
        try:
            decision = self._decider.decide(aggregated)
            if not decision.recommendation_required:
                return None
            recommendation = self._recommendation_generator.generate(aggregated, decision)
            return recommendation.to_dict()
        except Exception as error:  # noqa: BLE001 - recommendations must never break the answer
            logger.warning(
                "Recommendation layer skipped (%s): %s", type(error).__name__, error
            )
            return None

    # -- public entry point ----------------------------------------------- #

    def process(self, message: str, session_id: str, metadata: dict[str, Any] | None = None) -> str:
        """Run one user message through the full pipeline and return the answer.

        Raises a :class:`PipelineError` subclass on failure (already sanitized).
        """
        request_metadata = metadata or {}
        try:
            self._ensure_session(session_id)
            self._memory.increment_turn(session_id)
            self._memory.add_user_message(session_id, message)
            snapshot = self._memory.get_session(session_id).to_dict()

            decision = self._planner.plan(message, conversation_memory=snapshot)

            # Out-of-domain short-circuit: the query is not about groundwater, so
            # the entire execution pipeline is skipped. No specialist agent, no
            # SQL, no FAISS, no prediction, no recommendation, no Response
            # Generator, and no LLM run -- we return the predefined message.
            if getattr(decision, "is_out_of_domain", False):
                self._memory.add_assistant_message(session_id, OUT_OF_DOMAIN_MESSAGE)
                return OUT_OF_DOMAIN_MESSAGE

            # Clarification short-circuit: the planner asks the user for more
            # detail and no specialist agent runs (mirrors the orchestrator).
            if getattr(decision, "requires_clarification", False):
                clarification = (
                    getattr(decision, "clarification_question", None)
                    or "Could you please provide a bit more detail about your groundwater question?"
                )
                self._memory.add_assistant_message(session_id, clarification)
                return clarification

            aggregated = self._orchestrator.orchestrate(
                decision,
                message,
                session_id=session_id,
                conversation_context=snapshot,
                metadata=request_metadata,
            )

            # Fail fast on a terminal Data Agent failure (empty/failed SQL
            # generation, or an oversized-result refusal): do NOT run the
            # Recommendation layer or the Response Generator with invalid/absent
            # evidence. Raises a PipelineError subclass surfaced to the frontend.
            self._halt_if_data_agent_failed(aggregated)

            recommendation = self._maybe_generate_recommendation(aggregated)

            prompt = self._assemble_response_prompt(
                message, aggregated.to_dict(), recommendation
            )
            final = self._response_generator.generate(prompt)

            answer = final.response.strip()
            self._memory.add_assistant_message(session_id, answer)
            return answer
        except PipelineError:
            raise
        except Exception as error:  # noqa: BLE001 - centralize translation to safe errors
            logger.exception("Pipeline execution failed: %s", type(error).__name__)
            raise _classify(error) from error
