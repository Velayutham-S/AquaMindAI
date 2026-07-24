"""Backend configuration and pipeline import bootstrap.

This module owns two concerns:

1. :class:`Settings` -- environment-driven configuration (host, port, log level,
   CORS origins, session cookie, warmup).
2. :func:`bootstrap_pipeline_path` -- makes the existing AI pipeline packages
   importable "by location" WITHOUT modifying any of them. The production
   pipeline is a set of sibling packages under ``agents/`` that are imported by
   directory (the same convention the validated end-to-end harness uses); this
   function replicates exactly that ``sys.path`` setup.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BACKEND_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BACKEND_DIR.parent
AGENTS_DIR: Path = PROJECT_ROOT / "agents"

# Predefined reply for out-of-domain (non-groundwater) queries. When the Planner
# classifies a query as out of domain, the pipeline returns this message
# directly and runs NO specialist agent and NO LLM. Kept here so the production
# service and the deployment-test runner share one source of truth.
OUT_OF_DOMAIN_MESSAGE: str = (
    "I'm AquaMind AI, an Agentic RAG-based Groundwater Intelligence and Decision "
    "Support System. I'm designed specifically to help with groundwater-related "
    "information and analysis for Tamil Nadu.\n\n"
    "I can help with topics such as:\n"
    "- Groundwater levels and availability\n"
    "- Groundwater quality\n"
    "- Groundwater extraction statistics\n"
    "- Rainfall and groundwater relationships\n"
    "- Aquifer information\n"
    "- Artificial recharge methods\n"
    "- Groundwater sustainability\n"
    "- Groundwater predictions\n"
    "- Water conservation recommendations\n"
    "- District and Firka-level groundwater analysis\n\n"
    "Please ask a groundwater-related question, and I'll be happy to help."
)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Immutable, environment-driven backend settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    session_cookie_name: str = "aquamind_session"
    warmup_on_startup: bool = True

    @staticmethod
    def from_env() -> "Settings":
        default_origins = "http://localhost:5173,http://127.0.0.1:5173"
        origins = _split_csv(os.getenv("CORS_ORIGINS", default_origins))
        try:
            port = int(os.getenv("PORT", "8000"))
        except ValueError:
            port = 8000
        warmup = os.getenv("WARMUP_ON_STARTUP", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        return Settings(
            host=os.getenv("HOST", "0.0.0.0"),
            port=port,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            cors_origins=origins or ["http://localhost:5173"],
            session_cookie_name=os.getenv("SESSION_COOKIE_NAME", "aquamind_session"),
            warmup_on_startup=warmup,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, process-wide settings instance."""
    return Settings.from_env()


def pipeline_import_dirs() -> list[Path]:
    """Directories that must be on ``sys.path`` to import the pipeline packages.

    Mirrors the import-by-location layout used by the existing production
    pipeline. Only the prediction agent's ``tests`` dir is added (it hosts the
    reusable ``PredictionAgentRuntime``); the data agent's ``tests`` dir is
    intentionally excluded to avoid a duplicate ``end_to_end_pipeline_test``
    module name clash.
    """
    data_agent = AGENTS_DIR / "data_agent"
    knowledge_agent = AGENTS_DIR / "knowledge_agent"
    prediction_agent = AGENTS_DIR / "prediction_agent"
    return [
        AGENTS_DIR,  # recommendation_agent, response_generator (packages)
        AGENTS_DIR / "supervisor_agent",  # orchestrator, planner, memory packages
        data_agent / "llm",  # sql_generator
        data_agent / "database",  # sqlite_executor
        data_agent / "formatter",  # evidence_formatter
        knowledge_agent,  # knowledge_config
        knowledge_agent / "retrieval",  # retrieval_coordinator
        knowledge_agent / "formatter",  # knowledge_formatter
        prediction_agent,  # prediction_config
        prediction_agent / "training",  # dataset_integrator, feature_engineering, model_registry
        prediction_agent / "formatter",  # prediction_formatter
        prediction_agent / "tests",  # PredictionAgentRuntime (end_to_end_pipeline_test)
    ]


def bootstrap_pipeline_path() -> None:
    """Insert the pipeline import directories onto ``sys.path`` (idempotent)."""
    for directory in pipeline_import_dirs():
        path_str = str(directory)
        if directory.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)
