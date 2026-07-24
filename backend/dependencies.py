"""Dependency-injection providers for the API layer.

Provides a process-wide singleton :class:`PipelineService` (built lazily and
reused across requests) and a cookie-based conversation session id. Tests
override :func:`get_pipeline_service` with a fake, so the real pipeline is never
constructed during unit tests.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from uuid import uuid4

from fastapi import Request

from backend.config import get_settings
from backend.services.pipeline_service import PipelineService

logger = logging.getLogger("aquamind.backend.deps")


@lru_cache(maxsize=1)
def get_pipeline_service() -> PipelineService:
    """Return the shared PipelineService, constructing it once on first use."""
    logger.info("Constructing PipelineService singleton.")
    return PipelineService()


def get_session_id(request: Request) -> str:
    """Resolve the conversation session id from the request cookie, or mint one.

    Multi-turn memory works when the client returns the session cookie; when it
    does not (e.g. a cross-origin client without credentials), each request gets
    a fresh session, which degrades cleanly to single-turn behavior.
    """
    cookie_name = get_settings().session_cookie_name
    existing = request.cookies.get(cookie_name)
    if existing:
        return existing
    return uuid4().hex
