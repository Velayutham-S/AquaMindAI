"""The single public chat endpoint: POST /api/chat.

Receives a user message, invokes the complete production pipeline via
:class:`PipelineService`, and returns the Response Generator's answer. It adds
no business logic of its own.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response

from backend.api.models import ChatRequest, ChatResponse
from backend.config import get_settings
from backend.dependencies import get_pipeline_service, get_session_id
from backend.services.pipeline_service import PipelineService

logger = logging.getLogger("aquamind.backend.chat")

router = APIRouter(prefix="/api", tags=["chat"])

_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    response: Response,
    session_id: str = Depends(get_session_id),
    service: PipelineService = Depends(get_pipeline_service),
) -> ChatResponse:
    """Run one user message through the AquaMind AI pipeline."""
    answer = service.process(payload.message, session_id=session_id)

    # Persist the conversation session so follow-up turns share memory.
    response.set_cookie(
        key=get_settings().session_cookie_name,
        value=session_id,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return ChatResponse(response=answer)
