"""Pydantic request/response models for the public API.

These define the ONLY contract the frontend depends on. Internal pipeline
objects are never serialized to the client.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Body of POST /api/chat."""

    message: str = Field(..., min_length=1, max_length=4000, description="User question.")

    @field_validator("message")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be blank")
        return cleaned


class ChatResponse(BaseModel):
    """Successful chat response returned to the frontend."""

    status: Literal["SUCCESS"] = "SUCCESS"
    response: str


class ErrorResponse(BaseModel):
    """Uniform error envelope returned to the frontend."""

    status: Literal["ERROR"] = "ERROR"
    message: str


class HealthResponse(BaseModel):
    """Body of GET /health."""

    status: Literal["healthy"] = "healthy"
