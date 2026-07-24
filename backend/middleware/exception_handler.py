"""Centralized exception handling.

Maps validation, pipeline, timeout, and unexpected errors to appropriate HTTP
status codes and a uniform ``{"status": "ERROR", "message": "..."}`` body.
Stack traces are logged server-side and never returned to the client.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.services.pipeline_service import (
    PipelineError,
    PipelineTimeoutError,
    PipelineValidationError,
)

logger = logging.getLogger("aquamind.backend.errors")


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "ERROR", "message": message})


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the app."""

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info("Request validation failed: %s", exc.errors())
        # 422 literal (version-agnostic: the named constant is deprecated in
        # newer Starlette releases).
        return _error(422, "Invalid request. Please provide a non-empty 'message' field.")

    @app.exception_handler(PipelineValidationError)
    async def _handle_pipeline_validation(_: Request, exc: PipelineValidationError) -> JSONResponse:
        logger.info("Pipeline validation error: %s", exc)
        return _error(status.HTTP_400_BAD_REQUEST, str(exc) or "The request could not be processed.")

    @app.exception_handler(PipelineTimeoutError)
    async def _handle_timeout(_: Request, __: PipelineTimeoutError) -> JSONResponse:
        logger.warning("Pipeline timeout.")
        return _error(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "AquaMind AI took too long to respond. Please try again.",
        )

    @app.exception_handler(PipelineError)
    async def _handle_pipeline(_: Request, __: PipelineError) -> JSONResponse:
        logger.warning("Pipeline failure.")
        return _error(
            status.HTTP_502_BAD_GATEWAY,
            "AquaMind AI could not process your request right now. Please try again.",
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected error: %s", type(exc).__name__)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "An unexpected error occurred. Please try again later.",
        )
