"""AquaMind AI backend application factory.

Exposes exactly two public endpoints:
    GET  /health     -> {"status": "healthy"}
    POST /api/chat   -> {"status": "SUCCESS", "response": "..."}

Run locally:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import chat, health
from backend.config import get_settings
from backend.dependencies import get_pipeline_service
from backend.middleware.exception_handler import register_exception_handlers
from backend.middleware.logging import RequestLoggingMiddleware, configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("aquamind.backend")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the pipeline once at startup (best-effort) to avoid a slow first call."""
    if settings.warmup_on_startup:
        try:
            get_pipeline_service()
            logger.info("Pipeline warmed up on startup.")
        except Exception as error:  # noqa: BLE001 - never block startup on warmup
            logger.warning(
                "Pipeline warmup skipped (%s). It will initialize on first request.",
                type(error).__name__,
            )
    yield


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title="AquaMind AI",
        description="Backend integration layer for the AquaMind AI groundwater pipeline.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
