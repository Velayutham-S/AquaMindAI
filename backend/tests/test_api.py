"""API-layer tests for the AquaMind AI backend.

These test the integration layer in isolation: the real AI pipeline is replaced
via FastAPI dependency overrides, so no API keys, models, or network are needed.
Covered: health, chat success, pipeline failure, validation failure, LLM/timeout
failure, and unexpected errors.
"""

from __future__ import annotations

import os

# Disable startup warmup BEFORE importing the app so lifespan never builds the
# real (heavy) pipeline during tests.
os.environ["WARMUP_ON_STARTUP"] = "false"

import pytest
from fastapi.testclient import TestClient

from backend.dependencies import get_pipeline_service
from backend.main import app
from backend.services.pipeline_service import PipelineError, PipelineTimeoutError


# --------------------------------------------------------------------------- #
# Fake pipeline services (stand-ins for the real PipelineService)
# --------------------------------------------------------------------------- #

class _SuccessService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def process(self, message: str, session_id: str, metadata=None) -> str:
        self.calls.append((message, session_id))
        return "The groundwater level in Salem is 7.8 metres below ground level."


class _PipelineFailureService:
    def process(self, message: str, session_id: str, metadata=None) -> str:
        raise PipelineError("upstream boom with secret-key=SHOULD_NOT_LEAK")


class _TimeoutService:
    def process(self, message: str, session_id: str, metadata=None) -> str:
        raise PipelineTimeoutError("LLM request timed out")


class _UnexpectedService:
    def process(self, message: str, session_id: str, metadata=None) -> str:
        raise RuntimeError("unexpected internal failure with token=SECRET")


@pytest.fixture
def client():
    # Do not raise server exceptions so the 500 handler response is asserted.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _use(service) -> None:
    app.dependency_overrides[get_pipeline_service] = lambda: service


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# --------------------------------------------------------------------------- #
# Chat: success
# --------------------------------------------------------------------------- #

def test_chat_success(client):
    service = _SuccessService()
    _use(service)

    response = client.post("/api/chat", json={"message": "What is the groundwater level in Salem?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert "7.8 metres below ground level" in body["response"]
    # Session cookie is set for multi-turn continuity.
    assert "aquamind_session" in response.cookies
    # The message reached the pipeline.
    assert service.calls and service.calls[0][0].startswith("What is the groundwater level")


def test_chat_reuses_session_cookie(client):
    _use(_SuccessService())
    first = client.post("/api/chat", json={"message": "hello"})
    session = first.cookies.get("aquamind_session")
    assert session
    # TestClient persists cookies; a follow-up should carry the same session.
    second = client.post("/api/chat", json={"message": "again"})
    assert second.status_code == 200


# --------------------------------------------------------------------------- #
# Chat: validation failures
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "   "}, {"message": 123}])
def test_chat_validation_failure(client, payload):
    _use(_SuccessService())
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "ERROR"
    assert "message" in body


# --------------------------------------------------------------------------- #
# Chat: pipeline / LLM / timeout / unexpected failures
# --------------------------------------------------------------------------- #

def test_chat_pipeline_failure(client):
    _use(_PipelineFailureService())
    response = client.post("/api/chat", json={"message": "trigger failure"})
    assert response.status_code == 502
    body = response.json()
    assert body["status"] == "ERROR"
    # No internal detail / secret leaks to the client.
    assert "secret-key" not in body["message"]
    assert "SHOULD_NOT_LEAK" not in body["message"]


def test_chat_timeout_failure(client):
    _use(_TimeoutService())
    response = client.post("/api/chat", json={"message": "slow request"})
    assert response.status_code == 504
    body = response.json()
    assert body["status"] == "ERROR"


def test_chat_unexpected_error(client):
    _use(_UnexpectedService())
    response = client.post("/api/chat", json={"message": "boom"})
    assert response.status_code == 500
    body = response.json()
    assert body["status"] == "ERROR"
    assert "SECRET" not in body["message"]
