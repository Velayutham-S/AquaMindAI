"""Production Response Generator for AquaMind AI.

This is the ONLY component that creates the final user-facing natural-language
response. Its public API deliberately accepts one complete prompt string --
already assembled upstream in deterministic order by the Supervisor
Orchestrator -- and nothing else.

The component does not load the system prompt, inspect the user query, assemble
inputs, retrieve evidence, call specialist agents, or mutate evidence. It sends
the supplied prompt to the deterministic LLM and validates the strict
``FinalResponse`` JSON contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import build_response_client, extract_json_object
from .response_models import FinalResponse


@runtime_checkable
class CompletionClient(Protocol):
    """Minimal dependency-injection contract used by ResponseGenerator."""

    def complete(self, prompt: str) -> str: ...


class ResponseGenerator:
    """Generate a grounded FinalResponse from one preassembled prompt."""

    def __init__(self, client: CompletionClient | None = None) -> None:
        self._client = client or build_response_client()

    def generate(self, complete_prompt: str) -> FinalResponse:
        """Generate and validate the final response.

        ``complete_prompt`` must already contain, in order, the response system
        prompt, user query, and sanitized updated aggregate (including optional
        recommendations). No other input is accepted or inferred here.
        """
        if not isinstance(complete_prompt, str) or not complete_prompt.strip():
            raise ValueError("complete_prompt must be a non-empty string.")
        raw = self._client.complete(complete_prompt)
        return FinalResponse.parse(extract_json_object(raw))
