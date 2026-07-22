"""Typed output model for the AquaMind AI Response Generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class FinalResponseValidationError(Exception):
    """The Response Generator returned an invalid FinalResponse object."""


@dataclass(frozen=True)
class FinalResponse:
    """The final user-facing response produced from supplied evidence only."""

    status: str
    response: str

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "response": self.response}

    @classmethod
    def parse(cls, payload: Any) -> "FinalResponse":
        """Validate raw JSON and return a strict ``FinalResponse``.

        Extra fields are rejected so reasoning, internal JSON, or implementation
        details cannot leak through the public response contract.
        """
        if not isinstance(payload, dict):
            raise FinalResponseValidationError(
                f"FinalResponse must be a JSON object, got {type(payload).__name__}."
            )
        allowed = {"status", "response"}
        extra = set(payload) - allowed
        if extra:
            raise FinalResponseValidationError(
                f"FinalResponse contains unexpected fields: {sorted(extra)}."
            )
        if payload.get("status") != "SUCCESS":
            raise FinalResponseValidationError("'status' must be exactly 'SUCCESS'.")
        response = payload.get("response")
        if not isinstance(response, str) or not response.strip():
            raise FinalResponseValidationError("'response' must be a non-empty string.")
        return cls(status="SUCCESS", response=response.strip())
