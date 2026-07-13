"""Deterministic planning-prompt assembly for the AquaMind AI Supervisor Planner.

Single responsibility: load the permanent planner knowledge base (the five
``supervisor_llm_inputs`` documents) once, and assemble the complete planning
prompt for a given conversation-memory snapshot + user query, in this exact,
fixed order:

    1. Supervisor System Prompt
    2. Conversation Memory
    3. database_schema.md
    4. knowledge_agent_schema.md
    5. prediction_agent_schema.md
    6. routing_rules.md
    7. Current User Query

It performs NO LLM calls, NO reasoning, and NO agent execution. Given the same
inputs it always produces the same prompt (deterministic).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .planner_models import InputFileError

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PLANNER_DIR: Path = Path(__file__).resolve().parent
SUPERVISOR_DIR: Path = PLANNER_DIR.parent
DEFAULT_INPUTS_DIR: Path = SUPERVISOR_DIR / "supervisor_llm_inputs"

SYSTEM_PROMPT_FILE = "supervisor_system_prompt.txt"
DATABASE_SCHEMA_FILE = "database_schema.md"
KNOWLEDGE_SCHEMA_FILE = "knowledge_agent_schema.md"
PREDICTION_SCHEMA_FILE = "prediction_agent_schema.md"
ROUTING_RULES_FILE = "routing_rules.md"

_NO_MEMORY = "(no prior conversation context)"
_SECTION = "=" * 58


def _load_text_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise InputFileError(f"Required planner input file not found: {path}") from error
    except OSError as error:
        raise InputFileError(f"Could not read planner input file {path}: {error}") from error
    if not text.strip():
        raise InputFileError(f"Required planner input file is empty: {path}")
    return text


def render_memory_snapshot(snapshot: dict[str, Any] | None) -> str:
    """Render a Conversation Memory session snapshot into deterministic text.

    ``snapshot`` is a ``Session.to_dict()`` mapping (or ``None`` when there is no
    session). Only routing-relevant, populated fields are rendered, in a fixed
    order. This function never infers — it only formats what memory stored.
    """
    if not snapshot:
        return _NO_MEMORY

    lines: list[str] = []

    context = snapshot.get("context") or {}
    populated = {k: v for k, v in context.items() if v not in (None, "", {}, [])}
    if populated:
        lines.append("Current context:")
        for key in sorted(populated):
            lines.append(f"  {key}: {populated[key]}")

    entities = snapshot.get("entities") or {}
    if entities:
        lines.append(f"Referenced entities: {entities}")

    last_response = snapshot.get("last_response")
    if last_response:
        lines.append(f"Last response: {last_response}")

    turn = snapshot.get("turn")
    if turn:
        lines.append(f"Conversation turn: {turn}")

    history = snapshot.get("history") or []
    if history:
        lines.append("Recent messages:")
        for message in history[-6:]:
            role = message.get("role", "?")
            content = message.get("content", "")
            lines.append(f"  {role}: {content}")

    return "\n".join(lines) if lines else _NO_MEMORY


class PromptBuilder:
    """Loads the planner knowledge base once and assembles planning prompts."""

    def __init__(self, inputs_dir: Path | None = None) -> None:
        base = Path(inputs_dir) if inputs_dir else DEFAULT_INPUTS_DIR
        self._inputs_dir = base
        # Loaded once; the knowledge base is static across queries.
        self._system_prompt = _load_text_file(base / SYSTEM_PROMPT_FILE)
        self._database_schema = _load_text_file(base / DATABASE_SCHEMA_FILE)
        self._knowledge_schema = _load_text_file(base / KNOWLEDGE_SCHEMA_FILE)
        self._prediction_schema = _load_text_file(base / PREDICTION_SCHEMA_FILE)
        self._routing_rules = _load_text_file(base / ROUTING_RULES_FILE)

    @staticmethod
    def _block(title: str, body: str) -> str:
        return f"{_SECTION}\n{title}\n{_SECTION}\n{body.strip()}"

    def build(self, conversation_memory: str | dict[str, Any] | None, user_query: str) -> str:
        """Assemble the full planning prompt in the fixed order.

        ``conversation_memory`` may be pre-rendered text, a session snapshot
        dict, or ``None``. ``user_query`` is the current user query.
        """
        if isinstance(conversation_memory, dict) or conversation_memory is None:
            memory_text = render_memory_snapshot(conversation_memory)
        else:
            memory_text = conversation_memory.strip() or _NO_MEMORY

        parts = [
            self._system_prompt.strip(),
            self._block("CONVERSATION MEMORY", memory_text),
            self._block("DATABASE SCHEMA (Data Agent)", self._database_schema),
            self._block("KNOWLEDGE AGENT SCHEMA", self._knowledge_schema),
            self._block("PREDICTION AGENT SCHEMA", self._prediction_schema),
            self._block("ROUTING RULES (authoritative)", self._routing_rules),
            self._block("CURRENT USER QUERY", user_query.strip()),
            "Return only the single JSON routing decision. Nothing else.",
        ]
        return "\n\n".join(parts)
