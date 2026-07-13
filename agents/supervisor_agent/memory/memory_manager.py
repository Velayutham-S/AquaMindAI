"""Active-session registry for the AquaMind AI Conversation Memory Manager.

:class:`MemoryManager` coordinates all active (short-term, in-memory) sessions.
It is the single owner of the session registry and guards it with a lock, so
sessions can be created, looked up and removed safely from multiple threads
(multiple concurrent users). Each session's own state is protected by that
session's lock, so distinct conversations never share mutable state.

Long-term (persistent) memory is represented only by an interface placeholder
(:class:`LongTermMemoryStore`); no persistence is implemented in this milestone.

This module performs NO reasoning, NO routing, NO LLM calls and NO answer
generation. It only tracks sessions.
"""

from __future__ import annotations

import threading
import uuid
from typing import Protocol, runtime_checkable

from .session import Session


class SessionNotFoundError(KeyError):
    """Raised when an operation targets a session id that does not exist."""


@runtime_checkable
class LongTermMemoryStore(Protocol):
    """Interface for a future persistent (long-term) memory backend.

    Placeholder only -- NO concrete persistence is implemented in this milestone.
    A future backend (database, file, cache, ...) can implement this Protocol and
    be injected into :class:`MemoryManager` without changing any calling code
    (Open-Closed / Dependency Injection).
    """

    def save(self, session: Session) -> None: ...

    def load(self, session_id: str) -> Session | None: ...

    def delete(self, session_id: str) -> None: ...


class MemoryManager:
    """Thread-safe registry of active conversation sessions."""

    def __init__(self, long_term_store: LongTermMemoryStore | None = None) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        #: Reserved for future long-term persistence; unused in this milestone.
        self._long_term_store = long_term_store

    def create_session(self, session_id: str | None = None) -> Session:
        """Create and register a new session. Generates an id when none is given.

        Raises ``ValueError`` if the id already exists (use
        :meth:`get_or_create_session` for idempotent creation).
        """
        with self._lock:
            resolved_id = session_id or uuid.uuid4().hex
            if resolved_id in self._sessions:
                raise ValueError(f"Session already exists: {resolved_id}")
            session = Session(session_id=resolved_id)
            self._sessions[resolved_id] = session
            return session

    def get_or_create_session(self, session_id: str) -> Session:
        """Return the existing session for ``session_id`` or create it."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(session_id=session_id)
                self._sessions[session_id] = session
            return session

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def get_session(self, session_id: str) -> Session:
        """Return the session, or raise :class:`SessionNotFoundError`."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            return session

    def delete_session(self, session_id: str) -> bool:
        """Remove a session. Returns ``True`` if it existed, else ``False``."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
