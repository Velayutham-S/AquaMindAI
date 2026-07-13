"""Temporary integration test for the AquaMind AI Conversation Memory Manager.

NOT production code. Exercises the full public API and the design guarantees
(multi-session support, session isolation, thread safety, follow-up context and
entity storage) and writes a human-readable report to
``conversation_memory_output.txt`` at the project root.

It calls no LLM and no agent, does no reasoning and no inference -- it only
verifies that state is stored and retrieved correctly.

Run:
    python agents/supervisor_agent/tests/conversation_memory_test.py
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
SUPERVISOR_DIR = TEST_DIR.parent
PROJECT_ROOT = TEST_DIR.parents[2]  # tests -> supervisor_agent -> agents -> root
if str(SUPERVISOR_DIR) not in sys.path:
    sys.path.insert(0, str(SUPERVISOR_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from memory import (  # noqa: E402
    AgentName,
    ConversationMemory,
    IntentType,
    MessageRole,
    SessionNotFoundError,
)

OUTPUT_PATH = PROJECT_ROOT / "conversation_memory_output.txt"
SECTION = "=" * 50
RULE = "-" * 50


class Report:
    """Accumulates per-test output blocks and pass/fail counters."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.passed = 0
        self.failed = 0
        self.sessions_created = 0
        self.messages_stored = 0

    def check(self, condition: bool) -> bool:
        ok = bool(condition)
        self.passed += int(ok)
        self.failed += int(not ok)
        return ok

    def block(self, memory: ConversationMemory, title: str, session_id: str,
              retrieved=None, note: str = "", status: str = "PASS") -> None:
        self.lines.append(SECTION)
        self.lines.append(f"Test Case          : {title}")
        self.lines.append(f"Status             : {status}")
        self.lines.append(RULE)
        if memory.session_exists(session_id):
            snapshot = memory.get_session(session_id).to_dict()
            self.lines.append(f"Session ID         : {snapshot['session_id']}")
            self.lines.append(f"Conversation Turns : {snapshot['turn']}")
            self.lines.append(f"Stored Messages    : {snapshot['message_count']}")
            retrieved = retrieved or []
            self.lines.append(f"Retrieved Messages : {len(retrieved)}")
            for message in retrieved:
                self.lines.append(f"    - {message.role.value}: {message.content}")
            self.lines.append(f"Current Context    : "
                              f"{json.dumps(snapshot['context'], ensure_ascii=False)}")
            self.lines.append(f"Referenced Entities: "
                              f"{json.dumps(snapshot['entities'], ensure_ascii=False)}")
            self.lines.append(f"Active Agent       : {snapshot['context']['current_active_agent']}")
            self.lines.append(f"Current Intent     : {snapshot['context']['current_intent']}")
            self.lines.append(f"Last Response      : "
                              f"{json.dumps(snapshot['last_response'], ensure_ascii=False)}")
            self.lines.append(f"Conversation Metadata: "
                              f"{json.dumps(snapshot['metadata'], ensure_ascii=False)}")
            state = {
                "created_at": snapshot["created_at"],
                "updated_at": snapshot["updated_at"],
                "turn": snapshot["turn"],
                "message_count": snapshot["message_count"],
                "exists": True,
            }
            self.lines.append(f"Current Session State: {json.dumps(state, ensure_ascii=False)}")
        else:
            self.lines.append(f"Session ID         : {session_id}")
            self.lines.append("Current Session State: {\"exists\": false}")
        if note:
            self.lines.append(f"Note               : {note}")
        self.lines.append(SECTION)
        self.lines.append("")

    # -- counted helpers so the summary totals are meaningful --
    def new_session(self, memory: ConversationMemory, session_id: str) -> str:
        session = memory.create_session(session_id)
        self.sessions_created += 1
        return session.session_id

    def user(self, memory: ConversationMemory, session_id: str, text: str):
        self.messages_stored += 1
        return memory.add_user_message(session_id, text)

    def assistant(self, memory: ConversationMemory, session_id: str, text: str):
        self.messages_stored += 1
        return memory.add_assistant_message(session_id, text)


def main() -> int:
    report = Report()
    memory = ConversationMemory(short_term_limit=20)

    # --- 1. Create Session ---
    sid = report.new_session(memory, "session-1")
    ok = report.check(memory.session_exists(sid) and memory.get_turn(sid) == 0
                      and memory.get_recent_messages(sid) == [])
    report.block(memory, "Create Session", sid, status="PASS" if ok else "FAIL")

    # --- 2. Store User Messages (follow-up example 1) ---
    memory.increment_turn(sid)
    report.user(memory, sid, "What is the groundwater level in Salem?")
    last_user = memory.get_last_user_message(sid)
    ok = report.check(last_user is not None
                      and last_user.content == "What is the groundwater level in Salem?"
                      and last_user.role == MessageRole.USER)
    report.block(memory, "Store User Messages", sid,
                 retrieved=memory.get_recent_messages(sid), status="PASS" if ok else "FAIL")

    # --- 3. Store Assistant Messages ---
    report.assistant(memory, sid, "Groundwater level data for Salem is available.")
    last_assistant = memory.get_last_assistant_message(sid)
    ok = report.check(last_assistant is not None
                      and last_assistant.role == MessageRole.ASSISTANT)
    report.block(memory, "Store Assistant Messages", sid,
                 retrieved=memory.get_recent_messages(sid), status="PASS" if ok else "FAIL")

    # --- 4. Retrieve History (full) ---
    memory.increment_turn(sid)
    report.user(memory, sid, "What about Coimbatore?")
    report.assistant(memory, sid, "Groundwater level data for Coimbatore is available.")
    history = memory.get_recent_messages(sid, limit=1000)
    ok = report.check(len(history) == 4 and memory.get_session(sid).message_count() == 4)
    report.block(memory, "Retrieve History", sid, retrieved=history,
                 status="PASS" if ok else "FAIL")

    # --- 5. Retrieve Recent Messages (windowed) ---
    recent = memory.get_recent_messages(sid, limit=2)
    ok = report.check(len(recent) == 2 and recent[-1].content == history[-1].content)
    report.block(memory, "Retrieve Recent Messages", sid, retrieved=recent,
                 note="short-term window limit=2", status="PASS" if ok else "FAIL")

    # --- 6. Retrieve Context (defaults are None; nothing inferred) ---
    context = memory.get_context(sid)
    ok = report.check(context.current_district is None and context.current_topic is None
                      and context.conversation_summary is None)
    report.block(memory, "Retrieve Context", sid, note="fresh context -> all None",
                 status="PASS" if ok else "FAIL")

    # --- 7. Update Context + Entities (follow-up: Salem -> Coimbatore) ---
    memory.update_context(sid, current_topic="groundwater level", current_district="Salem",
                          current_year=2024)
    memory.update_entities(sid, districts=["Salem"])
    # Supervisor-style follow-up update: district changes, topic preserved.
    memory.update_context(sid, current_district="Coimbatore")
    memory.update_entities(sid, districts=["Salem", "Coimbatore"])
    context = memory.get_context(sid)
    ok = report.check(context.current_district == "Coimbatore"
                      and context.current_topic == "groundwater level"
                      and context.current_year == 2024
                      and memory.get_entities(sid).get("districts") == ["Salem", "Coimbatore"])
    report.block(memory, "Update Context", sid,
                 note="district Salem->Coimbatore; topic/year preserved (stored, not inferred)",
                 status="PASS" if ok else "FAIL")

    # --- 8. Multiple Sessions (prediction follow-up example 2) ---
    sid2 = report.new_session(memory, "session-2")
    memory.increment_turn(sid2)
    report.user(memory, sid2, "Predict groundwater level in 2030.")
    report.assistant(memory, sid2, "Prediction generated.")
    memory.update_context(sid2, current_topic="prediction",
                          current_prediction_target="groundwater_level_m", current_year=2030)
    ok = report.check(memory.session_exists("session-1") and memory.session_exists("session-2")
                      and memory.manager.session_count() == 2)
    report.block(memory, "Multiple Sessions", sid2,
                 retrieved=memory.get_recent_messages(sid2),
                 note="second independent conversation", status="PASS" if ok else "FAIL")

    # --- 9. Active Agent, Intent & Last Response (prediction follow-up) ---
    # Supervisor routes the prediction query and records where it went.
    memory.update_context(sid2, current_active_agent=AgentName.PREDICTION_AGENT.value,
                          current_intent=IntentType.PREDICTION_QUERY.value)
    memory.set_last_response(sid2, response_id="resp-001",
                             timestamp="2030-01-01T00:00:00+00:00",
                             agent_names=[AgentName.PREDICTION_AGENT.value], status="SUCCESS")
    # Follow-up "What about 2035?": Supervisor reads previous active agent from memory
    # (no reasoning here) and updates only the year; active agent/intent persist.
    previous_active_agent = memory.get_context(sid2).current_active_agent
    memory.increment_turn(sid2)
    report.user(memory, sid2, "What about 2035?")
    memory.update_context(sid2, current_year=2035)
    context2 = memory.get_context(sid2)
    last_response = memory.get_last_response(sid2)
    ok = report.check(previous_active_agent == "prediction_agent"
                      and context2.current_active_agent == "prediction_agent"
                      and context2.current_intent == "prediction_query"
                      and context2.current_year == 2035
                      and last_response is not None
                      and last_response.response_id == "resp-001"
                      and last_response.agent_names == ("prediction_agent",)
                      and last_response.status == "SUCCESS")
    report.block(memory, "Active Agent, Intent & Last Response", sid2,
                 retrieved=memory.get_recent_messages(sid2, limit=3),
                 note="follow-up 'What about 2035?': active_agent/intent persist, year->2035",
                 status="PASS" if ok else "FAIL")

    # --- 10. Conversation Metadata (session-level preferences) ---
    # Also stamp session-1 with routing state + a last response so the Clear test
    # can prove those reset while metadata (preferences) is preserved.
    memory.update_context(sid, current_active_agent=AgentName.DATA_AGENT.value,
                          current_intent=IntentType.DATA_QUERY.value)
    memory.set_last_response(sid, response_id="resp-000",
                             timestamp="2024-01-01T00:00:00+00:00",
                             agent_names=[AgentName.DATA_AGENT.value], status="SUCCESS")
    memory.update_metadata(sid, language="en", timezone="Asia/Kolkata",
                           preferred_units="metres")
    metadata = memory.get_metadata(sid)
    ok = report.check(metadata.language == "en" and metadata.timezone == "Asia/Kolkata"
                      and metadata.preferred_units == "metres"
                      and memory.get_context(sid).current_active_agent == "data_agent")
    report.block(memory, "Conversation Metadata", sid,
                 note="language/timezone/preferred_units stored (never inferred)",
                 status="PASS" if ok else "FAIL")

    # --- 11. Session Isolation + thread-safety (concurrency) ---
    c1 = memory.get_context(sid)
    c2 = memory.get_context(sid2)
    isolation_ok = (c1.current_district == "Coimbatore" and c2.current_district is None
                    and c2.current_prediction_target == "groundwater_level_m"
                    and c1.current_prediction_target is None)

    thread_count, per_thread_messages = 8, 10

    def worker(index: int) -> None:
        wsid = f"concurrent-{index}"
        memory.create_session(wsid)
        for _ in range(per_thread_messages):
            memory.add_user_message(wsid, f"user {index}")
            memory.add_assistant_message(wsid, f"assistant {index}")
        memory.update_context(wsid, current_district=f"District-{index}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    report.sessions_created += thread_count
    report.messages_stored += thread_count * per_thread_messages * 2

    concurrency_ok = True
    for index in range(thread_count):
        wsid = f"concurrent-{index}"
        session = memory.get_session(wsid)
        if (session.message_count() != per_thread_messages * 2
                or memory.get_context(wsid).current_district != f"District-{index}"):
            concurrency_ok = False
            break

    ok = report.check(isolation_ok and concurrency_ok)
    report.block(memory, "Session Isolation", sid,
                 note=(f"cross-session state isolated; {thread_count} concurrent sessions "
                       f"each stored {per_thread_messages * 2} messages with no cross-talk"),
                 status="PASS" if ok else "FAIL")

    # --- 12. Clear Session (conversation reset, metadata preserved, session kept) ---
    memory.clear_session(sid)
    session1 = memory.get_session(sid)
    ok = report.check(memory.session_exists(sid) and session1.message_count() == 0
                      and memory.get_turn(sid) == 0
                      and memory.get_context(sid).current_district is None
                      and memory.get_context(sid).current_active_agent is None
                      and memory.get_last_response(sid) is None
                      # metadata (user preferences) survives a conversation clear
                      and memory.get_metadata(sid).language == "en"
                      and memory.get_metadata(sid).preferred_units == "metres")
    report.block(memory, "Clear Session", sid,
                 note="history/turn/context/last_response reset; metadata preserved; session kept",
                 status="PASS" if ok else "FAIL")

    # --- 13. Delete Session (removed; idempotent) ---
    deleted = memory.delete_session(sid2)
    deleted_again = memory.delete_session(sid2)
    missing_raises = False
    try:
        memory.get_session(sid2)
    except SessionNotFoundError:
        missing_raises = True
    ok = report.check(deleted and not deleted_again
                      and not memory.session_exists(sid2) and missing_raises)
    report.block(memory, "Delete Session", sid2,
                 note="delete returns True once, False after; get raises SessionNotFoundError",
                 status="PASS" if ok else "FAIL")

    # --- summary ---
    total_tests = report.passed + report.failed
    overall = "PASS" if report.failed == 0 and total_tests > 0 else "FAIL"
    report.lines.append(SECTION)
    report.lines.append("CONVERSATION MEMORY SUMMARY")
    report.lines.append(SECTION)
    report.lines.append(f"Total Sessions     : {report.sessions_created}")
    report.lines.append(f"Total Messages     : {report.messages_stored}")
    report.lines.append(f"Successful Tests   : {report.passed}")
    report.lines.append(f"Failed Tests       : {report.failed}")
    report.lines.append(f"Overall Status     : {overall}")
    report.lines.append(SECTION)

    OUTPUT_PATH.write_text("\n".join(report.lines) + "\n", encoding="utf-8")

    print(f"Ran {total_tests} conversation-memory tests.")
    print(f"Sessions: {report.sessions_created} | Messages: {report.messages_stored} | "
          f"Passed: {report.passed} | Failed: {report.failed}")
    print(f"Report written to {OUTPUT_PATH}")
    print(f"Overall Status: {overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
