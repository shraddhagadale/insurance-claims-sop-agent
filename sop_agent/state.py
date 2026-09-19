"""Session state: phase, cross-phase memory, verification progress, counters, audit trail.

Memory is deliberately *phase-agnostic*: anything the caller says is captured the moment
they say it, tagged with the phase it was heard in. Phases decide what they may *use*,
never what they may *remember*.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    ESCALATED = "ESCALATED"      # handed to a human; terminal
    CLOSED = "CLOSED"            # call ended normally; terminal


PHASE_ORDER = [Phase.VERIFY_ID, Phase.RESOLVE_INTENT, Phase.PROCESS_CASE, Phase.POST_PROCESS]


@dataclass
class MemorySlot:
    value: Any
    captured_in: Phase
    turn: int
    source_text: str = ""
    consumed: bool = False  # set once a later phase has acted on it


@dataclass
class Memory:
    """Typed slot store. Last-write-wins, but every write is kept in history for audit."""
    slots: dict[str, MemorySlot] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def put(self, key: str, value: Any, phase: Phase, turn: int, source: str = "") -> bool:
        if value in (None, "", [], {}):
            return False
        prev = self.slots.get(key)
        if prev and prev.value == value:
            return False
        self.slots[key] = MemorySlot(value, phase, turn, source[:200])
        self.history.append({"key": key, "value": value, "phase": phase.value, "turn": turn})
        return True

    def get(self, key: str, default: Any = None) -> Any:
        slot = self.slots.get(key)
        return slot.value if slot else default

    def has(self, key: str) -> bool:
        return key in self.slots

    def consume(self, key: str) -> None:
        if key in self.slots:
            self.slots[key].consumed = True

    def snapshot(self) -> dict[str, Any]:
        return {
            k: {"value": v.value, "captured_in": v.captured_in.value, "turn": v.turn, "consumed": v.consumed}
            for k, v in self.slots.items()
        }


# Identity fields that count toward the verification threshold (per the SOP).
IDENTITY_FIELDS = ("full_name", "dob", "id_last4")


@dataclass
class Verification:
    verified: bool = False
    party_id: str | None = None
    candidate_party_id: str | None = None
    matched: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    attempts: int = 0            # number of full verification checks that failed
    declined_fields: list[str] = field(default_factory=list)
    caller_role: str = "policyholder"   # or "representative"
    representative_name: str | None = None
    rep_consent_status: str | None = None   # None | pending | approved
    consent_polls: int = 0


@dataclass
class Counters:
    offtopic_streak: int = 0
    offtopic_total: int = 0
    persuasion_attempts: int = 0  # pushes back against a required gate (refusal/frustration)
    negative_emotion_streak: int = 0
    turns: int = 0


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    phase: Phase = Phase.VERIFY_ID
    memory: Memory = field(default_factory=Memory)
    verification: Verification = field(default_factory=Verification)
    counters: Counters = field(default_factory=Counters)
    transcript: list[dict[str, str]] = field(default_factory=list)
    active_case_id: str | None = None
    resolved_intent: str | None = None
    last_claim_candidates: list[dict[str, Any]] = field(default_factory=list)
    last_claim_list_shown: bool = False
    discussed_topics: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    email_status: str = "not_offered"  # not_offered | offered | sent | declined
    email_record: dict[str, Any] | None = None
    awaiting: str | None = "identity"   # what the agent's last message asked for
    consent_scenario: str = "default"
    handoff: dict[str, Any] | None = None
    audit: list[dict[str, Any]] = field(default_factory=list)
    last_trace: dict[str, Any] = field(default_factory=dict)

    def log(self, kind: str, **data: Any) -> None:
        self.audit.append({"turn": self.counters.turns, "phase": self.phase.value, "kind": kind, **data})

    def transition(self, new_phase: Phase, reason: str) -> None:
        if new_phase != self.phase:
            self.log("transition", frm=self.phase.value, to=new_phase.value, reason=reason)
            self.phase = new_phase

    @property
    def is_terminal(self) -> bool:
        return self.phase in (Phase.ESCALATED, Phase.CLOSED)
