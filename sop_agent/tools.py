"""Bounded tools over the systems of record.

Each tool enforces its own preconditions (verification, case ownership). This is the second
line of defence behind phase data-scoping: even if a prompt were subverted, the tools
refuse to hand out data the SOP hasn't unlocked.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_DIR
from .fixtures import Fixtures
from .state import Session

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


class ToolDenied(Exception):
    """Raised when a tool is called before the SOP allows it."""


def _require_verified(session: Session) -> str:
    if not session.verification.verified or not session.verification.party_id:
        raise ToolDenied("identity not verified")
    return session.verification.party_id


# ---------------------------------------------------------------- claim index / selection
def list_claims(session: Session, fx: Fixtures) -> list[dict[str, Any]]:
    party = _require_verified(session)
    return [
        {k: c.get(k) for k in ("case_id", "case_type", "created_at", "status", "summary")}
        for c in sorted(fx.claims_for(party), key=lambda c: c["created_at"], reverse=True)
    ]


@dataclass
class CaseRanking:
    selected: str | None
    candidates: list[dict[str, Any]]
    reason: str


def rank_cases(index: list[dict[str, Any]], hints: dict[str, Any]) -> CaseRanking:
    """Deterministic scoring of the caller's claims against remembered/extracted hints."""
    if not index:
        return CaseRanking(None, [], "no_claims")
    want_id = (hints.get("case_id") or "").upper().replace(" ", "")
    if want_id:
        for c in index:
            if c["case_id"].upper() == want_id:
                return CaseRanking(c["case_id"], [c], "explicit_case_id")
    ctype = (hints.get("case_type") or "").lower()
    status = (hints.get("status") or "").lower()
    month = hints.get("month")
    year = hints.get("year")
    if not any([ctype, status, month, year]):
        return CaseRanking(None, index, "no_hints")

    scored = []
    for c in index:
        s = 0
        if ctype:
            s += 3 if c["case_type"] == ctype else -3
        if status:
            s += 3 if c["status"] == status else -2
        y, m, _ = (int(p) for p in c["created_at"].split("-"))
        if month:
            s += 2 if m == int(month) else -1
        if year:
            s += 1 if y == int(year) else -1
        scored.append((s, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[0][0]
    tied = [c for s, c in scored if s == top]
    if top > 0 and len(tied) == 1:
        return CaseRanking(tied[0]["case_id"], [tied[0]], "unique_best_match")
    plausible = [c for s, c in scored if s > 0] or [c for _, c in scored]
    return CaseRanking(None, plausible, "ambiguous")


def get_claim(session: Session, fx: Fixtures, case_id: str) -> dict[str, Any]:
    party = _require_verified(session)
    claim = fx.claim(case_id)
    if not claim or claim["party_id"] != party:
        raise ToolDenied("case does not belong to the verified caller")
    return dict(claim)


# ---------------------------------------------------------------- document & follow-up guidance
def _match_doc_key(doc: str, keys: list[str]) -> str | None:
    d = doc.lower()
    for k in keys:
        if d == k or d in k or k in d:
            return k
    dtoks = set(d.split())
    best = max(keys, key=lambda k: len(dtoks & set(k.split())), default=None)
    return best if best and dtoks & set(best.split()) - {"report", "note"} else None


def get_document_guidance(session: Session, fx: Fixtures, case_id: str) -> dict[str, Any]:
    claim = get_claim(session, fx, case_id)
    g = fx.doc_guidelines
    docs = claim.get("documents_needed", [])
    per_doc = []
    for doc in docs:
        key = _match_doc_key(doc, list(g["document_guidance"].keys()))
        per_doc.append({
            "document": doc,
            "requirements": g["document_guidance"].get(key, {}).get("en") if key else None,
            "if_unavailable": (g["document_alternative_guidance"].get(key)
                               or g["document_alternative_guidance"]["default"])["en"],
        })
    return {
        "case_id": claim["case_id"],
        "general": g["default_guidance"]["en"],
        "case_type_guidance": g["case_type_guidance"].get(claim["case_type"], {}).get("en"),
        "documents": per_doc,
        "human_review_rule": g["claim_followup_settings"]["human_review_after_document_alternatives_exhausted"]["en"],
    }


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def get_followup_guidance(session: Session, fx: Fixtures, case_id: str,
                          question: str = "", intent: str | None = None) -> dict[str, Any]:
    claim = get_claim(session, fx, case_id)
    g = fx.doc_guidelines
    docs = claim.get("documents_needed", [])
    fill = {
        "case_id": claim["case_id"],
        "documents": _join(docs) or "the requested documents",
        "average_processing_time_after_submission":
            g["claim_followup_settings"]["average_processing_time_after_submission"]["en"],
    }
    q = question.lower()
    hits, intent_hits = [], []
    for rule in g["claim_followup_guidance"]:
        if rule.get("requires_documents") and not docs:
            continue
        text = rule["en"].format(**fill)
        if any(p in q for p in rule.get("match_any", [])):
            hits.append({"topic": rule["topic"], "guidance": text})
        elif intent and intent in rule.get("intent_hints", []) and "match_any" not in rule:
            intent_hits.append({"topic": rule["topic"], "guidance": text})
    matched = hits or intent_hits
    return {
        "case_id": claim["case_id"],
        "matched": matched,
        "fallback": None if matched else g["claim_followup_fallback"]["en"],
        "average_processing_time": fill["average_processing_time_after_submission"],
    }


# ---------------------------------------------------------------- third-party consent (representatives)
def poll_policyholder_consent(session: Session, fx: Fixtures, scenario: str) -> str:
    """Simulates asking the policyholder (out of band) to approve a representative."""
    seq = fx.consent_scenarios.get(scenario, fx.consent_scenarios["default"])["status_sequence"]
    v = session.verification
    status = seq[min(v.consent_polls, len(seq) - 1)]
    v.consent_polls += 1
    v.rep_consent_status = status
    return status


# ---------------------------------------------------------------- outbound email (simulated)
OUTBOX = Path(PROJECT_DIR / "outbox")


def send_email_summary(session: Session, fx: Fixtures, subject: str, body: str) -> dict[str, Any]:
    party = _require_verified(session)
    if session.memory.get("email_consent") is not True:
        raise ToolDenied("no explicit consent to send email")
    holder = fx.policyholder(party)
    to = holder["email"]  # SOP: only ever the address on file
    record = {
        "to": to, "subject": subject, "body": body,
        "sent_at": time.strftime("%Y-%m-%d %H:%M:%S"), "session_id": session.id,
        "case_id": session.active_case_id,
    }
    OUTBOX.mkdir(exist_ok=True)
    (OUTBOX / f"{session.id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def mask_email(addr: str) -> str:
    user, _, domain = addr.partition("@")
    return (user[0] + "***" + (user[-1] if len(user) > 1 else "")) + "@" + domain


_MONTH_RE = re.compile(
    r"\b(january|february|march|april|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b|\b(?:in|from|of|since|last)\s+(may)\b"
)


def parse_month_year(text: str) -> tuple[int | None, int | None]:
    t = text.lower()
    m = _MONTH_RE.search(t)
    month = None
    if m:
        word = m.group(1) or m.group(2)
        month = next(n for name, n in MONTHS.items() if name.startswith(word[:3]))
    ym = re.search(r"\b(20\d{2})\b", t)
    return month, (int(ym.group(1)) if ym else None)
