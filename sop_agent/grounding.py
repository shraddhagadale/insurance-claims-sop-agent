"""Grounded facts per data scope, deterministic case answers, and the output guard."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .fixtures import Fixtures
from .sop import SOP, DataScope
from .state import Phase, Session
from . import tools

FIELD_LABELS = {
    "full_name": "your full name",
    "dob": "your date of birth",
    "phone": "the phone number on file",
    "email": "the email address on file",
    "id_last4": "the last 4 digits of your SSN (or national ID)",
}


def pretty_date(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        d = datetime.strptime(iso, "%Y-%m-%d")
        return d.strftime("%B %-d, %Y")
    except ValueError:
        return iso


def money(v: str | None) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)


def join_or(items: list[str]) -> str:
    if len(items) <= 2:
        return " or ".join(items)
    return ", ".join(items[:-1]) + ", or " + items[-1]


def join_and(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


# ---------------------------------------------------------------------------- directives
@dataclass
class Beat:
    instruction: str     # what the LLM speaker must convey
    template: str        # deterministic phrasing for tests and guard fallback


@dataclass
class Directive:
    action: str
    beats: list[Beat] = field(default_factory=list)
    empathy: str | None = None
    must_not: list[str] = field(default_factory=list)
    awaiting: str | None = None

    @property
    def instructions(self) -> list[str]:
        return [b.instruction for b in self.beats]

    def template(self) -> str:
        return " ".join(b.template.strip() for b in self.beats if b.template.strip())

    def add(self, instruction: str, template: str) -> "Directive":
        self.beats.append(Beat(instruction, template))
        return self


# ---------------------------------------------------------------------------- facts by scope
def case_facts(session: Session, fx: Fixtures, as_of: str, question: str | None, intent: str | None) -> dict[str, Any]:
    claim = tools.get_claim(session, fx, session.active_case_id)
    facts: dict[str, Any] = {
        "today": as_of,
        "claim": claim,
        "amount_field_meanings": {k: v["description"] for k, v in fx.claim_schema["field_descriptions"].items()},
    }
    if claim.get("appeal_deadline"):
        days = (date.fromisoformat(claim["appeal_deadline"]) - date.fromisoformat(as_of)).days
        facts["appeal_deadline_status"] = (
            f"{days} days remaining" if days > 0 else "deadline is today" if days == 0 else f"passed {-days} days ago")
    if claim.get("documents_needed"):
        facts["document_guidance"] = tools.get_document_guidance(session, fx, claim["case_id"])
    facts["followup_guidance"] = tools.get_followup_guidance(session, fx, claim["case_id"], question or "", intent)
    holder = fx.policyholder(session.verification.party_id)
    facts["caller_first_name"] = holder["name"].split()[0]
    facts["email_on_file_masked"] = tools.mask_email(holder["email"])
    return facts


def build_facts(session: Session, fx: Fixtures, as_of: str, question: str | None = None) -> dict[str, Any]:
    """THE data-scope gate: what the speaking model is allowed to see this turn."""
    spec = SOP.get(session.phase)
    scope = spec.data_scope if spec else DataScope.NONE
    if session.phase in (Phase.ESCALATED, Phase.CLOSED):
        scope = DataScope.FULL_CASE if session.active_case_id else DataScope.NONE
    if scope == DataScope.NONE or not session.verification.verified:
        return {}
    if scope == DataScope.CASE_INDEX or not session.active_case_id:
        holder = fx.policyholder(session.verification.party_id)
        return {"caller_first_name": holder["name"].split()[0],
                "claims_on_file": tools.list_claims(session, fx), "today": as_of}
    return case_facts(session, fx, as_of, question, session.resolved_intent)


# ---------------------------------------------------------------------------- deterministic case answers
def case_label(c: dict[str, Any]) -> str:
    return f"{c['case_type']} claim {c['case_id']} from {pretty_date(c['created_at'])}"


def answer_template(facts: dict[str, Any], intent: str | None, question: str | None) -> str:
    c = facts["claim"]
    docs = c.get("documents_needed") or []
    status = c["status"]
    out: list[str] = []
    dl = ""
    if c.get("appeal_deadline"):
        dl = f"The deadline to submit for reconsideration is {pretty_date(c['appeal_deadline'])} ({facts.get('appeal_deadline_status')})."

    if intent in (None, "none", "denial_question", "general_claim_question", "status_inquiry") or not intent:
        if status == "denied":
            out.append(f"This claim was denied because {c['denial_reason']}.")
            if docs:
                out.append(f"To have it reconsidered, we need the {join_and(docs)}.")
            out.append(dl)
        elif status == "open":
            out.append(f"This claim is still open and in progress. The expected reimbursement is "
                       f"{money(c['expected_reimbursement_amount'])} (allowed maximum {money(c['allowed_max_amount'])}); "
                       f"nothing has been paid out yet.")
        else:
            out.append(f"This claim is closed ({c['summary'].lower()}). The final payment was {money(c['net_pay'])} "
                       f"against an allowed maximum of {money(c['allowed_max_amount'])}.")
    if intent == "payment_question":
        out.append(f"Here are the amounts on file: expected reimbursement {money(c['expected_reimbursement_amount'])}, "
                   f"allowed maximum {money(c['allowed_max_amount'])}, and paid so far (net pay) {money(c['net_pay'])}.")
    if intent in ("document_submission", "next_steps") and docs:
        dg = facts.get("document_guidance", {})
        lines = []
        for d in dg.get("documents", []):
            req = d.get("requirements") or ""
            lines.append(f"- {d['document'].capitalize()}: {req}".strip())
        out.append("Here's what to send:\n" + "\n".join(lines))
        out.append(dg.get("general", ""))
        if intent == "next_steps":
            out.append(dl)
    fu = facts.get("followup_guidance", {})
    for m in fu.get("matched", [])[:2]:
        if question and m["guidance"] not in " ".join(out):
            out.append(m["guidance"])
    if question and not fu.get("matched") and intent not in ("denial_question", "payment_question") and \
            re.search(r"how long|when|after", question.lower()):
        out.append(fu.get("fallback") or "")
    return " ".join(o for o in out if o).strip()


def follow_up_items(facts: dict[str, Any]) -> list[str]:
    c = facts["claim"]
    items: list[str] = []
    dg = facts.get("document_guidance")
    if dg:
        for d in dg["documents"]:
            items.append(f"Submit the {d['document']} for claim {c['case_id']}.")
        items.append("Target: send the documents within a week; upload through the member portal or claim "
                     "upload link (support can arrange fax or mail if needed).")
    if c.get("appeal_deadline"):
        items.append(f"Reconsideration deadline: {pretty_date(c['appeal_deadline'])}.")
    if dg:
        items.append(f"After the files are received, review usually takes "
                     f"{facts['followup_guidance']['average_processing_time']}; check the claim status after a few business days.")
    if c["status"] == "open":
        items.append(f"Claim {c['case_id']} is in progress; expected reimbursement {money(c['expected_reimbursement_amount'])}.")
    if c["status"] == "closed":
        items.append(f"Claim {c['case_id']} is closed; no further action needed.")
    return items


# ---------------------------------------------------------------------------- output guard
_CASE_ID = re.compile(r"\bCL-\d{3,}\b", re.I)
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{2})?)")
_ISO = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_NAMED_DATE = re.compile(r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4})\b")
_INTERNAL = re.compile(r"\b(directive|harness|SOP|grounded[_ ]facts|workflow engine|JSON)\b", re.I)


def guard(session: Session, fx: Fixtures, text: str, facts: dict[str, Any]) -> list[str]:
    """Returns a list of violations. Any violation => the templated reply is used instead."""
    v: list[str] = []
    if _INTERNAL.search(text):
        v.append("mentions internal machinery")
    l4 = session.memory.get("id_last4")
    if l4 and re.search(rf"\b{re.escape(str(l4)[-4:])}\b", text):
        v.append("echoed SSN digits")
    dob = session.memory.get("dob")
    if dob and dob in text:
        v.append("echoed DOB")

    if not session.verification.verified:
        if _CASE_ID.search(text) or _MONEY.search(text):
            v.append("claim data before verification")
        for rec in fx.policyholders:
            for key in ("email", "phone", "policy_number"):
                val = rec.get(key)
                if val and val.lower() in text.lower() and val.lower() not in " ".join(
                        m["text"].lower() for m in session.transcript if m["role"] == "user"):
                    v.append(f"leaked {key} before verification")
        if re.search(r"\b(was|has been|got) (denied|approved|paid|rejected)\b", text, re.I):
            v.append("claim outcome before verification")
        return v

    blob = json.dumps(facts)
    blob_nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", blob)}
    for cid in _CASE_ID.findall(text):
        if cid.upper() not in blob:
            v.append(f"ungrounded case id {cid}")
    for amt in _MONEY.findall(text):
        a = amt.replace(",", "")
        if a not in blob_nums and f"{float(a):.2f}" not in blob_nums:
            v.append(f"ungrounded amount ${amt}")
    dates = set(_ISO.findall(text))
    for nd in _NAMED_DATE.findall(text):
        dates.add(datetime.strptime(nd, "%B %d, %Y").strftime("%Y-%m-%d"))
    for d in dates:
        if d not in blob:
            v.append(f"ungrounded date {d}")
    return v
