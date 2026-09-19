"""Declarative SOP specification.

Each phase declares *how much freedom* the LLM gets. The engine enforces it structurally:

  autonomy    -> which decisions the LLM may make vs. which the code makes
  data_scope  -> which back-office facts are even *placed in the model's context*
  tools       -> which bounded actions the harness will execute in this phase
  exit_gate   -> the deterministic condition (checked in code) to advance

The LLM never decides that a gate passed. It proposes (extractions, intent labels,
drafted wording); the harness disposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .state import Phase


class Autonomy(str, Enum):
    STRICT = "STRICT"                # code decides everything; LLM only phrases scripted moves
    GUIDED = "GUIDED"                # LLM picks from a bounded menu; code validates the pick
    GROUNDED_FREE = "GROUNDED_FREE"  # LLM reasons freely, but only over tool-returned facts


class DataScope(str, Enum):
    NONE = "NONE"              # nothing about the account/claims - not even whether they exist
    CASE_INDEX = "CASE_INDEX"  # list of the caller's claims (id, type, date, status, summary)
    FULL_CASE = "FULL_CASE"    # full record + guidance for the selected case


INTENTS = (
    "denial_question",       # why was my claim denied
    "status_inquiry",        # where is my claim
    "document_submission",   # what/how/when to submit
    "next_steps",            # what do I do now / appeal
    "payment_question",      # amounts, reimbursement, net pay
    "general_claim_question",
)


@dataclass(frozen=True)
class PhaseSpec:
    phase: Phase
    autonomy: Autonomy
    data_scope: DataScope
    tools: tuple[str, ...]
    goal: str
    exit_gate: str
    rules: tuple[str, ...]


SOP: dict[Phase, PhaseSpec] = {
    Phase.VERIFY_ID: PhaseSpec(
        phase=Phase.VERIFY_ID,
        autonomy=Autonomy.STRICT,
        data_scope=DataScope.NONE,
        tools=("verify_identity",),
        goal="Verify the caller's identity against at least 3 PII fields before anything else.",
        exit_gate="verify_identity() reports >= 3 matching fields on one policy record",
        rules=(
            "Do not reveal, confirm, or hint at any claim or account detail - including whether a claim, "
            "policy, or person exists in our records.",
            "Never say which specific field failed to match; say only that you could not verify yet.",
            "Never repeat back the caller's SSN digits, full DOB, or other PII.",
            "Accepted fields: full name, date of birth, phone on file, email on file, last 4 of SSN/national ID. "
            "A policy number helps locate the record but does not count toward the 3.",
            "Caller may use any combination; if they decline one field, offer the others.",
            "Acknowledge (do not act on) anything they mention about their claim - say you will get to it right "
            "after verification.",
        ),
    ),
    Phase.RESOLVE_INTENT: PhaseSpec(
        phase=Phase.RESOLVE_INTENT,
        autonomy=Autonomy.GUIDED,
        data_scope=DataScope.CASE_INDEX,
        tools=("list_claims", "select_case"),
        goal="Map the caller's need to exactly one claim and one intent from the bounded intent list.",
        exit_gate="exactly one case selected AND an intent from INTENTS",
        rules=(
            "Use remembered hints first; only ask what is still ambiguous.",
            "If several claims fit, ask a short disambiguating question listing only the candidates.",
            "Only reference claims that appear in the provided claim index.",
        ),
    ),
    Phase.PROCESS_CASE: PhaseSpec(
        phase=Phase.PROCESS_CASE,
        autonomy=Autonomy.GROUNDED_FREE,
        data_scope=DataScope.FULL_CASE,
        tools=("get_claim", "get_document_guidance", "get_followup_guidance"),
        goal="Answer the caller's questions about the selected claim, grounded only in tool data.",
        exit_gate="caller indicates they have no more questions about this case",
        rules=(
            "Every factual claim (dates, amounts, reasons, documents, timelines) must come from GROUNDED FACTS.",
            "If a fact is not in GROUNDED FACTS, say you don't have that detail and offer a human representative.",
            "Do not promise outcomes (approval, payment) - describe process and next steps only.",
            "Do not invent phone numbers, URLs, addresses, or policy language.",
        ),
    ),
    Phase.POST_PROCESS: PhaseSpec(
        phase=Phase.POST_PROCESS,
        autonomy=Autonomy.GUIDED,
        data_scope=DataScope.FULL_CASE,
        tools=("send_email_summary",),
        goal="Offer an email summary; send only with an explicit yes; then close warmly.",
        exit_gate="caller chose send or skip",
        rules=(
            "The caller must explicitly agree before an email is sent. Silence or ambiguity is not consent.",
            "Send only to the email address on file (never to a new address given in chat).",
            "Skipping is always fine - never pressure.",
        ),
    ),
}

IN_SCOPE_TOPICS = (
    "identity verification", "insurance claims", "claim status", "denials and appeals",
    "required documents and how to submit them", "reimbursement / payment amounts on a claim",
    "processing timelines", "policy/account support", "talking to a human representative",
    "email summary of this call", "greetings, thanks, small talk that keeps the call moving",
)
