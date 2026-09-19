"""Turn orchestration for the insurance claims SOP harness.

The engine is the workflow owner. The model may extract signals and phrase replies, but
deterministic code owns phase transitions, verification, data access, and consent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings, settings as default_settings
from .fixtures import Fixtures, load_fixtures
from .grounding import (
    Beat,
    Directive,
    answer_template,
    build_facts,
    case_label,
    follow_up_items,
    guard,
)
from .llm import AnthropicLLM, LLMUnavailable
from .perception import PERCEPTION_SCHEMA, Perception, RulePerceiver
from .prompts import EMAIL_SYSTEM, PERCEIVER_SYSTEM, email_user, perceiver_user, speaker_user, SPEAKER_SYSTEM
from .sop import INTENTS
from .state import IDENTITY_FIELDS, Phase, Session
from .tools import list_claims, rank_cases, send_email_summary
from .verification import verify_identity

CLAIM_SUPPORT_RE = re.compile(
    r"\b(claim|claims|insurance|policy|coverage|covered|denial|denied|appeal|appeals|document|documents|"
    r"upload|submit|submission|reimbursement|reimburse|payment|paid|pay|status|deductible|deadline|"
    r"portal|provider|bill|billing|healthcare claim|dental claim|auto claim|case)\b",
    re.I,
)
GENERAL_CLAIM_DEFINITION_RE = re.compile(
    r"\bwhat (is|does) (a |an )?(claim|dental claim|healthcare claim|auto claim|denial|appeal|reimbursement|deductible)\b",
    re.I,
)
CLAIM_LIST_REQUEST_RE = re.compile(
    r"\b(list|show|see)\b.*\b(claim|claims|case|cases)\b|"
    r"\b(all|other|another|any other)\b.*\b(claim|claims|case|cases)\b|"
    r"\bdo i have\b.*\b(claim|claims|case|cases)\b|"
    r"\bwhat (claims|cases) (do i have|are on file)\b|"
    r"\bwhat (other|all) (claims|cases)\b|"
    r"\bwhich (claims|cases) (do i have|are on file)\b|"
    r"\b(claims|cases) (again|on file)\b|"
    r"\b(list them|show them|those claims|the options|you listed|you showed)\b",
    re.I,
)
CASE_SWITCH_RE = re.compile(
    r"\b(switch|change|move|talk|discuss|look|check)\b.*\b(claim|case|one)\b|"
    r"\b(other|another|different|second|third|first|last|auto|dental|healthcare|closed|open)\b.*\b(claim|case|one)\b|"
    r"\b(the )?(auto|dental|healthcare|closed|open|denied) one\b",
    re.I,
)

class ModelClient(Protocol):
    def structured(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        ...

    def text(self, system: str, user: str) -> str:
        ...


@dataclass
class TurnResult:
    session_id: str
    phase: str
    reply: str
    verified: bool
    active_case_id: str | None = None
    resolved_intent: str | None = None
    email_status: str = "not_offered"
    handoff: dict[str, Any] | None = None
    trace: dict[str, Any] = field(default_factory=dict)


class SOPAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        fixtures: Fixtures | None = None,
        llm: ModelClient | None = None,
        *,
        require_api_key: bool = True,
        use_rule_perceiver: bool = False,
        use_template_speaker: bool = False,
    ) -> None:
        self.settings = settings or default_settings
        if require_api_key and llm is None:
            self.settings.require_credentials()
        self.fixtures = fixtures or load_fixtures(str(self.settings.fixtures_dir))
        self.llm = llm or AnthropicLLM(self.settings)
        self.rule_perceiver = RulePerceiver()
        self.use_rule_perceiver = use_rule_perceiver
        self.use_template_speaker = use_template_speaker

    def new_session(self) -> Session:
        session = Session(consent_scenario=self.settings.consent_scenario)
        greeting = self._initial_message(session)
        session.transcript.append({"role": "assistant", "text": greeting})
        return session

    def handle_turn(self, session: Session, text: str) -> TurnResult:
        session.counters.turns += 1
        session.transcript.append({"role": "user", "text": text})

        if session.phase == Phase.CLOSED:
            return self._finalize(session, "This conversation is already closed. If you need more help, please start a new chat.")
        if session.phase == Phase.ESCALATED:
            return self._finalize(session, "A human representative should take it from here.")

        perception = self._perceive(session, text)
        self._remember(session, perception, text)
        self._update_counters(session, perception)

        if perception.has("request_human"):
            reply = self._escalate(session, "caller_requested_human")
            return self._finalize(session, reply, perception)

        if perception.off_topic and not self._has_workflow_signal(perception):
            reply = self._handle_offtopic(session, perception)
            return self._finalize(session, reply, perception)

        if session.phase == Phase.VERIFY_ID:
            reply = self._handle_verify(session, perception)
        elif session.phase == Phase.RESOLVE_INTENT:
            reply = self._handle_resolve_intent(session, perception)
        elif session.phase == Phase.PROCESS_CASE:
            reply = self._handle_process_case(session, perception, text)
        elif session.phase == Phase.POST_PROCESS:
            reply = self._handle_post_process(session, perception)
        else:
            reply = "A human representative should take it from here."

        return self._finalize(session, reply, perception)

    # ------------------------------------------------------------------ perception/memory
    def _perceive(self, session: Session, text: str) -> Perception:
        candidates = None
        if session.verification.verified and session.phase in (Phase.RESOLVE_INTENT, Phase.PROCESS_CASE):
            try:
                candidates = list_claims(session, self.fixtures)
            except Exception:
                candidates = None
        ctx = {"awaiting": session.awaiting, "candidates": [c["case_id"] for c in candidates or []]}
        if self.use_rule_perceiver:
            p = self.rule_perceiver.perceive(text, ctx)
        else:
            data = self.llm.structured(PERCEIVER_SYSTEM, perceiver_user(session, text, candidates), PERCEPTION_SCHEMA)
            p = Perception.from_dict(data, "llm")
        session.last_trace["perception"] = p.to_dict()
        return p

    def _remember(self, session: Session, p: Perception, source: str) -> None:
        for key in (*IDENTITY_FIELDS, "policy_number"):
            session.memory.put(key, p.identity.get(key), session.phase, session.counters.turns, source)
        if p.caller_role in ("policyholder", "representative"):
            session.verification.caller_role = p.caller_role
            session.memory.put("caller_role", p.caller_role, session.phase, session.counters.turns, source)
        session.memory.put("representative_name", p.representative_name, session.phase, session.counters.turns, source)
        if p.intent in INTENTS:
            session.memory.put("intent", p.intent, session.phase, session.counters.turns, source)
        mapping = {
            "case_type": "case_type",
            "status": "case_status",
            "month": "case_month",
            "year": "case_year",
            "case_id": "case_id_hint",
        }
        for src, dest in mapping.items():
            session.memory.put(dest, p.case_hints.get(src), session.phase, session.counters.turns, source)
        for field_name in p.declined_fields:
            if field_name not in session.verification.declined_fields:
                session.verification.declined_fields.append(field_name)

    def _update_counters(self, session: Session, p: Perception) -> None:
        if p.off_topic:
            session.counters.offtopic_streak += 1
            session.counters.offtopic_total += 1
        else:
            session.counters.offtopic_streak = 0
        if p.negative:
            session.counters.negative_emotion_streak += 1
        else:
            session.counters.negative_emotion_streak = 0
        if session.phase == Phase.VERIFY_ID and (p.has("refuse_info") or p.has("request_protected_info")):
            session.counters.persuasion_attempts += 1

    # ------------------------------------------------------------------ phases
    def _handle_verify(self, session: Session, p: Perception) -> str:
        claims = {key: session.memory.get(key) for key in (*IDENTITY_FIELDS, "policy_number")}
        result = verify_identity(claims, self.fixtures, self.settings.required_id_matches)
        session.verification.matched = result.matched
        session.verification.mismatched = result.mismatched
        session.verification.candidate_party_id = result.party_id

        if result.verified and result.party_id:
            session.verification.verified = True
            session.verification.party_id = result.party_id
            session.transition(Phase.RESOLVE_INTENT, "identity verified by deterministic matcher")
            session.awaiting = "intent"
            resolved = self._try_resolve_from_memory(session)
            if resolved:
                return resolved
            directive = Directive("verified_need_intent", awaiting="intent")
            self._add_empathy(directive, p)
            directive.add(
                "Confirm identity is verified, then ask how you can help with the claim.",
                "Thanks, I have verified your identity. What would you like help with on your claim today?",
            )
            return self._speak(session, directive, build_facts(session, self.fixtures, self.settings.as_of_date))

        enough_to_fail = len(result.provided) >= self.settings.required_id_matches
        if enough_to_fail:
            session.verification.attempts += 1

        directive = Directive("continue_verification", awaiting="identity")
        directive.must_not.append("Do not address the caller by any name they provided until identity is verified.")
        directive.must_not.append("Do not ask again for identity fields the caller already provided unless all required fields were provided and verification failed.")
        self._add_empathy(directive, p)
        if p.has("request_protected_info"):
            directive.add(
                "Acknowledge the claim question without answering it yet.",
                "I can help with that right after identity verification.",
            )
        if enough_to_fail:
            directive.add(
                "Say identity could not be verified yet without naming which fields failed.",
                "I was not able to verify the identity with that information.",
            )
        else:
            missing = self._missing_identity_fields(session)
            missing_text = self._identity_fields_text(missing)
            directive.add(
                "Say you still need the missing required identity fields before discussing claim details.",
                f"I still need {missing_text} before I can discuss claim details.",
            )
        if self._should_explain_verification(p):
            directive.add(
                "Briefly explain why identity verification is required.",
                "Verification helps make sure claim details are only shared with the right person.",
            )
        directive.add(
            "Ask for the required identity fields without repeating sensitive values back.",
            self._identity_request_template(session),
        )
        return self._speak(session, directive, {})

    def _handle_resolve_intent(self, session: Session, p: Perception) -> str:
        resolved = self._try_resolve_from_memory(session)
        if resolved:
            return resolved
        return self._ask_for_intent_or_case(session, p)

    def _try_resolve_from_memory(self, session: Session) -> str | None:
        index = list_claims(session, self.fixtures)
        hints = self._memory_hints(session)
        ranking = rank_cases(index, hints)
        intent = session.memory.get("intent")

        session.last_trace["case_ranking"] = {
            "selected": ranking.selected,
            "candidate_ids": [c["case_id"] for c in ranking.candidates],
            "reason": ranking.reason,
            "hints": hints,
        }

        if ranking.selected and intent in INTENTS:
            session.active_case_id = ranking.selected
            session.resolved_intent = intent
            session.last_claim_candidates = ranking.candidates
            for key in ("intent", "case_type", "case_status", "case_month", "case_year", "case_id_hint"):
                session.memory.consume(key)
            session.transition(Phase.PROCESS_CASE, "case and intent resolved")
            return self._answer_current_case(session, None)

        if ranking.selected and intent not in INTENTS:
            session.active_case_id = ranking.selected
            session.awaiting = "intent"
            directive = Directive("need_intent", awaiting="intent")
            directive.add(
                "Tell the caller you found the claim and ask what they need help with.",
                f"I found {case_label(ranking.candidates[0])}. What would you like help with on that claim?",
            )
            return self._speak(session, directive, build_facts(session, self.fixtures, self.settings.as_of_date))

        if ranking.reason != "no_hints" and ranking.candidates:
            session.awaiting = "case"
            session.last_claim_candidates = ranking.candidates
            session.last_claim_list_shown = True
            labels = "; ".join(case_label(c) for c in ranking.candidates[:4])
            directive = Directive("disambiguate_case", awaiting="case")
            directive.add(
                "Ask a short disambiguating question using only candidate claims from the verified account.",
                f"I found a few possible matches: {labels}. Which one should we discuss?",
            )
            return self._speak(session, directive, build_facts(session, self.fixtures, self.settings.as_of_date))
        return None

    def _ask_for_intent_or_case(self, session: Session, p: Perception) -> str:
        facts = build_facts(session, self.fixtures, self.settings.as_of_date)
        claims = facts.get("claims_on_file", [])
        session.last_claim_candidates = claims
        session.last_claim_list_shown = True
        labels = "; ".join(case_label(c) for c in claims[:4])
        directive = Directive("need_case_and_intent", awaiting="intent")
        self._add_empathy(directive, p)
        directive.add(
            "Ask what claim or claim issue the caller needs help with, using the verified claim index if available.",
            f"What would you like help with today? I can help with claim status, denials, documents, next steps, or payments. Claims I can see here include: {labels}.",
        )
        return self._speak(session, directive, facts)

    def _handle_process_case(self, session: Session, p: Perception, text: str) -> str:
        if p.has("done"):
            session.transition(Phase.POST_PROCESS, "caller finished case questions")
            return self._offer_email_summary(session, p)
        case_navigation = self._handle_case_navigation(session, p, text)
        if case_navigation:
            return case_navigation
        general_answer = self._general_claim_support_answer(text)
        if general_answer:
            return self._answer_general_claim_question(session, general_answer)
        if p.has("ask_question") and not self._is_claim_support_question(text):
            return self._redirect_to_claim_scope(session)
        if p.intent in INTENTS:
            session.resolved_intent = p.intent
        return self._answer_current_case(session, p.question)

    def _answer_current_case(self, session: Session, question: str | None) -> str:
        facts = build_facts(session, self.fixtures, self.settings.as_of_date, question)
        answer = answer_template(facts, session.resolved_intent, question)
        session.discussed_topics.append(session.resolved_intent or "claim_question")
        session.follow_ups = follow_up_items(facts)
        session.awaiting = "case_question"

        directive = Directive("answer_case", awaiting="case_question")
        directive.add(
            "Answer the caller's claim question using only the provided grounded answer.",
            answer,
        )
        directive.add(
            "Invite one next claim-related question.",
            "What else can I help you with on this claim?",
        )
        return self._speak(session, directive, facts)

    def _handle_post_process(self, session: Session, p: Perception) -> str:
        if p.has("affirm"):
            session.memory.put("email_consent", True, session.phase, session.counters.turns, "explicit yes")
            facts = build_facts(session, self.fixtures, self.settings.as_of_date)
            body = self._email_body(facts, session)
            subject = f"Summary for claim {session.active_case_id}"
            record = send_email_summary(session, self.fixtures, subject, body)
            session.email_status = "sent"
            session.email_record = record
            session.transition(Phase.CLOSED, "email summary sent")
            directive = Directive("email_sent")
            directive.add(
                "Confirm the email summary was sent to the masked email on file and close warmly.",
                f"I sent the summary to the email address on file. Thanks for contacting us today.",
            )
            return self._speak(session, directive, facts)
        if p.has("deny") or p.has("done"):
            session.email_status = "declined"
            session.transition(Phase.CLOSED, "caller skipped email summary")
            directive = Directive("email_declined")
            directive.add(
                "Acknowledge the caller skipped the email summary and close warmly.",
                "No problem, I will skip the email summary. Thanks for contacting us today.",
            )
            return self._speak(session, directive, build_facts(session, self.fixtures, self.settings.as_of_date))
        return self._offer_email_summary(session, p)

    def _offer_email_summary(self, session: Session, p: Perception) -> str:
        session.email_status = "offered"
        session.awaiting = "email_consent"
        facts = build_facts(session, self.fixtures, self.settings.as_of_date)
        directive = Directive("offer_email_summary", awaiting="email_consent")
        self._add_empathy(directive, p)
        directive.add(
            "Offer to send an email summary and explain explicit consent is required.",
            "Would you like me to send an email summary of what we discussed and the next steps? I need your explicit yes before sending it, and it will only go to the email address on file.",
        )
        return self._speak(session, directive, facts)

    # ------------------------------------------------------------------ helpers
    def _speak(self, session: Session, directive: Directive, facts: dict[str, Any]) -> str:
        if self.use_template_speaker or session.phase == Phase.VERIFY_ID:
            text = directive.template()
        else:
            try:
                text = self.llm.text(SPEAKER_SYSTEM, speaker_user(session, directive, facts))
            except LLMUnavailable:
                text = directive.template()
        violations = guard(session, self.fixtures, text, facts)
        if violations:
            session.log("output_guard_fallback", violations=violations)
            text = directive.template()
        session.awaiting = directive.awaiting
        return text

    def _email_body(self, facts: dict[str, Any], session: Session) -> str:
        if self.use_template_speaker:
            return self._template_email_body(facts, session)
        try:
            return self.llm.text(EMAIL_SYSTEM, email_user(facts, session.transcript))
        except LLMUnavailable:
            return self._template_email_body(facts, session)

    def _template_email_body(self, facts: dict[str, Any], session: Session) -> str:
        claim = facts["claim"]
        followups = " ".join(session.follow_ups[:3])
        return (
            f"You contacted us about claim {claim['case_id']}, which is currently {claim['status']}. "
            f"We discussed the claim status, the outcome on file, and next steps. {followups}"
        ).strip()

    def _memory_hints(self, session: Session) -> dict[str, Any]:
        return {
            "case_id": session.memory.get("case_id_hint"),
            "case_type": session.memory.get("case_type"),
            "status": session.memory.get("case_status"),
            "month": session.memory.get("case_month"),
            "year": session.memory.get("case_year"),
        }

    def _has_workflow_signal(self, p: Perception) -> bool:
        return any(p.identity.values()) or p.intent in INTENTS or any(p.case_hints.values()) or p.has("done")

    def _handle_offtopic(self, session: Session, p: Perception) -> str:
        return self._redirect_to_claim_scope(session)

    def _redirect_to_claim_scope(self, session: Session) -> str:
        directive = Directive("off_topic_redirect", awaiting=session.awaiting)
        directive.add(
            "State what the agent can help with and redirect to insurance claims.",
            "I'm only able to help with insurance claims. If you have a question about a claim, I'm happy to help with that. Do you have a claim you'd like to discuss?",
        )
        return self._speak(session, directive, build_facts(session, self.fixtures, self.settings.as_of_date))

    def _escalate(self, session: Session, reason: str) -> str:
        session.handoff = {"reason": reason, "session_id": session.id}
        session.transition(Phase.ESCALATED, reason)
        return "I understand. A human representative should take it from here, so I will hand this conversation over for review."

    def _add_empathy(self, directive: Directive, p: Perception) -> None:
        if p.emotion in ("frustrated", "angry") and p.emotion_intensity >= 2:
            directive.empathy = "caller sounds frustrated"
            directive.beats.append(Beat("Acknowledge frustration before continuing.", "I understand why this feels frustrating."))
        elif p.emotion == "anxious" and p.emotion_intensity >= 2:
            directive.empathy = "caller sounds anxious"
            directive.beats.append(Beat("Reassure the caller before continuing.", "I know this can feel stressful, and I will walk through it step by step."))
        elif p.emotion == "confused" or p.has("clarification_request"):
            directive.empathy = "caller sounds confused"
            directive.beats.append(Beat("Clarify simply before continuing.", "I can clarify that."))

    def _initial_message(self, session: Session) -> str:
        directive = Directive("start_verification", awaiting="identity")
        directive.must_not.append("Do not address the caller by any name they provided until identity is verified.")
        directive.add(
            "Greet the caller, explain identity verification, and ask for acceptable identity fields.",
            "Hi, I am Sam. Before we discuss claim details, I need to verify your identity. Please provide your full name, date of birth, and the last four digits of your SSN.",
        )
        return self._speak(session, directive, {})

    def _should_explain_verification(self, p: Perception) -> bool:
        return p.has("refuse_info") or p.has("request_protected_info") or p.has("clarification_request") or p.negative

    def _missing_identity_fields(self, session: Session) -> list[str]:
        return [field for field in IDENTITY_FIELDS if not session.memory.get(field)]

    def _identity_fields_text(self, fields: list[str]) -> str:
        labels = {
            "full_name": "your full name",
            "dob": "your date of birth",
            "id_last4": "the last four digits of your SSN",
        }
        items = [labels[field] for field in fields] or ["your full name, date of birth, and the last four digits of your SSN"]
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + f" and {items[-1]}"

    def _identity_request_template(self, session: Session) -> str:
        missing = self._missing_identity_fields(session)
        if missing and len(missing) < len(IDENTITY_FIELDS):
            return f"Please provide {self._identity_fields_text(missing)}."
        return "Please provide your full name, date of birth, and the last four digits of your SSN."

    def _handle_case_navigation(self, session: Session, p: Perception, text: str) -> str | None:
        if self._is_claim_list_request(text):
            return self._list_verified_claims(session)
        if not self._wants_case_switch(p, text):
            return None

        index = list_claims(session, self.fixtures)
        ranking = rank_cases(index, self._perception_hints(p))
        session.last_trace["case_switch_ranking"] = {
            "selected": ranking.selected,
            "candidate_ids": [c["case_id"] for c in ranking.candidates],
            "reason": ranking.reason,
            "hints": self._perception_hints(p),
        }

        if ranking.selected:
            session.active_case_id = ranking.selected
            session.last_claim_candidates = ranking.candidates
            if p.intent in INTENTS:
                session.resolved_intent = p.intent
            return self._answer_current_case(session, p.question)

        candidates = ranking.candidates or index
        session.last_claim_candidates = candidates
        session.last_claim_list_shown = True
        labels = "; ".join(case_label(c) for c in candidates[:4])
        directive = Directive("disambiguate_case_switch", awaiting="case")
        directive.add(
            "Ask which verified claim the caller wants to discuss.",
            f"I found these claims: {labels}. Which one would you like to discuss?",
        )
        return self._speak(session, directive, build_facts(session, self.fixtures, self.settings.as_of_date))

    def _list_verified_claims(self, session: Session) -> str:
        claims = list_claims(session, self.fixtures)
        session.last_claim_candidates = claims
        session.last_claim_list_shown = True
        labels = "; ".join(case_label(c) for c in claims[:4])
        directive = Directive("list_verified_claims", awaiting="case")
        directive.add(
            "List the verified caller's claim options and ask which one to discuss.",
            f"Here are the claims I can help with: {labels}. Which one would you like to discuss?",
        )
        return self._speak(session, directive, build_facts(session, self.fixtures, self.settings.as_of_date))

    def _is_claim_list_request(self, text: str) -> bool:
        return bool(CLAIM_LIST_REQUEST_RE.search(text))

    def _wants_case_switch(self, p: Perception, text: str) -> bool:
        hints = p.case_hints or {}
        has_case_reference = bool(re.search(r"\b(claim|case|one)\b", text, re.I))
        if hints.get("case_id") or hints.get("month") or hints.get("year"):
            return True
        if hints.get("case_type") and has_case_reference:
            return True
        if hints.get("status") and re.search(r"\b(one|claim|case)\b", text, re.I):
            return True
        return bool(CASE_SWITCH_RE.search(text))

    def _perception_hints(self, p: Perception) -> dict[str, Any]:
        hints = p.case_hints or {}
        return {
            "case_id": hints.get("case_id"),
            "case_type": hints.get("case_type"),
            "status": hints.get("status"),
            "month": hints.get("month"),
            "year": hints.get("year"),
        }

    def _is_claim_support_question(self, text: str) -> bool:
        return bool(CLAIM_SUPPORT_RE.search(text) or GENERAL_CLAIM_DEFINITION_RE.search(text))

    def _general_claim_support_answer(self, text: str) -> str | None:
        low = text.lower()
        if re.search(r"\bwhat (is|does) (a |an )?claim\b", low):
            return "A claim is a request for the insurance company to review a covered event, service, or expense and decide what can be paid or reimbursed under the policy."
        if re.search(r"\bwhat (is|does) (a |an )?dental claim\b", low):
            return "A dental claim is a request for insurance review of dental care, such as a visit, procedure, or related expense."
        if re.search(r"\bwhat (is|does) (a |an )?healthcare claim\b", low):
            return "A healthcare claim is a request for insurance review of medical care, services, or related expenses."
        if re.search(r"\bwhat (is|does) (a |an )?auto claim\b", low):
            return "An auto claim is a request for insurance review of vehicle damage, an accident, or a related covered expense."
        if re.search(r"\bwhat (is|does) (a |an )?denial\b|\bwhat does denied mean\b", low):
            return "A denial means the claim was reviewed and was not approved for payment based on the information on file."
        if re.search(r"\bwhat (is|does) (a |an )?appeal\b", low):
            return "An appeal is a request to have a denied claim reviewed again, usually with additional information or documents."
        if re.search(r"\bwhat (is|does) (a |an )?reimbursement\b", low):
            return "A reimbursement is money the insurer pays back for an eligible covered expense."
        return None

    def _answer_general_claim_question(self, session: Session, answer: str) -> str:
        directive = Directive("answer_general_claim_question", awaiting="case_question")
        directive.add(
            "Answer the insurance claims support question briefly.",
            answer,
        )
        directive.add(
            "Invite one next claim-related question.",
            "What else would you like to know about your claim?",
        )
        return self._speak(session, directive, build_facts(session, self.fixtures, self.settings.as_of_date))

    def _finalize(self, session: Session, reply: str, perception: Perception | None = None) -> TurnResult:
        session.transcript.append({"role": "assistant", "text": reply})
        trace = {
            "phase": session.phase.value,
            "awaiting": session.awaiting,
            "memory": session.memory.snapshot(),
            "verification": {
                "verified": session.verification.verified,
                "matched_count": len(session.verification.matched),
                "matched": session.verification.matched,
            },
            **session.last_trace,
        }
        if perception:
            trace["perception"] = perception.to_dict()
        return TurnResult(
            session_id=session.id,
            phase=session.phase.value,
            reply=reply,
            verified=session.verification.verified,
            active_case_id=session.active_case_id,
            resolved_intent=session.resolved_intent,
            email_status=session.email_status,
            handoff=session.handoff,
            trace=trace,
        )
