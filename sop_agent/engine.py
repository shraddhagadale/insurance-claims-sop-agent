"""Turn orchestration for the insurance claims SOP harness.

The engine is the workflow owner. The model may extract signals and phrase replies, but
deterministic code owns phase transitions, verification, data access, and consent.
"""
from __future__ import annotations

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
            reply = self._handle_process_case(session, perception)
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

        if session.verification.attempts >= self.settings.max_persuasion_attempts:
            return self._escalate(session, "verification_failed_repeatedly")

        directive = Directive("continue_verification", awaiting="identity")
        self._add_empathy(directive, p)
        if p.has("request_protected_info"):
            directive.add(
                "Acknowledge the claim question without answering it yet.",
                "I can help with that right after identity verification.",
            )
        if enough_to_fail:
            directive.add(
                "Say identity could not be verified yet without naming which fields failed.",
                "I could not verify the account with what I have so far.",
            )
        else:
            directive.add(
                "Say you have some information if applicable, but still need enough identity fields to continue.",
                "I still need to verify your identity before discussing claim details.",
            )
        directive.add(
            "Explain that verification protects private claim, health, financial, and policy information.",
            "That step protects private claim, health, financial, and policy information.",
        )
        directive.add(
            "Ask for any acceptable combination of identity fields, without repeating sensitive values back.",
            "You can use any three of these: full name, date of birth, phone number on file, email on file, or the last four digits of your SSN or national ID.",
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
        labels = "; ".join(case_label(c) for c in claims[:4])
        directive = Directive("need_case_and_intent", awaiting="intent")
        self._add_empathy(directive, p)
        directive.add(
            "Ask what claim or claim issue the caller needs help with, using the verified claim index if available.",
            f"What would you like help with today? I can help with claim status, denials, documents, next steps, or payments. Claims I can see here include: {labels}.",
        )
        return self._speak(session, directive, facts)

    def _handle_process_case(self, session: Session, p: Perception) -> str:
        if p.has("done"):
            session.transition(Phase.POST_PROCESS, "caller finished case questions")
            return self._offer_email_summary(session, p)
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
        if self.use_template_speaker:
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
        if session.counters.offtopic_streak >= self.settings.max_offtopic_strikes:
            return self._escalate(session, "repeated_off_topic")
        directive = Directive("off_topic_redirect", awaiting=session.awaiting)
        directive.add(
            "Politely decline the out-of-scope question and redirect to insurance claim support.",
            "I can only help with insurance claim and account-support questions here, so I cannot answer that. I can keep helping with your claim or connect you with a human representative.",
        )
        return self._speak(session, directive, build_facts(session, self.fixtures, self.settings.as_of_date))

    def _escalate(self, session: Session, reason: str) -> str:
        session.handoff = {"reason": reason, "session_id": session.id}
        session.transition(Phase.ESCALATED, reason)
        return "I understand. A human representative should take it from here, so I will hand this conversation over for review."

    def _add_empathy(self, directive: Directive, p: Perception) -> None:
        if p.emotion in ("frustrated", "angry"):
            directive.empathy = "caller sounds frustrated"
            directive.beats.append(Beat("Acknowledge frustration before continuing.", "I understand why this feels frustrating."))
        elif p.emotion == "anxious":
            directive.empathy = "caller sounds anxious"
            directive.beats.append(Beat("Reassure the caller before continuing.", "I know this can feel stressful, and I will walk through it step by step."))
        elif p.emotion == "confused" or p.has("clarification_request"):
            directive.empathy = "caller sounds confused"
            directive.beats.append(Beat("Clarify simply before continuing.", "I can clarify that."))

    def _initial_message(self, session: Session) -> str:
        directive = Directive("start_verification", awaiting="identity")
        directive.add(
            "Greet the caller, explain identity verification, and ask for acceptable identity fields.",
            "Hi, I am Sam. Before we discuss claim details, I need to verify your identity because claim records can include private health, financial, and policy information. Please provide any three of these: full name, date of birth, phone number on file, email on file, or the last four digits of your SSN or national ID.",
        )
        return self._speak(session, directive, {})

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
