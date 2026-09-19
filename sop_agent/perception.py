"""Perception = turning one messy caller utterance into structured, typed signals.

The LLM is used here purely as an NLU component with a strict JSON schema. It proposes
values; it never decides gates. A rule-based twin (RulePerceiver) provides the same
contract for deterministic unit tests without an API call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .sop import INTENTS
from .tools import parse_month_year

EMOTIONS = ("neutral", "positive", "frustrated", "angry", "anxious", "confused", "sad")
ACTS = (
    "provide_info", "ask_question", "refuse_info", "request_human", "request_protected_info",
    "clarification_request", "affirm", "deny", "done", "greeting", "thanks", "complaint", "off_topic",
)
CASE_TYPES = ("healthcare", "dental", "auto")
STATUSES = ("denied", "open", "closed")


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


PERCEPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "identity": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "full_name": _nullable({"type": "string"}),
                "dob": _nullable({"type": "string", "description": "ISO YYYY-MM-DD"}),
                "phone": _nullable({"type": "string"}),
                "email": _nullable({"type": "string"}),
                "id_last4": _nullable({"type": "string", "description": "last 4 digits of SSN"}),
                "policy_number": _nullable({"type": "string"}),
            },
            "required": ["full_name", "dob", "phone", "email", "id_last4", "policy_number"],
        },
        "caller_role": {"type": "string", "enum": ["policyholder", "representative", "unknown"]},
        "representative_name": _nullable({"type": "string"}),
        "intent": {"type": "string", "enum": list(INTENTS) + ["none"]},
        "case_hints": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "case_type": _nullable({"type": "string", "enum": list(CASE_TYPES)}),
                "status": _nullable({"type": "string", "enum": list(STATUSES)}),
                "month": _nullable({"type": "integer"}),
                "year": _nullable({"type": "integer"}),
                "case_id": _nullable({"type": "string"}),
            },
            "required": ["case_type", "status", "month", "year", "case_id"],
        },
        "emotion": {"type": "string", "enum": list(EMOTIONS)},
        "emotion_intensity": {"type": "integer", "enum": [0, 1, 2, 3]},
        "acts": {"type": "array", "items": {"type": "string", "enum": list(ACTS)}},
        "declined_fields": {"type": "array", "items": {"type": "string",
                            "enum": ["full_name", "dob", "phone", "email", "id_last4"]}},
        "off_topic": {"type": "boolean"},
        "off_topic_subject": _nullable({"type": "string"}),
        "question": _nullable({"type": "string"}),
    },
    "required": [
        "identity", "caller_role", "representative_name", "intent", "case_hints", "emotion",
        "emotion_intensity", "acts", "declined_fields", "off_topic", "off_topic_subject", "question",
    ],
}


@dataclass
class Perception:
    identity: dict[str, str | None] = field(default_factory=dict)
    caller_role: str = "unknown"
    representative_name: str | None = None
    intent: str = "none"
    case_hints: dict[str, Any] = field(default_factory=dict)
    emotion: str = "neutral"
    emotion_intensity: int = 0
    acts: list[str] = field(default_factory=list)
    declined_fields: list[str] = field(default_factory=list)
    off_topic: bool = False
    off_topic_subject: str | None = None
    question: str | None = None
    source: str = "rules"

    @classmethod
    def from_dict(cls, d: dict[str, Any], source: str) -> "Perception":
        p = cls(**{k: d.get(k, getattr(cls(), k)) for k in cls.__dataclass_fields__ if k != "source"})
        p.source = source
        p.acts = [a for a in (p.acts or []) if a in ACTS]
        p.identity = {k: (v.strip() if isinstance(v, str) and v.strip() else None)
                      for k, v in (p.identity or {}).items()}
        return p

    def has(self, act: str) -> bool:
        return act in self.acts

    @property
    def negative(self) -> bool:
        return self.emotion in ("frustrated", "angry", "anxious", "sad") and self.emotion_intensity >= 1

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ============================================================================ rule-based twin
_NAME = r"([A-Z][a-z]+(?:[ '\-][A-Z][a-z]+){1,3})"
_WORD_NUM = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
             "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_IN_SCOPE = re.compile(
    r"claim|insur|policy|polic|denied|deni|appeal|status|document|upload|submit|portal|pay|"
    r"reimburs|amount|refund|deductible|coverage|covered|auto claim|car claim|accident claim|"
    r"pathology|office note|report|deadline|verify|verification|ssn|social|birth|dob|email|phone|"
    r"representative|human|agent|person|supervisor|summary|help|account|money|fee|bill|"
    r"provider|fax|mail|process|review|how long|when|next step|what now|case|file",
    re.I,
)
_QUESTIONISH = re.compile(r"\?|^(what|who|why|how|when|where|can you|could you|tell me|explain|is it)\b", re.I)


def _spoken_digits(text: str) -> str:
    return re.sub(r"\b(" + "|".join(_WORD_NUM) + r")\b", lambda m: _WORD_NUM[m.group(1).lower()], text, flags=re.I)


class RulePerceiver:
    """Deterministic, regex-based perception for local unit tests."""

    def perceive(self, text: str, ctx: dict[str, Any]) -> Perception:
        raw = text.strip()
        t = _spoken_digits(raw)
        low = t.lower()
        p = Perception(source="rules")
        ident: dict[str, str | None] = {k: None for k in
                                         ("full_name", "dob", "phone", "email", "id_last4", "policy_number")}

        # ---- representative detection
        rep = re.search(r"(?i:on behalf of|calling for|calling about)\s+(?i:my \w+,? )?" + _NAME, t)
        if rep or re.search(r"\b(my (mother|mom|father|dad|wife|husband|son|daughter)'?s? claim)\b", low):
            p.caller_role = "representative"
        if rep:
            ident["full_name"] = rep.group(1)
        name = re.search(r"(?i:my name is|name's|i am|i'm|this is|name:)\s+" + _NAME, t)
        if name:
            nm = name.group(1)
            if p.caller_role == "representative":
                p.representative_name = nm
            else:
                ident["full_name"] = nm
                if "policyholder" in low or "policy holder" in low:
                    p.caller_role = "policyholder"
        elif not ident["full_name"]:
            bare = re.fullmatch(_NAME + r"\.?", raw)
            if bare and ctx.get("awaiting") == "identity":
                ident["full_name"] = bare.group(1)

        # ---- DOB
        dob = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|"
                        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}|"
                        r"\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4})\b", t, re.I)
        if dob:
            from .verification import normalize_dob
            ident["dob"] = normalize_dob(dob.group(1)) or dob.group(1)

        # ---- email / phone / policy / last4
        em = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", t)
        if em:
            ident["email"] = em.group(0).rstrip(".")
        ph = re.search(r"(\+?1?[\s\-.(]*\d{3}[\s\-.)]*\d{3}[\s\-.]*\d{4})\b", t)
        if ph and not (dob and ph.group(1) in dob.group(1)):
            ident["phone"] = ph.group(1)
        pol = re.search(r"\bPOL[\s\-]?\d{3,}\b", t, re.I)
        if pol:
            ident["policy_number"] = pol.group(0).upper().replace(" ", "-")
        l4 = re.search(r"(?:ssn|social|last (?:four|4)(?: digits)?)[^0-9]{0,30}(\d{4})\b", t, re.I)
        if l4:
            ident["id_last4"] = l4.group(1)
        elif ctx.get("awaiting") == "identity" and re.fullmatch(r"\D{0,20}(\d{4})\D{0,5}", t):
            ident["id_last4"] = re.search(r"\d{4}", t).group(0)
        p.identity = ident

        # ---- intent & case hints
        intent = "none"
        if re.search(r"\bdeni|why .*(reject|not paid)", low):
            intent = "denial_question"
        if re.search(r"what (do|should) i (do|submit|send)|next step|appeal|what now", low):
            intent = "next_steps"
        if re.search(r"submit|upload|send (the|my|in)|which documents|what documents|pdf|scan|photo|portal|format", low):
            intent = "document_submission"
        if re.search(r"\bstatus\b|where is|update on|in progress|any news", low):
            intent = intent if intent != "none" else "status_inquiry"
        if re.search(r"how much|amount|reimburs|paid|payment|net pay|fee", low):
            intent = "payment_question"
        if intent == "none" and re.search(r"\bclaim\b", low):
            intent = "general_claim_question"
        p.intent = intent
        month, year = parse_month_year(low)
        ctype = ("healthcare" if re.search(r"health|medical|hospital|doctor", low)
                 else "dental" if "dental" in low or "dentist" in low
                 else "auto" if re.search(r"\bauto\b|\bcar\b|vehicle|accident", low) else None)
        status = ("denied" if re.search(r"\bdeni", low) else "open" if re.search(r"\bopen\b|in progress", low)
                  else "closed" if re.search(r"\bclosed\b|settled", low) else None)
        cid = re.search(r"\bCL[\s\-]?\d{3,}\b", t, re.I)
        p.case_hints = {"case_type": ctype, "status": status, "month": month, "year": year,
                        "case_id": cid.group(0).upper().replace(" ", "-") if cid else None}
        if ctx.get("candidates") and not any(p.case_hints.values()):
            ordinal = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "last": -1}
            for word, i in ordinal.items():
                if re.search(rf"\b{word}\b", low):
                    p.case_hints["case_id"] = ctx["candidates"][i]
                    break

        # ---- emotion
        if re.search(r"ridiculous|absurd|stupid|useless|waste of|fed up|sick of|unacceptable|wtf|damn", low) or "!!" in t:
            p.emotion, p.emotion_intensity = "frustrated", 2
        if re.search(r"furious|angry|pissed|outrageous|sue you|lawyer", low) or (raw.isupper() and len(raw) > 12):
            p.emotion, p.emotion_intensity = "angry", 3
        elif re.search(r"worried|anxious|scared|nervous|stress|can't afford|panick", low):
            p.emotion, p.emotion_intensity = "anxious", 2
        elif re.search(r"confus|don't understand|not sure what|makes no sense|lost", low):
            p.emotion, p.emotion_intensity = "confused", 1
        elif re.search(r"already told you|again\?|how many times|annoy|frustrat", low):
            p.emotion, p.emotion_intensity = "frustrated", 2
        elif re.search(r"thank|great|perfect|awesome|appreciate", low):
            p.emotion, p.emotion_intensity = "positive", 1

        # ---- dialogue acts
        acts: list[str] = []
        if any(ident.values()) or p.representative_name:
            acts.append("provide_info")
        if re.search(r"\b(real person|human|live agent|talk to (a|an|some) ?(person|agent|representative|someone)|"
                     r"speak (to|with) (a|an|some) ?(person|agent|representative|someone|supervisor)|supervisor|manager)\b", low):
            acts.append("request_human")
        if re.search(r"(don'?t|do not|won'?t|not going to|rather not|refuse to|not comfortable)\s+(want to\s+)?"
                     r"(give|share|provide|tell)|why do you need|none of your business|not giving", low):
            acts.append("refuse_info")
            for f, pat in {"id_last4": r"ssn|social|last (four|4)", "dob": r"birth|dob|birthday",
                           "phone": r"phone|number", "email": r"email"}.items():
                if re.search(pat, low):
                    p.declined_fields.append(f)
        if re.search(r"just tell me|tell me why|why (was|is) my claim|what('s| is) (the )?status|what happened to my claim", low):
            acts.append("request_protected_info")
        if re.search(r"already told you|ridiculous|this is (so )?(stupid|absurd)|unacceptable", low):
            acts.append("complaint")
        if re.fullmatch(r"(yes|yeah|yep|sure|ok(ay)?|please( do)?|go ahead|sounds good|yes please|y|absolutely|"
                        r"yes please send( it)?|please send( it)?|send it)[.! ]*", low.strip()):
            acts.append("affirm")
        if re.fullmatch(r"(no|nope|nah|no thanks|no thank you|skip( it)?|don'?t send( it)?|not needed|n)[.! ]*", low.strip()):
            acts.append("deny")
        if re.search(r"that'?s all|that is all|nothing else|no more questions|that'?s it|"
                     r"i'?m (good|all set|done)|i am (good|all set|done)|"
                     r"all good|no,? that'?s (all|it)|bye|goodbye|have a good", low):
            acts.append("done")
        if re.match(r"^(hi|hello|hey|good (morning|afternoon|evening))\b", low):
            acts.append("greeting")
        if re.search(r"\bthank", low):
            acts.append("thanks")
        if re.search(r"what do you mean|can you explain|i don'?t understand|clarify|which (one|fields)", low):
            acts.append("clarification_request")
        if _QUESTIONISH.search(t) or "request_protected_info" in acts:
            acts.append("ask_question")
            p.question = raw

        # ---- scope
        conversational = set(acts) & {"affirm", "deny", "done", "greeting", "thanks", "provide_info",
                                      "request_human", "refuse_info", "complaint", "clarification_request"}
        if not _IN_SCOPE.search(t) and not conversational and len(low.split()) > 1:
            p.off_topic = True
            p.off_topic_subject = raw[:80]
            acts.append("off_topic")
        p.acts = list(dict.fromkeys(acts))
        return p
