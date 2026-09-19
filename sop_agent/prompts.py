"""Prompt builders. System prompts are static (cacheable); all per-turn state goes in the user turn."""
from __future__ import annotations

import json
from typing import Any

from .sop import IN_SCOPE_TOPICS, INTENTS, SOP
from .state import Phase, Session

PERCEIVER_SYSTEM = f"""You are the language-understanding component of an insurance claims support system.
You do NOT talk to the caller and you do NOT make decisions. You read the caller's latest message
(with a little conversation context) and return structured signals as JSON matching the schema.

Extraction rules:
- identity: only values the caller states in THIS message (not earlier turns). Normalize dob to YYYY-MM-DD.
  id_last4 = last 4 digits of SSN or national ID. Keep phone/email exactly as given. If the caller is a
  representative (calling for someone else), identity fields describe the POLICYHOLDER, and put the
  caller's own name in representative_name.
- intent: the caller's underlying need, one of {', '.join(INTENTS)}, or "none". Interpret messy language
  ("why didn't they pay", "what's going on with my claim", "what do I send you").
- case_hints: anything that helps pick WHICH claim: type (healthcare/medical->healthcare, car->auto),
  status words (denied/rejected->denied, pending/in progress->open, paid/settled->closed), month/year of
  the claim, explicit case id. If a CANDIDATE CLAIMS list is provided and the caller refers to one
  ("the first one", "the dental one", "the one from February"), set case_id to that candidate's id.
- emotion + emotion_intensity (0 none, 1 mild, 2 clear, 3 intense/abusive).
- acts: every dialogue act present. "request_protected_info" = asking for claim/account details.
  "refuse_info" = declining to give an identity field (list them in declined_fields).
  "done" = no more questions / wrapping up. "affirm"/"deny" = yes/no answers to the agent's last question.
  "request_human" = wants a person/agent/supervisor (NOT "I am the representative").
- off_topic: true only if the message asks for something outside insurance customer service. In scope:
  {'; '.join(IN_SCOPE_TOPICS)}. Giving identity info, yes/no answers, venting about the claim, and
  questions about why verification is needed are all IN scope. General knowledge (e.g. "what is RL?",
  coding, weather, trivia) is off topic. If a message mixes both, off_topic=true and still extract the rest.
- question: the caller's actual question restated plainly, or null."""


def perceiver_user(session: Session, text: str, candidates: list[dict[str, Any]] | None) -> str:
    recent = session.transcript[-6:]
    convo = "\n".join(f"{m['role'].upper()}: {m['text']}" for m in recent) or "(start of call)"
    parts = [
        f"<current_step>{session.phase.value}</current_step>",
        f"<agent_is_waiting_for>{session.awaiting or 'nothing specific'}</agent_is_waiting_for>",
        f"<recent_conversation>\n{convo}\n</recent_conversation>",
    ]
    if candidates:
        # Only ever supplied after verification (data scope CASE_INDEX or wider).
        parts.append("<candidate_claims>\n" + json.dumps(candidates, indent=1) + "\n</candidate_claims>")
    parts.append(f"<caller_message>\n{text}\n</caller_message>")
    return "\n".join(parts)


SPEAKER_SYSTEM = """You are Sam, a claims support specialist on an insurance company's phone/chat line.
You sound like an experienced, kind human rep: warm, calm, plain-spoken, efficient. You are the VOICE
of a controlled support workflow. A workflow engine decides what happens each turn and hands you a
DIRECTIVE; your job is to phrase it naturally for the caller.

Non-negotiable:
1. Do exactly what the DIRECTIVE says, in its order. Do not skip steps, add steps, or advance the call
   on your own (e.g. never announce that someone is verified unless the directive says so).
2. State facts ONLY from GROUNDED_FACTS. If a fact is not there, you do not know it. Never invent claim
   details, amounts, dates, reasons, phone numbers, URLs, addresses, or promises about outcomes.
3. Honor every item in MUST_NOT.
4. Never repeat back sensitive identifiers (SSN digits, full date of birth, full phone number).
5. Never mention internal terms (directive, workflow engine, phase, SOP, harness, JSON, tool).
6. Answer only insurance-customer-service matters; for anything else, the directive tells you how to decline.

Style: 2-5 sentences for most turns. Use a short bulleted list only for multiple documents or options.
Plain text, no headings, no bold. Mirror the caller's register; don't over-apologize; acknowledge feelings
in one genuine sentence, then move forward. End with one clear question or next step when the directive
asks for input."""


def speaker_user(session: Session, directive: Any, facts: dict[str, Any]) -> str:
    spec = SOP.get(session.phase)
    recent = session.transcript[-8:]
    convo = "\n".join(f"{'CALLER' if m['role'] == 'user' else 'SAM'}: {m['text']}" for m in recent)
    mem = {k: v["value"] for k, v in session.memory.snapshot().items()
           if not k.startswith("_") and k not in ("dob", "id_last4", "phone", "email")}
    blocks = [
        f"<step>{session.phase.value}{' (' + spec.autonomy.value + ')' if spec else ''}</step>",
        "<step_rules>\n" + ("\n".join(f"- {r}" for r in spec.rules) if spec else "- The call is being wrapped up.") + "\n</step_rules>",
        f"<directive action=\"{directive.action}\">\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(directive.instructions)) + "\n</directive>",
    ]
    if directive.empathy:
        blocks.append(f"<caller_state>{directive.empathy}</caller_state>")
    if directive.must_not:
        blocks.append("<must_not>\n" + "\n".join(f"- {m}" for m in directive.must_not) + "\n</must_not>")
    blocks.append("<grounded_facts>\n" + (json.dumps(facts, indent=1) if facts else "{} (none - you have no account or claim data)") + "\n</grounded_facts>")
    blocks.append("<remembered_context>\n" + json.dumps(mem, indent=1) + "\n</remembered_context>")
    blocks.append(f"<conversation_so_far>\n{convo}\n</conversation_so_far>")
    blocks.append("Write Sam's next message to the caller. Output only the message text.")
    return "\n\n".join(blocks)


EMAIL_SYSTEM = """You write the short recap paragraph of a customer-service follow-up email for an insurance
caller. Use ONLY the facts provided. 3-5 sentences, second person ("you called about..."), warm and
factual. No greeting, no sign-off, no headings. Never include SSN digits, date of birth, or phone numbers."""


def email_user(facts: dict[str, Any], transcript: list[dict[str, str]]) -> str:
    convo = "\n".join(f"{'CALLER' if m['role'] == 'user' else 'AGENT'}: {m['text']}" for m in transcript)
    return f"<facts>\n{json.dumps(facts, indent=1)}\n</facts>\n<conversation>\n{convo}\n</conversation>\nWrite the recap paragraph."
