from __future__ import annotations

from insurance_claims.sop_agent.config import Settings
from insurance_claims.sop_agent.engine import SOPAgent
from insurance_claims.sop_agent.state import Phase
from insurance_claims.sop_agent import tools


def agent() -> SOPAgent:
    settings = Settings(api_key="test-key", max_offtopic_strikes=3)
    return SOPAgent(
        settings=settings,
        require_api_key=False,
        use_rule_perceiver=True,
        use_template_speaker=True,
    )


def test_margaret_sample_verifies_remembers_hint_and_answers_denial() -> None:
    a = agent()
    session = a.new_session()

    result = a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    assert result.verified is True
    assert result.phase == Phase.PROCESS_CASE.value
    assert result.active_case_id == "CL-2048"
    assert result.resolved_intent == "denial_question"
    assert "pathology report" in result.reply
    assert "office note" in result.reply


def test_claim_details_are_not_disclosed_before_verification() -> None:
    a = agent()
    session = a.new_session()

    result = a.handle_turn(
        session,
        "My name is Margaret Chen and my policy is POL-9921. "
        "Just tell me why my denied claim happened.",
    )

    assert result.verified is False
    assert result.phase == Phase.VERIFY_ID.value
    assert "CL-2048" not in result.reply
    assert "pathology" not in result.reply.lower()
    assert "office note" not in result.reply.lower()


def test_repeated_off_topic_questions_escalate() -> None:
    a = agent()
    session = a.new_session()

    a.handle_turn(session, "what is RL?")
    a.handle_turn(session, "what is RL?")
    result = a.handle_turn(session, "what is RL?")

    assert result.phase == Phase.ESCALATED.value
    assert result.handoff["reason"] == "repeated_off_topic"


def test_email_summary_requires_explicit_consent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tools, "OUTBOX", tmp_path)
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )
    offer = a.handle_turn(session, "That is all, no more questions.")

    assert offer.phase == Phase.POST_PROCESS.value
    assert offer.email_status == "offered"

    sent = a.handle_turn(session, "yes please send it")

    assert sent.phase == Phase.CLOSED.value
    assert sent.email_status == "sent"
    assert session.email_record is not None
    assert session.email_record["to"] == "margaret@email.com"


def test_email_summary_can_be_skipped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tools, "OUTBOX", tmp_path)
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )
    a.handle_turn(session, "I am done.")
    declined = a.handle_turn(session, "no thanks")

    assert declined.phase == Phase.CLOSED.value
    assert declined.email_status == "declined"
