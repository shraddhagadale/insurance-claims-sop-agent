from __future__ import annotations

from insurance_claims.sop_agent.config import Settings
from insurance_claims.sop_agent.engine import SOPAgent
from insurance_claims.sop_agent.state import Phase
from insurance_claims.sop_agent import tools

CLAIM_SCOPE_REDIRECT = (
    "I'm only able to help with insurance claims. If you have a question about a claim, "
    "I'm happy to help with that. Do you have a claim you'd like to discuss?"
)


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


def test_initial_verification_asks_only_required_identity_fields() -> None:
    a = agent()
    session = a.new_session()
    greeting = session.transcript[-1]["text"].lower()

    assert "full name" in greeting
    assert "date of birth" in greeting
    assert "last four" in greeting
    assert "phone" not in greeting
    assert "email" not in greeting
    assert "national id" not in greeting
    assert "—" not in greeting


def test_partial_verification_asks_for_missing_required_fields_only() -> None:
    a = agent()
    session = a.new_session()

    result = a.handle_turn(session, "My name is Margaret Chen.")
    reply = result.reply.lower()

    assert result.phase == Phase.VERIFY_ID.value
    assert "date of birth" in reply
    assert "last four" in reply
    assert "phone" not in reply
    assert "email" not in reply
    assert "full name" not in reply


def test_partial_name_and_dob_asks_only_for_ssn() -> None:
    a = agent()
    session = a.new_session()

    result = a.handle_turn(session, "My name is Margaret Chen and DOB is 1985-03-15.")
    reply = result.reply.lower()

    assert result.phase == Phase.VERIFY_ID.value
    assert "last four" in reply
    assert "full name" not in reply
    assert "date of birth" not in reply
    assert "national id" not in reply


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


def test_failed_verification_does_not_repeat_rationale_or_use_unverified_name() -> None:
    a = agent()
    session = a.new_session()

    result = a.handle_turn(
        session,
        "My name is Nakul Gupta, DOB is 2004-08-04, SSN last four is 0000.",
    )
    reply = result.reply.lower()

    assert result.verified is False
    assert result.phase == Phase.VERIFY_ID.value
    assert "nakul" not in reply
    assert "private claim" not in reply
    assert "health" not in reply
    assert "financial" not in reply
    assert "verify" in reply
    assert "—" not in result.reply


def test_repeated_off_topic_questions_keep_redirecting_without_escalation() -> None:
    a = agent()
    session = a.new_session()

    a.handle_turn(session, "what is RL?")
    a.handle_turn(session, "what is RL?")
    result = a.handle_turn(session, "what is RL?")

    assert result.phase == Phase.VERIFY_ID.value
    assert result.handoff is None
    assert result.reply == CLAIM_SCOPE_REDIRECT


def test_claim_definition_is_allowed_after_verification() -> None:
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    result = a.handle_turn(session, "What is claim?")

    assert result.phase == Phase.PROCESS_CASE.value
    assert "request" in result.reply.lower()
    assert "insurance" in result.reply.lower()


def test_dental_claim_definition_is_allowed_after_verification() -> None:
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    result = a.handle_turn(session, "What is a dental claim?")

    assert result.phase == Phase.PROCESS_CASE.value
    assert "dental claim" in result.reply.lower()


def test_random_anatomy_question_redirects_to_claim_scope_after_verification() -> None:
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    result = a.handle_turn(session, "What is teeth?")

    assert result.phase == Phase.PROCESS_CASE.value
    assert result.reply == CLAIM_SCOPE_REDIRECT


def test_gums_question_redirects_to_claim_scope_after_verification() -> None:
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    result = a.handle_turn(session, "What are gums?")

    assert result.phase == Phase.PROCESS_CASE.value
    assert result.reply == CLAIM_SCOPE_REDIRECT


def test_can_list_claims_again_after_processing_one_claim() -> None:
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    result = a.handle_turn(session, "List my claims again.")

    assert result.phase == Phase.PROCESS_CASE.value
    assert "healthcare claim CL-2048" in result.reply
    assert "dental claim CL-1899" in result.reply
    assert "auto claim CL-2102" in result.reply


def test_can_list_other_claims_after_processing_one_claim() -> None:
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    result = a.handle_turn(session, "What other claims do I have?")

    assert result.phase == Phase.PROCESS_CASE.value
    assert "healthcare claim CL-2048" in result.reply
    assert "dental claim CL-1899" in result.reply
    assert "auto claim CL-2102" in result.reply


def test_can_list_claims_without_specific_claim_context_after_processing() -> None:
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    result = a.handle_turn(session, "Do I have any other claims?")

    assert result.phase == Phase.PROCESS_CASE.value
    assert "healthcare claim CL-2048" in result.reply
    assert "dental claim CL-1899" in result.reply
    assert "auto claim CL-2102" in result.reply


def test_can_switch_from_healthcare_claim_to_auto_claim() -> None:
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    result = a.handle_turn(session, "What about my auto claim?")

    assert result.phase == Phase.PROCESS_CASE.value
    assert result.active_case_id == "CL-2102"
    assert "open and in progress" in result.reply.lower()


def test_can_switch_from_healthcare_claim_to_dental_claim() -> None:
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    result = a.handle_turn(session, "Can we discuss the dental one?")

    assert result.phase == Phase.PROCESS_CASE.value
    assert result.active_case_id == "CL-1899"
    assert "closed" in result.reply.lower()


def test_standalone_dental_question_redirects_after_verification() -> None:
    a = agent()
    session = a.new_session()
    a.handle_turn(
        session,
        "I’m the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I’m calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
    )

    result = a.handle_turn(session, "What is dental?")

    assert result.phase == Phase.PROCESS_CASE.value
    assert result.reply == CLAIM_SCOPE_REDIRECT


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
