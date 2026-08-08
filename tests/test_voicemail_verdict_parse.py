"""Tests for the pure voicemail verdict parser.

The offline eval harness (api/evals/voicemail) imports classify_verdict so
that eval measurements use production's exact parse. These tests pin the
load-bearing semantics: substring match on the uppercased response, and
CONVERSATION taking precedence over VOICEMAIL when both appear.
"""

from pipecat.extensions.voicemail.voicemail_detector import classify_verdict


def test_conversation_detected():
    assert classify_verdict("CONVERSATION") == "conversation"


def test_voicemail_detected():
    assert classify_verdict("VOICEMAIL") == "voicemail"


def test_case_insensitive_and_embedded():
    assert classify_verdict("The answer is: conversation.") == "conversation"
    assert classify_verdict("voicemail detected") == "voicemail"


def test_conversation_wins_when_both_present():
    # Hanging up on a live person is worse than listening to a recording:
    # a response containing both markers must resolve to conversation.
    assert classify_verdict("VOICEMAIL... no wait, CONVERSATION") == "conversation"
    assert classify_verdict("CONVERSATION or VOICEMAIL") == "conversation"


def test_no_decision_on_neither():
    assert classify_verdict("") is None
    assert classify_verdict("I cannot classify this.") is None
