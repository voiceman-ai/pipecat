"""A VOICEMAIL verdict is acted on only when the evidence check backs it.

``ClassificationProcessor`` takes an optional ``voicemail_evidence`` check. On
text it turns down, a VOICEMAIL answer from the LLM is treated as a
CONVERSATION; on text it accepts, or with no check at all, the LLM's verdict
stands. Every decision is recorded with the text it was given on.
"""

import unittest

from pipecat.extensions.voicemail.voicemail_detector import (
    ClassificationProcessor,
    backed_verdict,
)
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test
from pipecat.utils.sync.event_notifier import EventNotifier


def _processor(heard: str, evidence=None):
    context = LLMContext([{"role": "user", "content": heard}])
    gate, conversation, voicemail = EventNotifier(), EventNotifier(), EventNotifier()
    processor = ClassificationProcessor(
        gate_notifier=gate,
        conversation_notifier=conversation,
        voicemail_notifier=voicemail,
        context=context,
        voicemail_evidence=evidence,
    )
    fired = []

    @processor.event_handler("on_conversation_detected")
    async def on_conversation(_):
        fired.append("conversation")

    @processor.event_handler("on_voicemail_detected")
    async def on_voicemail(_):
        fired.append("voicemail")

    return processor, fired


def _answer(text: str):
    return [LLMFullResponseStartFrame(), LLMTextFrame(text), LLMFullResponseEndFrame()]


class TestVoicemailEvidence(unittest.IsolatedAsyncioTestCase):
    async def test_unbacked_voicemail_becomes_conversation(self):
        processor, fired = _processor("ובחדשות הערב, ראש הממשלה נפגש היום", lambda t: False)
        await run_test(processor, frames_to_send=_answer("VOICEMAIL"))
        self.assertEqual(fired, ["conversation"])
        self.assertEqual(processor.decision["verdict"], "conversation")
        self.assertTrue(processor.decision["overruled"])
        self.assertEqual(processor.decision["heard"], "ובחדשות הערב, ראש הממשלה נפגש היום")

    async def test_backed_voicemail_stands(self):
        processor, fired = _processor("הגעתם לתא הקולי של", lambda t: "קולי" in t)
        await run_test(processor, frames_to_send=_answer("VOICEMAIL"))
        self.assertEqual(fired, ["voicemail"])
        self.assertEqual(processor.decision["verdict"], "voicemail")
        self.assertFalse(processor.decision["overruled"])

    async def test_no_check_keeps_the_llm_verdict(self):
        processor, fired = _processor("ובחדשות הערב")
        await run_test(processor, frames_to_send=_answer("VOICEMAIL"))
        self.assertEqual(fired, ["voicemail"])

    async def test_a_failing_check_keeps_the_llm_verdict(self):
        def broken(text):
            raise ValueError("boom")

        processor, fired = _processor("הגעתם לתא הקולי של", broken)
        await run_test(processor, frames_to_send=_answer("VOICEMAIL"))
        self.assertEqual(fired, ["voicemail"])

    async def test_conversation_is_never_checked(self):
        checked = []
        processor, fired = _processor("הלו?", lambda t: checked.append(t) or False)
        await run_test(processor, frames_to_send=_answer("CONVERSATION"))
        self.assertEqual(fired, ["conversation"])
        self.assertEqual(checked, [])
        self.assertFalse(processor.decision["overruled"])

    async def test_no_marker_records_nothing(self):
        processor, fired = _processor("הלו?", lambda t: False)
        await run_test(processor, frames_to_send=_answer("I am not sure"))
        self.assertEqual(fired, [])
        self.assertIsNone(processor.decision)


def test_backed_verdict():
    def no(_text):
        return False

    def boom(_text):
        raise ValueError("boom")

    assert backed_verdict("voicemail", "x", no) == ("conversation", True)
    assert backed_verdict("voicemail", "x", lambda _t: True) == ("voicemail", False)
    assert backed_verdict("voicemail", "x", None) == ("voicemail", False)
    assert backed_verdict("voicemail", "x", boom) == ("voicemail", False)
    assert backed_verdict("conversation", "x", no) == ("conversation", False)
    assert backed_verdict(None, "x", no) == (None, False)


if __name__ == "__main__":
    unittest.main()
