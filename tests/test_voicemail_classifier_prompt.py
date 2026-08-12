"""The voicemail classifier's system prompt must survive a system-instruction recompose.

Regression guard for a production incident (2026-08-12, prod run 101602): every caller
turn stalled ~5s and end-to-end latency went 2.3s -> 5.2s.

The chain was:

1. ``VoicemailDetector`` sets the classifier's prompt by writing
   ``llm._settings.system_instruction`` directly. That is the live value, not the
   ``_base_system_instruction`` the composer rebuilds from (which is snapshotted only in
   ``LLMService.__init__`` or on an ``LLMUpdateSettingsFrame`` -- the detector uses neither).
2. Upstream then began calling ``_compose_system_instruction()`` at the end of
   ``_sync_registered_tool_handlers()``, which ``LLMService`` runs for *every*
   ``LLMContextFrame``. The composer does
   ``self._settings.system_instruction = composed or None``, so a classifier with no base
   had its prompt replaced by ``None`` on its very first context frame.
3. A prompt-less classifier answers the transcript conversationally instead of emitting a
   VOICEMAIL/CONVERSATION marker, so ``classify_verdict()`` returns ``None`` and no decision
   is ever made.
4. The ``LLMGate`` holding the main LLM only opens on a decision -- so the main LLM never
   ran, and every turn was rescued only by the api's 5s ResponseGuard retry.

Voicemail detection was also completely inert while this was true: answering machines were
never detected and never hung up on.

These tests fail (prompt becomes None) without the paired
``_base_system_instruction`` assignment in ``VoicemailDetector.__init__``.
"""

import unittest

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.extensions.voicemail.voicemail_detector import VoicemailDetector
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings


class _MockLLMService(LLMService):
    """Minimal concrete LLMService, mirroring tests/test_llm_service.py."""

    def __init__(self, **kwargs):
        settings = LLMSettings(
            model="test-model",
            system_instruction=kwargs.pop("system_instruction", None),
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            extra=None,
        )
        super().__init__(settings=settings, **kwargs)

    def create_adapter(self) -> BaseLLMAdapter:
        return BaseLLMAdapter()


def _classifier_of(detector: VoicemailDetector) -> LLMService:
    return detector._classifier_llm


class TestClassifierPromptSurvivesRecompose(unittest.TestCase):
    def setUp(self):
        self.llm = _MockLLMService()
        self.detector = VoicemailDetector(llm=self.llm)
        self.prompt = _classifier_of(self.detector)._settings.system_instruction

    def test_prompt_is_set_at_construction(self):
        """Baseline: the detector installs a non-trivial classifier prompt."""
        self.assertIsInstance(self.prompt, str)
        self.assertGreater(len(self.prompt), 100)
        self.assertIn("VOICEMAIL", self.prompt)
        self.assertIn("CONVERSATION", self.prompt)

    def test_prompt_survives_tool_handler_sync(self):
        """The incident path: every LLMContextFrame syncs tools, which recomposes.

        Without the base being set too, the composer rebuilds from an empty base and
        `composed or None` replaces the whole prompt with None.
        """
        classifier = _classifier_of(self.detector)
        classifier._sync_registered_tool_handlers(None)
        self.assertEqual(
            classifier._settings.system_instruction,
            self.prompt,
            "classifier prompt was wiped by a system-instruction recompose",
        )

    def test_prompt_survives_repeated_syncs(self):
        """The stall recurred on every turn, so the guard must hold under repetition."""
        classifier = _classifier_of(self.detector)
        for _ in range(3):
            classifier._sync_registered_tool_handlers(None)
        self.assertEqual(classifier._settings.system_instruction, self.prompt)

    def test_prompt_survives_direct_recompose(self):
        """Guard the composer itself, not just the one caller that exposed it."""
        classifier = _classifier_of(self.detector)
        classifier._compose_system_instruction()
        self.assertEqual(classifier._settings.system_instruction, self.prompt)

    def test_appended_instruction_extends_rather_than_replaces(self):
        """An append must add to the prompt, not discard it.

        This was already broken before the incident -- appending to a classifier whose
        base was unset dropped the prompt entirely -- so it is guarded here too.
        """
        classifier = _classifier_of(self.detector)
        classifier.append_system_instruction("EXTRA-DIRECTIVE")
        composed = classifier._settings.system_instruction
        self.assertIn("EXTRA-DIRECTIVE", composed)
        self.assertIn("VOICEMAIL", composed)


if __name__ == "__main__":
    unittest.main()
