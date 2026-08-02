#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The user aggregator's inference gate.

``LLMUserAggregatorParams.should_suppress_inference`` lets an application
hold the LLM response for a completed user turn while still committing the
turn's text to the context — e.g. while a scripted opening line is still
being delivered, where a reply generated now would play back-to-back with
it. The contract under test:

* The user message is committed to the context in every case.
* Returning True suppresses the ``LLMContextFrame`` push; False pushes.
* A crashing callback fails open (the push happens).
"""

import unittest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.utils.string import TextPartForConcatenation


def _aggregator(callback):
    context = LLMContext()
    aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(should_suppress_inference=callback),
    )
    pushes = []

    async def _record_push(*args, **kwargs):
        pushes.append(1)

    aggregator.push_context_frame = _record_push
    return context, aggregator, pushes


def _seed(aggregator, text: str):
    aggregator._aggregation.append(
        TextPartForConcatenation(text, includes_inter_part_spaces=False)
    )


def _user_messages(context):
    return [m for m in context.get_messages() if m.get("role") == "user"]


class TestUserInferenceGate(unittest.IsolatedAsyncioTestCase):
    async def test_no_callback_commits_and_pushes(self):
        context, aggregator, pushes = _aggregator(None)
        _seed(aggregator, "hello?")
        result = await aggregator.push_aggregation()
        self.assertEqual(result, "hello?")
        self.assertEqual(len(_user_messages(context)), 1)
        self.assertEqual(len(pushes), 1)

    async def test_suppressed_commits_but_does_not_push(self):
        seen = []

        async def suppress(aggregation: str) -> bool:
            seen.append(aggregation)
            return True

        context, aggregator, pushes = _aggregator(suppress)
        _seed(aggregator, "hello?")
        result = await aggregator.push_aggregation()
        self.assertEqual(result, "hello?")
        self.assertEqual(seen, ["hello?"])
        # Committed to context — the words are not lost…
        self.assertEqual(len(_user_messages(context)), 1)
        # …but no inference was dispatched.
        self.assertEqual(len(pushes), 0)

    async def test_not_suppressed_pushes(self):
        async def allow(_aggregation: str) -> bool:
            return False

        context, aggregator, pushes = _aggregator(allow)
        _seed(aggregator, "hello?")
        await aggregator.push_aggregation()
        self.assertEqual(len(_user_messages(context)), 1)
        self.assertEqual(len(pushes), 1)

    async def test_callback_error_fails_open(self):
        async def boom(_aggregation: str) -> bool:
            raise RuntimeError("gate crashed")

        context, aggregator, pushes = _aggregator(boom)
        _seed(aggregator, "hello?")
        result = await aggregator.push_aggregation()
        self.assertEqual(result, "hello?")
        self.assertEqual(len(_user_messages(context)), 1)
        self.assertEqual(len(pushes), 1)

    async def test_empty_aggregation_never_consults_gate(self):
        calls = []

        async def suppress(aggregation: str) -> bool:
            calls.append(aggregation)
            return True

        _context, aggregator, pushes = _aggregator(suppress)
        result = await aggregator.push_aggregation()
        self.assertEqual(result, "")
        self.assertEqual(calls, [])
        self.assertEqual(len(pushes), 0)


if __name__ == "__main__":
    unittest.main()
