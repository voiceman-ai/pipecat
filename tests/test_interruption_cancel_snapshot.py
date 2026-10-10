#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A barge-in cancels function calls over a snapshot of what is registered.

``_handle_interruptions`` awaits for each function it cancels, and while it
awaits a node transition can (un)register functions on the same dict. Iterated
live, that raised ``RuntimeError: dictionary changed size during iteration``,
which voice-platform treats as a pipeline error and ends the call (campaigns
105/107, 10 live calls cut).
"""

import asyncio
import unittest

from pipecat.frames.frames import InterruptionFrame

from tests.test_llm_service import MockLLMService


async def _noop(params):
    await params.result_callback(None)


class TestInterruptionCancelsOverASnapshot(unittest.IsolatedAsyncioTestCase):
    async def test_a_function_registered_mid_cancel_does_not_raise(self):
        llm = MockLLMService()
        llm.register_function("transition_to_a", _noop)
        llm.register_function("transition_to_b", _noop)

        cancelled = []

        async def _cancel(function_name):
            cancelled.append(function_name)
            # A node transition lands while the cancel awaits.
            llm.register_function(f"registered_after_{len(cancelled)}", _noop)
            llm.unregister_function("transition_to_b") if "transition_to_b" in llm._functions else None
            await asyncio.sleep(0)

        llm._cancel_function_call = _cancel
        await llm._handle_interruptions(InterruptionFrame())
        self.assertEqual(cancelled[:1], ["transition_to_a"])

    async def test_a_call_finishing_mid_cancel_does_not_raise(self):
        llm = MockLLMService()
        llm.register_function("lookup", _noop)

        class _Item:
            def __init__(self, name, tool_call_id):
                self.function_name = name
                self.tool_call_id = tool_call_id
                self.arguments = {}
                self.context = None
                self.registry_item = llm._functions["lookup"]

        loop = asyncio.get_running_loop()
        first = loop.create_future()
        second = loop.create_future()
        llm._function_call_tasks = {first: _Item("lookup", "t1"), second: _Item("lookup", "t2")}

        async def _cancel_task(task, timeout=None):
            # Another call finished meanwhile and removed itself.
            llm._function_call_tasks.pop(second, None)
            await asyncio.sleep(0)

        async def _broadcast(*args, **kwargs):
            await asyncio.sleep(0)

        llm.cancel_task = _cancel_task
        llm.broadcast_frame = _broadcast
        llm._function_call_task_finished = lambda task: llm._function_call_tasks.pop(task, None)
        await llm._cancel_function_call("lookup")
        await llm._cancel_function_calls_by_tool_call_id("t2")


if __name__ == "__main__":
    unittest.main()
