#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reconnect attempts must be spread over time to be worth anything.

``_reconnect`` disconnected and immediately reconnected with no delay, so all
MAX_CONSECUTIVE_FAILURES attempts landed inside the same failing instant: a
Google-side abort that cleared in under a second burned the entire retry budget
in milliseconds and tore down a live realtime session that would have recovered
on its own (VoiceMan Sentry issue 118).

Three instant retries are one retry. These tests pin the two properties that
make the budget real — a non-zero, growing wait, and jitter — without asserting
exact durations, which would make the schedule unadjustable.
"""

import ast
import unittest
from pathlib import Path

_LLM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "pipecat"
    / "services"
    / "google"
    / "gemini_live"
    / "llm.py"
)


def _module_constants() -> dict:
    """Read the module's constants without importing the google SDK."""
    tree = ast.parse(_LLM_PATH.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


class TestReconnectBackoff(unittest.TestCase):
    def setUp(self):
        self.consts = _module_constants()

    def test_every_delay_is_non_zero(self):
        """THE regression: a zero wait makes the retry budget meaningless."""
        schedule = self.consts["_RECONNECT_BACKOFF_S"]
        self.assertTrue(schedule, "no backoff schedule defined")
        for delay in schedule:
            self.assertGreater(delay, 0, f"zero-delay reconnect in {schedule}")

    def test_the_schedule_grows(self):
        schedule = list(self.consts["_RECONNECT_BACKOFF_S"])
        self.assertEqual(
            schedule,
            sorted(schedule),
            "backoff must not shrink between attempts",
        )

    def test_the_budget_is_bounded_for_a_person_waiting_in_silence(self):
        """A realtime session is someone on a call; recovery must not outlast
        their patience. Total worst-case wait before giving up."""
        schedule = list(self.consts["_RECONNECT_BACKOFF_S"])
        attempts = self.consts["MAX_CONSECUTIVE_FAILURES"]
        worst = sum(schedule[min(i, len(schedule) - 1)] for i in range(attempts))
        self.assertLess(worst, 15.0, f"reconnect budget is {worst:.1f}s of silence")

    def test_jitter_is_applied(self):
        """A fleet-wide blip must not have every live call reconnect in lockstep."""
        self.assertGreater(self.consts["_RECONNECT_JITTER"], 0)

    def test_reconnect_actually_sleeps(self):
        """The constants are inert unless _reconnect awaits them."""
        source = _LLM_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_reconnect":
                body = ast.get_source_segment(source, node) or ""
                self.assertIn(
                    "asyncio.sleep",
                    body,
                    "_reconnect does not wait between attempts",
                )
                return
        self.fail("_reconnect not found")


if __name__ == "__main__":
    unittest.main()
