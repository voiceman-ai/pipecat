#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The 400-recovery path must be able to run at all.

``SpeachesLLMService.get_chat_completions`` wraps the API call so a vLLM 400
(role alternation, usually) is healed by retrying once against a re-normalized
context instead of dropping a live call. That rescue imported ``assert_given``
from ``pipecat.services.settings``, which does not export it — it lives in
``pipecat.utils.types``.

Because the import sat inside the ``except`` block, it only ran on the failure
path, so the ImportError REPLACED the very 400 the block exists to heal and the
whole recovery was dead code for two months. Nothing caught it: the happy path
never touches that line (VoiceMan Sentry issue 296).

The lesson is the shape, not the name: an import that only executes on a
failure path is untested by construction. This asserts both that the symbol
resolves and that it is imported at module scope, so a future move fails at
import time and in CI rather than during someone's call.
"""

import ast
import unittest
from pathlib import Path

_LLM_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "pipecat" / "services" / "speaches" / "llm.py"
)


class TestAssertGivenIsImportable(unittest.TestCase):
    def test_assert_given_lives_where_the_import_says_it_does(self):
        """The bug, directly: the OLD source module does not define it.

        Asserted against the source rather than by importing either module —
        sibling tests deliberately stub ``pipecat.*`` in ``sys.modules`` to load
        llm.py in isolation, and a real import here would resolve against
        whichever of those ran first.
        """
        src = _LLM_PATH.parents[2]

        defined_in = {
            path.relative_to(src).as_posix()
            for path in src.rglob("*.py")
            if "def assert_given" in path.read_text(encoding="utf-8")
        }
        self.assertEqual(
            defined_in,
            {"utils/types.py"},
            "assert_given moved; update the import in speaches/llm.py",
        )

        settings_src = (src / "services" / "settings.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "def assert_given",
            settings_src,
            "pipecat.services.settings now defines assert_given — this test's "
            "premise no longer holds",
        )

    def test_the_import_is_at_module_scope_not_inside_a_handler(self):
        """A failure-path-only import is untested by construction."""
        tree = ast.parse(_LLM_PATH.read_text(encoding="utf-8"))

        top_level = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertIn(
            "assert_given",
            top_level,
            "assert_given must be imported at module scope so an unresolvable "
            "name fails at import time, not only when a 400 arrives",
        )

    def test_no_import_of_assert_given_remains_inside_a_function(self):
        tree = ast.parse(_LLM_PATH.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.ImportFrom) and any(
                    a.name == "assert_given" for a in inner.names
                ):
                    offenders.append(f"{node.name}:{inner.lineno}")
        self.assertEqual(offenders, [], f"lazy assert_given import(s): {offenders}")


if __name__ == "__main__":
    unittest.main()
