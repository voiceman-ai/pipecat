#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The question-swallowed retry: a reply that lost its question to the gate re-asks it.

Prod runs 2137/2139 (wf39): once a barge-in reconciliation note appeared in
context, Gemma-26B started wrapping each turn's owed question inside a
parroted ``[interrupted …]`` note behind a bare acknowledgment. The
:class:`InterruptionMarkerGate` rightly discarded the note — and silently
deleted the question with it. The caller heard "סבבה, הבנתי" and nothing
else, twice in a row, and hung up. The completion was NOT empty (the ack
survived), so the empty-completion retry never fired.

Pinned behaviors:

- the run-2139 shape (ack + note-wrapped question, no tool call) triggers
  exactly ONE corrective retry, whose plain question is pushed downstream
  after the already-spoken ack;
- the retry request carries an ephemeral correction, merged INTO the trailing
  plain user turn — strict-alternation chat templates (gemma3 pythonic) 400
  on two consecutive plain user turns, and the Speaches 400-recovery would
  then rebuild the request without the note;
- a note-only completion rides the EXISTING empty-completion retry and now
  carries the correction too;
- no retry when the visible text still asks a question, when the suppressed
  note carried none, when the completion produced a tool call (the
  transition delivers the destination's opening deterministically), when the
  turn's generation budget is spent, or twice in one dispatch (a retry that
  parrots again gives up rather than loops).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import _append_ephemeral_user_note
from pipecat.services.openai.llm import OpenAILLMService


def _chunk(content=None, tool_call=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_call)
    return SimpleNamespace(usage=None, model=None, choices=[SimpleNamespace(delta=delta)])


def _tool_call_chunk(name=None, arguments=None, call_id=None, index=0):
    tc = SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return _chunk(tool_call=[tc])


async def _stream(chunks):
    for c in chunks:
        yield c


class _FakeStream:
    """Mimics the openai AsyncStream: async-iterable + close()."""

    def __init__(self, chunks):
        self._it = _stream(chunks)

    def __aiter__(self):
        return self._it

    async def close(self):
        await self._it.aclose()


class _FakeClock:
    """Manually advanced monotonic clock; reading it never moves it."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


def _make_service(streams, secs_per_generation: float = 0):
    """Real OpenAILLMService with a scripted sequence of completion streams.

    Records the value of ``_marker_correction_pending`` at each completion
    request (then consumes it, as the real ``get_chat_completions`` does), so
    tests can pin exactly which attempt ran under the correction.
    """
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="test-model"))
    clock = _FakeClock()
    service._generation_clock = clock.monotonic
    calls = {"n": 0, "pending": []}

    async def fake_get_chat_completions(context):
        calls["pending"].append(getattr(service, "_marker_correction_pending", None))
        service._marker_correction_pending = None
        clock.advance(secs_per_generation)
        i = min(calls["n"], len(streams) - 1)
        calls["n"] += 1
        return _FakeStream(streams[i])

    service.get_chat_completions = fake_get_chat_completions
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.start_llm_usage_metrics = AsyncMock()
    pushed = []

    async def fake_push_frame(frame, direction=None):
        pushed.append(frame)

    service.push_frame = fake_push_frame
    service.run_function_calls = AsyncMock()
    return service, pushed, calls


def _pushed_text(pushed):
    return "".join(f.text for f in pushed if isinstance(f, LLMTextFrame))


# Run 2139 turn 8, as vLLM streamed it: a bare ack, then the survey's next
# question wrapped in a parroted reconciliation note (58 completion tokens,
# of which the caller heard only the ack — twice in a row).
RUN_2139_ACK = ["סבבה", ", הבנתי", ". —", "\n"]
RUN_2139_NOTE = [
    "[in",
    "terrupted by the caller; the caller did NOT hear the rest: ",
    '"ומי מבין השלושה הכי מתאים להיות ראש ממשלה לדעתך: ',
    'ביבי, אייזנקוט או בנט?"]',
]
RETRY_QUESTION = ["ומי מבין השלושה ", "הכי מתאים להיות ראש ממשלה לדעתך?"]


def _chunks(texts):
    return [_chunk(t) for t in texts]


@pytest.mark.asyncio
async def test_run_2139_replay_retries_once_and_asks_plainly():
    service, pushed, calls = _make_service(
        [_chunks(RUN_2139_ACK + RUN_2139_NOTE), _chunks(RETRY_QUESTION)]
    )
    await service._process_context(
        LLMContext(messages=[{"role": "user", "content": "לכלכלה."}])
    )

    text = _pushed_text(pushed)
    # The ack was already streaming to TTS when the note was detected — it
    # stays, and the retry's question follows it as the turn's next sentence.
    assert text.startswith("סבבה, הבנתי.")
    assert text.rstrip().endswith("ראש ממשלה לדעתך?")
    assert "interrupted" not in text
    assert calls["n"] == 2
    # First attempt ran clean; the retry ran under the correction, primed
    # with the visible prefix the caller has already heard.
    assert calls["pending"][0] is None
    assert calls["pending"][1] == "סבבה, הבנתי. —"


@pytest.mark.asyncio
async def test_no_retry_when_the_visible_text_still_asks():
    service, pushed, calls = _make_service(
        [_chunks(["רגע, ומה חשוב לך יותר? "] + RUN_2139_NOTE)]
    )
    await service._process_context(
        LLMContext(messages=[{"role": "user", "content": "לכלכלה."}])
    )
    assert calls["n"] == 1
    assert "?" in _pushed_text(pushed)


@pytest.mark.asyncio
async def test_no_retry_when_the_note_carried_no_question():
    note = ['[interrupted by the caller; the caller did NOT hear the rest: "תודה רבה"]']
    service, pushed, calls = _make_service([_chunks(RUN_2139_ACK + note)])
    await service._process_context(
        LLMContext(messages=[{"role": "user", "content": "לכלכלה."}])
    )
    assert calls["n"] == 1
    assert _pushed_text(pushed).startswith("סבבה, הבנתי.")


@pytest.mark.asyncio
async def test_no_retry_when_a_tool_call_was_produced():
    # A transition alongside the parroted note: the destination's opening is
    # delivered deterministically by the engine, so nothing is owed here.
    chunks = _chunks(RUN_2139_ACK + RUN_2139_NOTE) + [
        _tool_call_chunk("transition_to_9", "{}", "call_1")
    ]
    service, pushed, calls = _make_service([chunks])
    await service._process_context(
        LLMContext(messages=[{"role": "user", "content": "לכלכלה."}])
    )
    assert calls["n"] == 1
    # Visible prose was generated, so the call is deferred until after TTS.
    assert len(service._pending_function_calls) == 1


@pytest.mark.asyncio
async def test_a_retry_that_parrots_again_gives_up():
    parrot_again = _chunks(["אוקיי. —\n"] + RUN_2139_NOTE)
    service, pushed, calls = _make_service(
        [_chunks(RUN_2139_ACK + RUN_2139_NOTE), parrot_again]
    )
    await service._process_context(
        LLMContext(messages=[{"role": "user", "content": "לכלכלה."}])
    )
    # One corrective retry, never a second: a model this stuck would loop.
    assert calls["n"] == 2
    assert "interrupted" not in _pushed_text(pushed)


@pytest.mark.asyncio
async def test_note_only_completion_rides_the_empty_retry_with_the_correction():
    service, pushed, calls = _make_service(
        [_chunks(RUN_2139_NOTE), _chunks(RETRY_QUESTION)]
    )
    await service._process_context(
        LLMContext(messages=[{"role": "user", "content": "לכלכלה."}])
    )
    text = _pushed_text(pushed)
    assert text.rstrip().endswith("?")
    assert "interrupted" not in text
    assert calls["n"] == 2
    # The empty-completion retry carried the correction in its no-prefix form.
    assert calls["pending"][1] == ""


@pytest.mark.asyncio
async def test_budget_exhaustion_skips_the_corrective_retry():
    # Each generation costs more than the whole per-turn budget: by the time
    # the swallowed question is detected there is no wall clock left to spend.
    from pipecat.services.openai import base_llm

    service, pushed, calls = _make_service(
        [_chunks(RUN_2139_ACK + RUN_2139_NOTE)],
        secs_per_generation=base_llm.EMPTY_RETRY_TOTAL_BUDGET_SECS + 1,
    )
    await service._process_context(
        LLMContext(messages=[{"role": "user", "content": "לכלכלה."}])
    )
    assert calls["n"] == 1
    assert _pushed_text(pushed).startswith("סבבה, הבנתי.")


class TestEphemeralUserNote:
    def test_merges_into_a_trailing_plain_user_turn(self):
        original_user = {"role": "user", "content": "לכלכלה."}
        params = {"messages": [{"role": "system", "content": "s"}, original_user]}
        _append_ephemeral_user_note(params, "(correction)")
        assert len(params["messages"]) == 2
        merged = params["messages"][-1]
        assert merged["role"] == "user"
        assert merged["content"] == "לכלכלה.\n\n(correction)"
        # Request-only: the dict may be shared with the live context, so the
        # merge must copy, never mutate.
        assert original_user["content"] == "לכלכלה."

    def test_appends_when_history_ends_in_a_tool_result(self):
        params = {
            "messages": [
                {"role": "user", "content": "כן"},
                {"role": "tool", "content": "{}", "tool_call_id": "t1"},
            ]
        }
        _append_ephemeral_user_note(params, "(correction)")
        assert params["messages"][-1] == {"role": "user", "content": "(correction)"}
        assert len(params["messages"]) == 3

    def test_appends_on_an_empty_history(self):
        params = {}
        _append_ephemeral_user_note(params, "(correction)")
        assert params["messages"] == [{"role": "user", "content": "(correction)"}]


@pytest.mark.asyncio
async def test_real_request_carries_the_merged_correction_and_consumes_it():
    """Through the REAL get_chat_completions: the correction reaches the wire
    merged into the trailing user turn (alternation-safe) and is consumed."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="test-model"))
    create = AsyncMock(return_value=_FakeStream([]))
    service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    service._marker_correction_pending = "סבבה, הבנתי. —"
    await service.get_chat_completions(
        LLMContext(messages=[{"role": "user", "content": "לכלכלה."}])
    )

    assert service._marker_correction_pending is None
    messages = create.call_args.kwargs["messages"]
    assert messages[-1]["role"] == "user"
    assert "לכלכלה." in messages[-1]["content"]
    assert "Internal correction" in messages[-1]["content"]
    assert "סבבה, הבנתי. —" in messages[-1]["content"]
    # Alternation stays template-valid: no two consecutive plain user turns.
    roles = [m.get("role") for m in messages]
    assert all(a != b or a != "user" for a, b in zip(roles, roles[1:]))


@pytest.mark.asyncio
async def test_real_request_without_pending_correction_is_untouched():
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="test-model"))
    create = AsyncMock(return_value=_FakeStream([]))
    service._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    await service.get_chat_completions(
        LLMContext(messages=[{"role": "user", "content": "לכלכלה."}])
    )

    messages = create.call_args.kwargs["messages"]
    assert messages[-1] == {"role": "user", "content": "לכלכלה."}
