"""InterruptionMarkerGate must keep parroted interruption notes out of the stream.

Run 2022 (wf39): after a genuine barge-in, the engine patched the previous
assistant context message with its bracketed reconciliation note. On the next
turn Gemma-26B parroted the pattern — its reply was an apology followed by a
verbatim ``[interrupted by the caller; the caller did NOT hear the rest: "…"]``
copy carrying the question it should have asked plainly — and the TTS path
read the note aloud to the caller. The note is context-only metadata; it is
never legitimate model output, so the gate suppresses any ``[interrupted``-
opened span from the visible stream.

Pinned behaviors:
- normal text streams through immediately (no added latency);
- a note is suppressed whole in any casing, however the stream splits it,
  including a chunk-final partial opener ("[interr");
- an unclosed note (generation cancelled mid-span, or never terminated) is
  suppressed to end of stream and discarded at flush;
- a note-only completion yields empty visible text, so the caller-side
  empty-completion retry regenerates a real reply;
- ordinary bracketed text ("[in progress]", Hebrew stage directions) passes
  through untouched — only ``[interrupted``-opened spans gate.
"""

from pipecat.services.openai.base_llm import InterruptionMarkerGate

RUN_2022_NOTE = (
    '[interrupted by the caller; the caller did NOT hear the rest: '
    '"אולי תגידי לי פשוט כמה חשוב לך שהאדם שציינת יהיה ראש ממשלה, '
    'בסולם של אחת עד חמש? "]'
)


def run_stream(chunks):
    """Feed chunks through a fresh gate; return (visible, suppressed_chars)."""
    gate = InterruptionMarkerGate()
    out = "".join(gate.feed(c) for c in chunks)
    out += gate.flush()
    assert out == gate.visible_text
    return out, gate.suppressed_chars


class TestPassThrough:
    def test_plain_hebrew_streams_unchanged(self):
        out, suppressed = run_stream(["שלום, ", "מה שלומך היום?"])
        assert out == "שלום, מה שלומך היום?"
        assert suppressed == 0

    def test_no_holding_latency_on_normal_text(self):
        gate = InterruptionMarkerGate()
        assert gate.feed("היי, מדבר עידן ") == "היי, מדבר עידן "

    def test_hebrew_bracketed_aside_untouched(self):
        out, suppressed = run_stream(["אמרתי [בצחוק] ", "שזה בסדר"])
        assert out == "אמרתי [בצחוק] שזה בסדר"
        assert suppressed == 0

    def test_bracket_word_sharing_opener_prefix_survives(self):
        # "[in progress]" shares the "[in" head with the opener: it may be
        # held one delta, but must come out untouched.
        out, suppressed = run_stream(["the task is [in", " progress] now"])
        assert out == "the task is [in progress] now"
        assert suppressed == 0

    def test_partial_opener_at_stream_end_is_released(self):
        out, _ = run_stream(["נתראה [interr"])
        assert out == "נתראה [interr"

    def test_empty_feed_is_noop(self):
        gate = InterruptionMarkerGate()
        assert gate.feed("") == ""
        assert gate.flush() == ""


class TestSuppression:
    def test_run_2022_parroted_note_suppressed(self):
        prefix = 'סליחה, לא הבנתי למה התכוונת ב"סדרה הזה". —\n'
        out, suppressed = run_stream([prefix, RUN_2022_NOTE])
        assert out == prefix
        assert suppressed == len(RUN_2022_NOTE)

    def test_note_split_per_token_leaks_nothing(self):
        text = "בסדר גמור. " + RUN_2022_NOTE + " נמשיך?"
        for size in (1, 2, 3, 7):
            chunks = [text[i : i + size] for i in range(0, len(text), size)]
            out, suppressed = run_stream(chunks)
            assert out == "בסדר גמור. " + " נמשיך?"
            assert suppressed == len(RUN_2022_NOTE)

    def test_hybrid_fabricated_variant_suppressed(self):
        # Run 2022's earlier fabrication mixed the two note templates — the
        # gate keys on the opener alone, not on either exact template.
        out, _ = run_stream(
            ['[interrupted by the caller; the caller did NOT hear: "זה בסדר גמור אם"]']
        )
        assert out == ""

    def test_any_casing_suppressed(self):
        out, _ = run_stream(["טוב. ", "[Interrupted by the caller; …] ", "נמשיך"])
        assert out == "טוב.  נמשיך"

    def test_unterminated_note_discarded_at_flush(self):
        # Generation cancelled mid-note (the run 2022 T5 shape): nothing of
        # the note may leak, text before it survives.
        out, suppressed = run_stream(
            ["זה ממש לא חייב, ", '[interrupted by the caller; the caller did NOT hear: "זה בסדר']
        )
        assert out == "זה ממש לא חייב, "
        assert suppressed > 0

    def test_note_only_completion_is_empty(self):
        out, _ = run_stream([RUN_2022_NOTE])
        assert out == ""

    def test_text_resumes_after_closed_note(self):
        out, _ = run_stream(["לפני ", '[interrupted x; y "z"]', " אחרי"])
        assert out == "לפני  אחרי"

    def test_two_notes_one_stream(self):
        out, _ = run_stream(
            ["א ", "[interrupted note one] ", "ב ", "[interrupted note two]", " ג"]
        )
        assert out == "א  ב  ג"

    def test_chunk_final_partial_opener_then_note(self):
        out, suppressed = run_stream(
            ["מעולה. [interr", 'upted by the caller; note "q"] ', "נמשיך?"]
        )
        assert out == "מעולה.  נמשיך?"
        assert suppressed > 0

    def test_suppressed_preview_captures_note_start(self):
        gate = InterruptionMarkerGate()
        gate.feed("היי ")
        gate.feed(RUN_2022_NOTE)
        gate.flush()
        assert gate.suppressed_preview.startswith("[interrupted by the caller")


class TestSplitInvariance:
    def test_split_invariance_fuzz(self):
        # The visible result must not depend on how the stream is chunked.
        text = 'טוב מאוד. —\n' + RUN_2022_NOTE + " ומה עכשיו? [in progress]"
        expected_visible = "טוב מאוד. —\n" + " ומה עכשיו? [in progress]"
        for size in (1, 2, 4, 5, 9, 16, 33, len(text)):
            chunks = [text[i : i + size] for i in range(0, len(text), size)]
            out, suppressed = run_stream(chunks)
            assert out == expected_visible, f"split size {size}"
            assert suppressed == len(RUN_2022_NOTE)
