"""MetaAsideGate must keep model asides-to-the-system out of the stream.

Run 2044 (wf39): Gemma-26B answered the caller's small talk correctly and then
appended its own reasoning as a labelled parenthetical — ``<br> *(Note to
system: … I should transition to the next stage using `transition_to_3`. The
opening sentence of the next stage will be played automatically.)*``. The
downstream sanitizers run per TTS chunk, so the aside's English tail was
stripped while its head was not: the caller heard the leftover ``<br>. *`` as
"br", and the whole aside was stored in the transcript and committed to the
LLM context. The aside is meta, never speech.

Pinned behaviors:
- normal text streams through immediately (no added latency);
- an aside is suppressed whole however the stream splits it, together with the
  Markdown emphasis wrapping it on either side;
- an unclosed aside is suppressed to end of stream and discarded at flush;
- an aside-only completion yields empty visible text, so the caller-side
  empty-completion retry regenerates a real reply;
- ordinary parentheses — including Hebrew ones and English prose asides with
  no label — pass through untouched.
"""

from pipecat.services.openai.base_llm import MetaAsideGate

RUN_2044_ASIDE = (
    "*(Note to system: The user responded with a casual greeting/question. "
    "According to instructions for the current stage, since they provided a "
    "non-hostile response, I should transition to the next stage using "
    "`transition_to_3`. The opening sentence of the next stage will be played "
    "automatically.)*"
)
RUN_2044_SPEECH = "מצוין, תודה! אז נחזור לרגע לעניין שלנו. "


def run_stream(chunks):
    """Feed chunks through a fresh gate; return (visible, suppressed_chars)."""
    gate = MetaAsideGate()
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
        gate = MetaAsideGate()
        assert gate.feed("היי, מדברת מיכל ") == "היי, מדברת מיכל "

    def test_ordinary_parenthesis_survives(self):
        out, suppressed = run_stream(['העלות היא מאה שקל (כולל מע"מ) לחודש'])
        assert out == 'העלות היא מאה שקל (כולל מע"מ) לחודש'
        assert suppressed == 0

    def test_unlabelled_english_parenthetical_survives(self):
        # No "label:" head — the gate is not a general English filter.
        out, suppressed = run_stream(["we open at nine (except on Fridays) sharp"])
        assert out == "we open at nine (except on Fridays) sharp"
        assert suppressed == 0

    def test_label_without_colon_survives(self):
        out, suppressed = run_stream(["please read the (note) attached"])
        assert out == "please read the (note) attached"
        assert suppressed == 0

    def test_long_ordinary_parenthesis_releases_before_it_closes(self):
        # Past the cue lookahead the "(" must be released even though the span
        # is still open — an open paren must never stall the stream.
        gate = MetaAsideGate()
        emitted = gate.feed("שלום (" + "א" * 200)
        assert emitted.startswith("שלום (")
        assert gate.suppressed_chars == 0

    def test_emphasis_without_an_aside_survives(self):
        out, suppressed = run_stream(["this is *important* today"])
        assert out == "this is *important* today"
        assert suppressed == 0

    def test_trailing_emphasis_at_stream_end_is_released(self):
        out, _ = run_stream(["נתראה *"])
        assert out == "נתראה *"

    def test_empty_feed_is_noop(self):
        gate = MetaAsideGate()
        assert gate.feed("") == ""
        assert gate.flush() == ""


class TestSuppression:
    def test_run_2044_aside_suppressed_whole(self):
        out, suppressed = run_stream([RUN_2044_SPEECH, RUN_2044_ASIDE])
        assert out == RUN_2044_SPEECH
        assert suppressed == len(RUN_2044_ASIDE)

    def test_run_2044_aside_suppressed_when_split_mid_span(self):
        # The prod split fell inside the aside, which is exactly what defeated
        # the per-chunk sanitizers.
        head, tail = RUN_2044_ASIDE[:90], RUN_2044_ASIDE[90:]
        out, _ = run_stream([RUN_2044_SPEECH + head, tail])
        assert out == RUN_2044_SPEECH

    def test_aside_suppressed_one_char_at_a_time(self):
        text = RUN_2044_SPEECH + RUN_2044_ASIDE
        out, _ = run_stream(list(text))
        assert out == RUN_2044_SPEECH

    def test_wrapper_emphasis_goes_with_the_aside(self):
        # Only the surrounding whitespace is left behind — never a "*" for TTS
        # to voice.
        out, _ = run_stream(["בסדר גמור. *(Note: internal only.)* "])
        assert out.strip() == "בסדר גמור."
        assert "*" not in out

    def test_run_663_shape_suppressed(self):
        # The 28-seconds-of-spoken-English incident, same labelled shape.
        aside = (
            "(Note: Since I cannot actually check a database, and per "
            "instructions, I should not invent information for the caller.)"
        )
        out, _ = run_stream(["רגע אחד. ", aside])
        assert out == "רגע אחד. "

    def test_casing_and_spacing_variants(self):
        for head in ("(NOTE TO SYSTEM:", "( note :", "(Internal note:", "(System:"):
            out, suppressed = run_stream(["טקסט ", head, " meta text here)"])
            assert out == "טקסט ", f"leaked for {head!r}"
            assert suppressed > 0

    def test_unclosed_aside_discarded_at_flush(self):
        out, suppressed = run_stream(["טוב. *(Note to system: the model was cut"])
        assert out == "טוב. "
        assert suppressed > 0

    def test_aside_only_completion_is_empty(self):
        out, _ = run_stream([RUN_2044_ASIDE])
        assert out == ""

    def test_speech_after_an_aside_still_streams(self):
        out, _ = run_stream(["היי. (Note: meta.) ", "איך אפשר לעזור?"])
        assert out == "היי.  איך אפשר לעזור?"


class TestHebrewAsides:
    """Hebrew-labelled asides are the same leak in the platform's own language."""

    def test_hebrew_system_note_is_suppressed(self):
        out, _ = run_stream(["בסדר גמור. *(הערה למערכת: יש לעבור לשלב הבא)*"])
        assert out == "בסדר גמור. "

    def test_hebrew_plain_note_is_suppressed(self):
        out, _ = run_stream(["(הערה: המתקשר נשמע מהוסס) נמשיך?"])
        assert out.strip() == "נמשיך?"

    def test_hebrew_thought_label_is_suppressed(self):
        out, _ = run_stream(["טוב. (מחשבה: כדאי לעבור לשאלה הבאה עכשיו)"])
        assert out == "טוב. "

    def test_hebrew_aside_split_across_chunks(self):
        out, _ = run_stream(["כן. *(הער", "ה למערכת: לעבור לשלב 3)* אז נתקדם."])
        assert out == "כן.  אז נתקדם."

    def test_ordinary_hebrew_parens_pass(self):
        out, suppressed = run_stream(["זה עולה (בערך) שישים שקלים"])
        assert out == "זה עולה (בערך) שישים שקלים"
        assert suppressed == 0

    def test_hebrew_parens_with_late_colon_pass(self):
        # A paren whose head is not a meta label must stream even with a colon
        # later in the sentence.
        out, suppressed = run_stream(["(כמו שאמרתי: נדבר מחר) בסדר?"])
        assert out == "(כמו שאמרתי: נדבר מחר) בסדר?"
        assert suppressed == 0
