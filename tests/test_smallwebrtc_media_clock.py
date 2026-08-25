#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the SmallWebRTC media-clock surfaces used by wire-fidelity capture.

Two additions are covered, both consumed by recorders that need to know WHEN
audio was really on the line rather than when Python happened to handle it:

1. ``RawAudioTrack.set_transmit_tap`` — an observer of the audio actually
   handed to the RTP sender, reporting each chunk's position on the outbound
   media clock. The tap must see exactly the transmitted audio: no
   auto-generated silence, and nothing that ``mark_pending_futures_done``
   discarded at connection close.

2. ``media_pts_ns`` on the frames yielded by
   ``SmallWebRTCClient.read_audio_frame`` — the capture instant on the
   sender's media clock in nanoseconds. The raw ``pts`` copied alongside it is
   in the source track's time_base and unreadable without it.
"""

import asyncio
import fractions
import unittest
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from av import AudioFrame

from pipecat.transports.smallwebrtc.transport import RawAudioTrack, SmallWebRTCClient

RATE = 16_000
BYTES_PER_10MS = RATE * 10 // 1000 * 2


class _TapRecorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, chunk, sample_rate, pts_ns, at):
        self.calls.append((chunk, sample_rate, pts_ns, at))


class TestRawAudioTrackTransmitTap(unittest.IsolatedAsyncioTestCase):
    async def test_tap_sees_queued_audio_on_the_media_clock(self):
        """Each transmitted chunk reports its grid position, not its pop time."""
        track = RawAudioTrack(sample_rate=RATE)
        tap = _TapRecorder()
        track.set_transmit_tap(tap)

        track.add_audio_bytes(b"\x01\x02" * (BYTES_PER_10MS // 2) * 3)  # 30ms
        for _ in range(3):
            await track.recv()

        self.assertEqual(len(tap.calls), 3)
        pts_values = [pts for _, _, pts, _ in tap.calls]
        self.assertEqual(pts_values, [0, 10_000_000, 20_000_000])
        for chunk, sample_rate, _, _ in tap.calls:
            self.assertEqual(len(chunk), BYTES_PER_10MS)
            self.assertEqual(sample_rate, RATE)

    async def test_auto_silence_never_reaches_the_tap(self):
        """Silence generated for an empty queue is filler, not transmitted speech."""
        track = RawAudioTrack(sample_rate=RATE)
        tap = _TapRecorder()
        track.set_transmit_tap(tap)

        await track.recv()  # queue empty -> auto silence
        self.assertEqual(tap.calls, [])

        # Real audio queued after the silence lands on the ADVANCED grid: the
        # silence consumed a 10ms slot, so the chunk transmits at pts=10ms.
        track.add_audio_bytes(b"\x01\x02" * (BYTES_PER_10MS // 2))
        await track.recv()
        self.assertEqual([pts for _, _, pts, _ in tap.calls], [10_000_000])

    async def test_discarded_audio_never_reaches_the_tap(self):
        """Audio cleared at connection close was never sent, so it is never reported.

        This is the teardown lie the tap exists to kill: the post-write frame
        push fires for these bytes (their futures resolve as done), but no
        listener ever heard them.
        """
        track = RawAudioTrack(sample_rate=RATE)
        tap = _TapRecorder()
        track.set_transmit_tap(tap)

        future = track.add_audio_bytes(b"\x01\x02" * (BYTES_PER_10MS // 2) * 4)
        track.mark_pending_futures_done()

        self.assertTrue(future.done())
        await track.recv()  # queue is empty now -> auto silence
        self.assertEqual(tap.calls, [])

    async def test_a_raising_tap_does_not_break_recv(self):
        track = RawAudioTrack(sample_rate=RATE)

        async def bad_tap(chunk, sample_rate, pts_ns, at):
            raise RuntimeError("boom")

        track.set_transmit_tap(bad_tap)
        track.add_audio_bytes(b"\x01\x02" * (BYTES_PER_10MS // 2))
        frame = await track.recv()
        self.assertEqual(frame.samples, RATE * 10 // 1000)


def _make_audio_self(track):
    fake = MagicMock()
    fake._audio_input_track = track
    fake._webrtc_connection = MagicMock()
    fake._webrtc_connection.is_connected.return_value = True
    fake._in_sample_rate = RATE
    fake._audio_in_channels = 1
    fake._audio_in_layout = "mono"
    # Passthrough resampler.
    fake._audio_in_resampler.resample.side_effect = lambda f: [f]
    return fake


def _audio_frame(pts, time_base):
    arr = np.zeros((1, 320), dtype=np.int16)
    f = AudioFrame.from_ndarray(arr, format="s16", layout="mono")
    f.sample_rate = RATE
    f.pts = pts
    if time_base is not None:
        f.time_base = time_base
    return f


async def _read_one(fake):
    gen = SmallWebRTCClient.read_audio_frame(fake)
    try:
        return await asyncio.wait_for(gen.__anext__(), timeout=2.0)
    finally:
        await gen.aclose()


class TestMediaPtsNs(unittest.IsolatedAsyncioTestCase):
    async def test_media_pts_ns_is_absolute_nanoseconds(self):
        """pts x time_base -> ns, independent of the source clock's rate."""
        track = MagicMock()
        track.recv = AsyncMock(
            side_effect=[_audio_frame(pts=960, time_base=fractions.Fraction(1, 48_000))]
        )
        frame = await _read_one(_make_audio_self(track))
        self.assertEqual(frame.media_pts_ns, 20_000_000)  # 960/48000 s

    async def test_no_pts_means_no_media_clock_claim(self):
        """A frame the source did not timestamp must not invent one."""
        track = MagicMock()
        track.recv = AsyncMock(side_effect=[_audio_frame(pts=None, time_base=None)])
        frame = await _read_one(_make_audio_self(track))
        self.assertFalse(hasattr(frame, "media_pts_ns"))


if __name__ == "__main__":
    unittest.main()
