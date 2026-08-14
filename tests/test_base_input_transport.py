#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for frame-based audio handling in :class:`BaseInputTransport`."""

import unittest
import asyncio
import warnings
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InputTransportStartAudioStreamingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams


class TestBaseInputTransportFrameAudio(unittest.IsolatedAsyncioTestCase):
    def _transport(self) -> BaseInputTransport:
        return BaseInputTransport(TransportParams(audio_in_enabled=True))

    async def test_incoming_audio_frame_routed_to_push_audio_frame(self):
        transport = self._transport()
        transport.push_audio_frame = AsyncMock()
        transport.push_frame = AsyncMock()
        frame = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
        await transport.process_frame(frame, FrameDirection.DOWNSTREAM)
        # Fed into the VAD path, not forwarded as a plain frame.
        transport.push_audio_frame.assert_called_once_with(frame)

    async def test_start_audio_streaming_frame_triggers_streaming(self):
        transport = self._transport()
        transport._start_audio_in_streaming = AsyncMock()
        await transport.process_frame(
            InputTransportStartAudioStreamingFrame(), FrameDirection.DOWNSTREAM
        )
        transport._start_audio_in_streaming.assert_called_once()

    async def test_start_audio_in_streaming_method_is_deprecated(self):
        transport = self._transport()
        transport._start_audio_in_streaming = AsyncMock()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await transport.start_audio_in_streaming()
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        transport._start_audio_in_streaming.assert_called_once()


class TestBaseInputTransportGapFill(unittest.IsolatedAsyncioTestCase):
    """Gap fill: synthesized silence while a DTX-style source stops sending."""

    def _transport(self, **params) -> BaseInputTransport:
        transport = BaseInputTransport(
            TransportParams(audio_in_enabled=True, audio_in_sample_rate=8000, **params)
        )
        transport._sample_rate = 8000
        transport.push_frame = AsyncMock()
        return transport

    async def _run_audio_task(self, transport, secs: float):
        # Drive the audio task handler directly with a shortened wall clock.
        task = asyncio.get_running_loop().create_task(transport._audio_task_handler())
        await asyncio.sleep(secs)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_gap_after_real_audio_is_filled_with_silence(self):
        transport = self._transport(audio_in_gap_fill_enabled=True)
        transport._audio_in_queue = asyncio.Queue()
        real = InputAudioRawFrame(audio=b"\x01\x02" * 160, sample_rate=8000, num_channels=1)
        await transport._audio_in_queue.put(real)

        await self._run_audio_task(transport, 0.35)

        frames = [call.args[0] for call in transport.push_frame.call_args_list]
        self.assertIs(frames[0], real)
        fill_frames = frames[1:]
        # ~0.35s with a 0.1s fill cadence: at least two synthesized frames.
        self.assertGreaterEqual(len(fill_frames), 2)
        for frame in fill_frames:
            self.assertIsInstance(frame, InputAudioRawFrame)
            self.assertEqual(frame.sample_rate, 8000)
            self.assertEqual(frame.audio, bytes(len(frame.audio)))
            self.assertEqual(len(frame.audio), 2 * 800)  # 0.1s of 16-bit mono @8kHz

    async def test_no_fill_before_first_real_audio(self):
        transport = self._transport(audio_in_gap_fill_enabled=True)
        transport._audio_in_queue = asyncio.Queue()

        await self._run_audio_task(transport, 0.35)

        transport.push_frame.assert_not_called()

    async def test_no_fill_when_disabled(self):
        transport = self._transport()
        transport._audio_in_queue = asyncio.Queue()
        real = InputAudioRawFrame(audio=b"\x01\x02" * 160, sample_rate=8000, num_channels=1)
        await transport._audio_in_queue.put(real)

        await self._run_audio_task(transport, 0.7)

        transport.push_frame.assert_called_once_with(real)


if __name__ == "__main__":
    unittest.main()
