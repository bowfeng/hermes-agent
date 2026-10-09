"""Tests for the Discord voice-channel streaming-TTS (speak-while-generating) path.

Covers the _StreamingVoiceSource frame geometry (20 ms / 48 kHz / stereo / s16le),
the _DiscordStreamingTTSHandle, and the DiscordAdapter handle protocol
(begin/write/finish/abort) against a fake VoiceClient + config gate.
No live audio, network, or TTS synthesis.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.platforms.base import AudioFormat
from plugins.platforms.discord.adapter import (
    DISCORD_AVAILABLE,
    _DiscordStreamingTTSHandle,
    _StreamingVoiceSource,
)

pytestmark = pytest.mark.skipif(not DISCORD_AVAILABLE, reason="discord.py not installed")


# ---------------------------------------------------------------------------
# _StreamingVoiceSource
# ---------------------------------------------------------------------------

def test_frame_geometry_is_discord_native():
    # 48000 Hz * 2 ch * 2 bytes * 20 ms == 3840 (== discord OpusEncoder.FRAME_SIZE)
    assert _StreamingVoiceSource.FRAME_SIZE == 3840


def test_push_read_roundtrip_exact_frames():
    src = _StreamingVoiceSource()
    pcm = bytes(range(256)) * 4800  # 1,228,800 bytes = 320 frames
    src.push(pcm)
    frames = []
    for _ in range(320):
        frames.append(src.read())
    assert all(len(f) == 3840 for f in frames)
    assert b"".join(frames) == pcm
    src.finish()
    assert src.read() == b""  # drained + done -> stop signal (no extra partial frame)


def test_finish_pads_tail_and_stops():
    src = _StreamingVoiceSource()
    src.push(b"\x01" * 100)  # 100 bytes: 0 full frames, 100-byte tail
    src.finish()
    f1 = src.read()
    assert len(f1) == 3840 and f1[:100] == b"\x01" * 100 and f1[100:] == b"\x00" * (3840 - 100)
    assert src.read() == b""  # drained + done -> stop signal


def test_push_after_finish_is_dropped():
    src = _StreamingVoiceSource()
    src.finish()
    src.push(b"\xff" * 8000)  # must not raise, must not re-arm
    assert src.read() == b""


def test_reader_blocks_bounded_then_wakes_on_push():
    src = _StreamingVoiceSource()
    src.push(b"\x02" * 3840)
    result = {}

    def reader():
        result["f"] = src.read()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t.join(timeout=3)
    assert not t.is_alive()
    assert result.get("f") == b"\x02" * 3840


def test_cleanup_is_idempotent_and_safe():
    src = _StreamingVoiceSource()
    src.push(b"\x03" * 3840)
    src.cleanup()
    src.cleanup()
    assert src.read() == b""


# ---------------------------------------------------------------------------
# _DiscordStreamingTTSHandle
# ---------------------------------------------------------------------------

def test_handle_carries_vc_state():
    src = _StreamingVoiceSource()
    h = _DiscordStreamingTTSHandle(
        chat_id="123", audio_format=AudioFormat(48000, 2, 2),
        _guild_id=999, _source=src, _done_event=None,
    )
    assert h.chat_id == "123"
    assert h._guild_id == 999
    assert h._source is src
    assert h.audible is False and h.aborted is False


# ---------------------------------------------------------------------------
# DiscordAdapter handle protocol (fake VoiceClient, config gate)
# ---------------------------------------------------------------------------

class _FakeReceiver:
    def __init__(self):
        self.paused = False
        self.resumed = 0

    def pause(self):
        self.paused = True

    def resume(self):
        self.resumed += 1


class _FakeVoiceClient:
    def __init__(self):
        self.played = []
        self.stopped = 0
        self.playing = False

    def is_connected(self):
        return True

    def is_playing(self):
        return self.playing

    def stop(self):
        self.stopped += 1
        self.playing = False

    def play(self, source, after=None):
        self.played.append(source)
        self.playing = True


def _make_adapter(monkeypatch, *, streaming_tts=True):
    import plugins.platforms.discord.adapter as mod
    adapter = object.__new__(mod.DiscordAdapter)
    adapter._voice_clients = {999: _FakeVoiceClient()}
    adapter._voice_text_channels = {999: 42}
    adapter._voice_receivers = {999: _FakeReceiver()}
    adapter._voice_streaming_sources = {}
    adapter._voice_timeout_tasks = {}
    monkeypatch.setattr(mod.DiscordAdapter, "_voice_streaming_enabled",
                        lambda self: streaming_tts)
    return adapter


def test_supports_requires_gate_and_48k_s16(monkeypatch):
    adapter = _make_adapter(monkeypatch, streaming_tts=False)
    assert adapter.supports_streaming_tts("42", AudioFormat(48000, 2, 2)) is False
    adapter = _make_adapter(monkeypatch, streaming_tts=True)
    assert adapter.supports_streaming_tts("42", AudioFormat(48000, 2, 2)) is True
    assert adapter.supports_streaming_tts("42", AudioFormat(48000, 1, 2)) is True
    # 24 kHz mono (default AudioFormat) must be declined -> whole-file fallback
    assert adapter.supports_streaming_tts("42", AudioFormat(24000, 1, 2)) is False
    # unknown chat (no /voice join binding) declined
    assert adapter.supports_streaming_tts("nope", AudioFormat(48000, 2, 2)) is True  # format ok...
    # ...but begin must decline
    assert asyncio.run(adapter.begin_streaming_tts("nope", AudioFormat(48000, 2, 2))) is None


def test_begin_write_finish_abort_full_cycle(monkeypatch):
    adapter = _make_adapter(monkeypatch, streaming_tts=True)
    vc = adapter._voice_clients[999]
    fmt = AudioFormat(48000, 2, 2)
    handle = asyncio.run(adapter.begin_streaming_tts("42", fmt))
    assert handle is not None and handle._guild_id == 999
    assert vc.played and isinstance(vc.played[-1], _StreamingVoiceSource)
    assert adapter._voice_receivers[999].paused is True

    # mono format exercises the L/R duplicator on write
    mono_handle = asyncio.run(adapter.begin_streaming_tts("42", AudioFormat(48000, 1, 2)))
    mono_write = b"\x01\x02\x03\x04" * 4800  # 19200 mono bytes -> 38400 stereo = 10 frames
    asyncio.run(adapter.write_streaming_tts(mono_handle, mono_write))
    src = mono_handle._source
    frames = []
    while len(frames) < 2:
        frames.append(src.read())
    # first 8 stereo bytes: each 2-byte mono sample duplicated to L and R
    # mono \x01\x02\x03\x04 -> stereo \x01\x02\x01\x02 \x03\x04\x03\x04
    assert frames[0][:8] == b"\x01\x02\x01\x02\x03\x04\x03\x04"

    # finish drains the stereo handle; abort the mono one (idempotent + resumes receiver).
    asyncio.run(adapter.finish_streaming_tts(handle))
    asyncio.run(adapter.abort_streaming_tts(mono_handle, error="test"))
    assert mono_handle.aborted is True
    assert adapter._voice_receivers[999].resumed >= 1


def test_write_on_aborted_handle_is_noop(monkeypatch):
    adapter = _make_adapter(monkeypatch, streaming_tts=True)
    handle = asyncio.run(adapter.begin_streaming_tts("42", AudioFormat(48000, 2, 2)))
    asyncio.run(adapter.abort_streaming_tts(handle, error="x"))
    asyncio.run(adapter.write_streaming_tts(handle, b"\x00" * 3840))  # must not raise
    assert handle.aborted is True
