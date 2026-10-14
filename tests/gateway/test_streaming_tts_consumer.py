"""Tests for the gateway streaming-TTS consumer and adapter contract (#60671).

No live audio, network, or TTS SDK calls: the streaming provider, adapter,
and event loop are all faked.  Covers the adapter contract defaults, the
consumer lifecycle (begin/write/finish/abort), fallback safety, duplicate
suppression, cancellation idempotency, and concurrent-turn isolation.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.platforms.base import AudioFormat, StreamingTTSHandle
from gateway.streaming_tts_consumer import StreamingTTSConsumer
from tools.tts_streaming import SentenceChunker


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeStreamer:
    """Fake streaming provider that yields deterministic PCM chunks.

    Chunk numbers follow the stream() call count — failed attempts count too — so
    a retried clause's successful chunks are numbered after the intervening calls."""

    def __init__(self, chunks_per_clause=3, fail_on_clause=None, sample_rate=24000, channels=1, sample_width=2):
        self.chunks_per_clause = chunks_per_clause
        self.fail_on_clause = fail_on_clause
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self._clause_count = 0

    def stream(self, text: str):
        self._clause_count += 1
        if self.fail_on_clause and self._clause_count >= self.fail_on_clause:
            raise RuntimeError(f"fake streamer failure on clause {self._clause_count}")
        for i in range(self.chunks_per_clause):
            yield f"chunk-{self._clause_count}-{i}".encode()


class FakeVoiceAdapter:
    """Fake adapter that accepts streaming TTS."""

    def __init__(self, name="fake-voice", supports=True, fail_after_write=False):
        self.name = name
        self._supports = supports
        self._fail_after_write = fail_after_write
        self.handle = None
        self.written_chunks: list[bytes] = []
        self.begin_count = 0
        self.finish_count = 0
        self.abort_count = 0

    def _should_auto_tts_for_chat(self, chat_id):
        return True

    def supports_streaming_tts(self, chat_id, audio_format):
        return self._supports

    async def begin_streaming_tts(self, chat_id, audio_format, metadata=None):
        self.begin_count += 1
        if not self._supports:
            return None
        self.handle = StreamingTTSHandle(chat_id=chat_id, audio_format=audio_format)
        return self.handle

    async def write_streaming_tts(self, handle, chunk):
        if self._fail_after_write and len(self.written_chunks) >= 2:
            raise RuntimeError("adapter write failure after partial output")
        self.written_chunks.append(chunk)
        if not handle.audible:
            handle.audible = True

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        self.finish_count += 1

    async def abort_streaming_tts(self, handle, error=None):
        self.abort_count += 1
        if handle:
            handle.aborted = True


class BlockingSecondChunkStreamer(FakeStreamer):
    """Yields one chunk immediately, then blocks before the remaining chunks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, chunks_per_clause=2, **kwargs)
        self.started = threading.Event()
        self.first_chunk_written = threading.Event()
        self.allow_remaining_chunks = threading.Event()
        self.finished = threading.Event()

    def stream(self, text: str):
        self.started.set()
        try:
            yield b"chunk-1-0"
            self.first_chunk_written.set()
            self.allow_remaining_chunks.wait(timeout=5.0)
            yield b"chunk-1-1"
        finally:
            self.finished.set()


def _make_consumer(adapter, chat_id, loop, streamer):
    """Build a StreamingTTSConsumer with pre-set internals for testing."""
    consumer = StreamingTTSConsumer.__new__(StreamingTTSConsumer)
    consumer._adapter = adapter
    consumer._chat_id = chat_id
    consumer._loop = loop
    consumer._metadata = None
    consumer._audio_format = AudioFormat(
        sample_rate=int(getattr(streamer, "sample_rate", 24000)) if streamer is not None else 24000,
        channels=int(getattr(streamer, "channels", 1)) if streamer is not None else 1,
        sample_width=int(getattr(streamer, "sample_width", 2)) if streamer is not None else 2,
    )
    consumer._streamer = streamer  # type: ignore[assignment]
    consumer._chunker = SentenceChunker()
    consumer._queue = queue.Queue(maxsize=256)
    consumer._handle = None
    consumer._completed = False
    consumer._partial = False
    consumer._aborted = False
    consumer._finished = False
    consumer._dropped = False
    consumer._suppress_whole_file = False
    consumer._task = None
    consumer._lock = threading.Lock()
    consumer._strip_markdown = None
    # 断点续播 stream-level budget (normally set in __init__, skipped here via __new__).
    consumer._consecutive_clause_failures = 0
    return consumer


def _run_test(coro_factory, timeout=10.0):
    """Run an async test in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            asyncio.wait_for(coro_factory(loop), timeout=timeout)
        )
    finally:
        loop.close()


@pytest.fixture
def gateway_tts_turn(monkeypatch, tmp_path):
    """Real agent delivery + real TurnRunner callback wiring; only the speech provider and PCM sink are fake."""
    from agent.agent_runtime_helpers import strip_think_blocks
    from agent.stream_delivery import StreamDeliveryMixin
    from gateway.config import StreamingConfig
    from gateway.run_turn_runner import TurnRunner
    from gateway.stream_consumer import StreamConsumerConfig
    from gateway.turn_context import TurnContext

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    class Agent(StreamDeliveryMixin):
        _strip_think_blocks = strip_think_blocks

    class Streamer(FakeStreamer):
        def __init__(self):
            super().__init__()
            self.clauses = []

        def stream(self, text):
            self.clauses.append(text)
            yield b"\x01\x00" * 480

    class Adapter(FakeVoiceAdapter):
        def __init__(self):
            super().__init__()
            self.heard = asyncio.Queue()

        async def write_streaming_tts(self, handle, chunk):
            await super().write_streaming_tts(handle, chunk)
            self.heard.put_nowait(chunk)

    def make(loop):
        streamer, adapter = Streamer(), Adapter()
        monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: streamer)
        tts = StreamingTTSConsumer(adapter, "voice", {}, loop)
        ctx = TurnContext(
            streaming_tts_consumer_holder=[tts], user_config={},
            resolve_display_setting=lambda *args: True, interim_assistant_messages_enabled=True,
            source=SimpleNamespace(platform=SimpleNamespace(value="realtime"), chat_id="voice"),
            _run_still_current=lambda: True,
        )
        runner = SimpleNamespace(
            config=SimpleNamespace(streaming=StreamingConfig()),
            _delivery_adapter_for=lambda source: adapter,
            _build_stream_consumer_config=lambda *args, **kwargs: (StreamConsumerConfig(), None),
        )
        _, delta, interim, _ = TurnRunner(runner, ctx)._setup_stream_consumer("realtime")
        agent = Agent()
        agent.stream_delta_callback, agent._stream_callback, agent.interim_assistant_callback = delta, None, interim
        return agent, tts, adapter, streamer

    return make


async def _speak_then_tool_result(agent, tts, adapter, streamer, before_tool, acknowledgment, result):
    """Invariant: the acknowledgment reaches PCM before the tool result exists, then the result follows once."""
    tts.start()
    try:
        await asyncio.to_thread(before_tool)
        await asyncio.wait_for(adapter.heard.get(), timeout=3)  # spoken during the tool pause, not after
        assert streamer.clauses == [acknowledgment]
        assert adapter.finish_count == 0 and not tts.done
        await asyncio.to_thread(agent.stream_delta_callback, None)
        await asyncio.to_thread(agent.stream_delta_callback, result)
        tts.finish()
        assert await tts.wait_complete(timeout=3)
        assert streamer.clauses == [acknowledgment, result]
        assert adapter.begin_count == adapter.finish_count == 1
    finally:
        tts.finish()
        await tts.wait_complete(timeout=3)


def test_tool_boundary_none_flushes_streamed_acknowledgment(gateway_tts_turn):
    """The ``None`` tool-boundary delta releases already-streamed speech without ending the audio stream."""
    async def run():
        agent, tts, adapter, streamer = gateway_tts_turn(asyncio.get_running_loop())
        acknowledgment, result = "I will check that.", "The result is available."

        def before_tool():
            assert agent._deliver_to_stream_callbacks(acknowledgment)
            agent._record_streamed_assistant_text(acknowledgment)
            agent._emit_interim_assistant_message({"role": "assistant", "content": acknowledgment})
            agent.stream_delta_callback(None)

        await _speak_then_tool_result(agent, tts, adapter, streamer, before_tool, acknowledgment, result)

    asyncio.run(run())


def test_completed_commentary_is_spoken_exactly_once(gateway_tts_turn):
    """Completed commentary (no text delta exists for it) reaches TTS once; the later interim dedupes."""
    async def run():
        agent, tts, adapter, streamer = gateway_tts_turn(asyncio.get_running_loop())
        acknowledgment, result = "I will check that.", "The result is available."

        def before_tool():
            agent._fire_streamed_codex_commentary(acknowledgment)
            agent._emit_interim_assistant_message({"role": "assistant", "content": "", "codex_message_items": [{
                "type": "message", "phase": "commentary",
                "content": [{"type": "output_text", "text": acknowledgment}],
            }]})
            agent.stream_delta_callback(None)

        await _speak_then_tool_result(agent, tts, adapter, streamer, before_tool, acknowledgment, result)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Adapter contract defaults (BasePlatformAdapter)
# ---------------------------------------------------------------------------


def _make_minimal_adapter():
    """Create a minimal concrete BasePlatformAdapter for testing defaults."""
    from gateway.platforms.base import BasePlatformAdapter

    class _Minimal(BasePlatformAdapter):
        async def send(self, chat_id, content, **kw):
            pass

        async def send_voice(self, chat_id, audio_path, **kw):
            pass

        async def connect(self, **kw):
            return True

        async def disconnect(self):
            pass

        async def get_chat_info(self, chat_id):
            return {}

    adapter = object.__new__(_Minimal)
    adapter._streaming_tts_completed_turns = set()
    return adapter


class TestAdapterContractDefaults:
    """Verify the default adapter reports unsupported and is source-compatible."""

    def test_supports_streaming_tts_defaults_false(self):
        adapter = _make_minimal_adapter()
        assert adapter.supports_streaming_tts("chat1", AudioFormat()) is False



# ---------------------------------------------------------------------------
# StreamingTTSConsumer lifecycle
# ---------------------------------------------------------------------------


class TestConsumerLifecycle:
    """Begin/write/finish lifecycle exactly once on success."""

    def test_successful_stream_produces_ordered_chunks(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=2)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("This is the first sentence. ")
            consumer.on_delta("Here is the second one. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is True
            assert adapter.begin_count == 1
            assert adapter.finish_count == 1
            assert adapter.abort_count == 0
            # 2 clauses * 2 chunks each = 4 chunks
            assert len(adapter.written_chunks) == 4
            # Verify ordering
            assert adapter.written_chunks[0] == b"chunk-1-0"
            assert adapter.written_chunks[1] == b"chunk-1-1"
            assert adapter.written_chunks[2] == b"chunk-2-0"
            assert adapter.written_chunks[3] == b"chunk-2-1"

        _run_test(run)




class TestStreamerFormatAndLooping:
    """Constructor wiring should derive format and keep provider I/O off-loop."""

    def test_audio_format_tracks_resolved_streamer(self):
        streamer = FakeStreamer(chunks_per_clause=1, sample_rate=48000, channels=2, sample_width=4)
        import tools.tts_streaming as tts_streaming
        original_resolve = tts_streaming.resolve_streaming_provider
        tts_streaming.resolve_streaming_provider = lambda *_args, **_kwargs: streamer
        loop = asyncio.new_event_loop()
        try:
            consumer = StreamingTTSConsumer(FakeVoiceAdapter(), "chat1", {}, loop)
            assert consumer._audio_format.sample_rate == 48000
            assert consumer._audio_format.channels == 2
            assert consumer._audio_format.sample_width == 4
        finally:
            tts_streaming.resolve_streaming_provider = original_resolve
            loop.close()

    def test_chunker_min_len_comes_from_tts_streaming_config(self):
        """The gateway consumer honours tts.streaming.min_len (#96927) instead of the class default."""
        import tools.tts_streaming as tts_streaming
        original_resolve = tts_streaming.resolve_streaming_provider
        tts_streaming.resolve_streaming_provider = lambda *_args, **_kwargs: None
        loop = asyncio.new_event_loop()
        try:
            consumer = StreamingTTSConsumer(FakeVoiceAdapter(), "chat1", {"streaming": {"min_len": 6}}, loop)
            assert consumer._chunker.min_len == 6
        finally:
            tts_streaming.resolve_streaming_provider = original_resolve
            loop.close()


class TestGatewayIntegrationSeam:
    """The actual adapter seam is per-turn, not chat-only."""

    def test_duplicate_suppression_is_per_turn(self):
        from gateway.platforms.base import streaming_tts_should_skip_whole_file

        adapter = _make_minimal_adapter()
        turn_one = adapter._streaming_tts_turn_key("chat-1", 101)
        turn_two = adapter._streaming_tts_turn_key("chat-1", 102)
        assert turn_one != turn_two

        adapter._mark_streaming_tts_completed_turn("chat-1", 101)
        assert streaming_tts_should_skip_whole_file(
            adapter._streaming_tts_completed_turns,
            "chat-1",
            101,
        ) is True
        assert streaming_tts_should_skip_whole_file(
            adapter._streaming_tts_completed_turns,
            "chat-1",
            102,
        ) is False
        assert adapter._streaming_tts_turn_completed("chat-1", 101) is True
        assert adapter._streaming_tts_turn_completed("chat-1", 102) is False
        assert adapter._streaming_tts_turn_completed("chat-2", 101) is False


class TestAbortAndCancellation:
    """Abort lifecycle: idempotent, prevents late chunks."""

    def test_abort_is_idempotent(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=10)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("A sentence. ")
            # Abort multiple times
            consumer.abort("test")
            consumer.abort("test2")
            consumer.abort("test3")
            consumer.finish()

            await consumer.wait_complete(timeout=5.0)
            # Abort should have been called at most once on the adapter
            assert adapter.abort_count <= 1

        _run_test(run)


class TestFallbackSafety:
    """Pre-audio failure falls back; post-audio failure does not replay."""

    def test_pre_audio_failure_falls_back(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(fail_on_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("A sentence. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            # Pre-audio failure: should NOT report completed (fall back); the adapter handle is
            # only opened on the first PCM chunk (#76466), so nothing was begun or aborted.
            assert completed is False
            assert consumer.suppress_whole_file is False
            assert adapter.begin_count == 0

        _run_test(run)


class TestConcurrentTurnIsolation:
    """Per-turn state is isolated across concurrent chats."""

    def test_two_concurrent_turns_do_not_cross_contaminate(self):
        async def run(loop):
            adapter1 = FakeVoiceAdapter(name="adapter1")
            adapter2 = FakeVoiceAdapter(name="adapter2")
            streamer = FakeStreamer(chunks_per_clause=2)

            c1 = _make_consumer(adapter1, "chat1", loop, streamer)
            c2 = _make_consumer(adapter2, "chat2", loop, streamer)

            c1.start()
            c2.start()

            c1.on_delta("Sentence for chat one. ")
            c2.on_delta("Sentence for chat two. ")

            c1.finish()
            c2.finish()

            await c1.wait_complete(timeout=5.0)
            await c2.wait_complete(timeout=5.0)

            # Each adapter should only have its own chunks
            assert adapter1.written_chunks != adapter2.written_chunks
            # Both should have completed
            assert c1.completed is True
            assert c2.completed is True

        _run_test(run)


class TestThinkBlockSuppression:
    """Think blocks split across deltas are never synthesised."""

    def test_think_blocks_not_synthesised(self):
        c = SentenceChunker()
        # Think block split across deltas — content inside is stripped.
        # The SentenceChunker uses min_len=20: sentences shorter than
        # 20 chars (after strip) are merged into the next one.
        assert c.feed("\x3cthink\x3esecret reasoning") == []
        # Feed a long enough sentence after the think block closes.
        result = c.feed(" about the answer.\x3c/think\x3e This is the actual spoken answer that is long enough. ")
        assert len(result) == 1
        assert "This is the actual spoken answer that is long enough." in result[0]
        assert c.flush() == []


class TestQueueBackpressure:
    """on_delta does not block when the queue is full."""

    def test_full_queue_drops_clause_not_blocks(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)
            # Tiny queue to trigger backpressure.
            consumer._queue = queue.Queue(maxsize=1)
            consumer._queue.put_nowait("prefilled")

            start = time.perf_counter()
            for i in range(50):
                consumer.on_delta(f"Sentence number {i}. ")
            elapsed = time.perf_counter() - start
            assert elapsed < 0.05

            consumer.finish()
            assert consumer.dropped is True
            completed = await consumer.wait_complete(timeout=1.0)
            assert completed is False
            assert consumer.completed is False
            assert consumer.suppress_whole_file is False

        _run_test(run, timeout=15.0)


# ---------------------------------------------------------------------------
# Finish race: _DONE sentinel guarantees the final clause is not lost (#60671)
# ---------------------------------------------------------------------------


class DelayedFlushChunker:
    """Chunker whose flush() returns a clause only after a signal is set.

    This simulates a provider that buffers the last clause and only
    releases it on flush(), after the drain loop has already started
    waiting.  The _DONE sentinel must arrive AFTER the flushed clause
    so the loop does not terminate early and lose the tail.
    """

    def __init__(self):
        self._flushed = threading.Event()
        self.allow_flush = threading.Event()

    def feed(self, delta: str):
        # Accumulate into a buffer; no sentences are released until flush.
        return []

    def flush(self):
        self._flushed.set()
        self.allow_flush.wait(timeout=5.0)
        return ["The final tail clause that must not be lost."]


class TestFinishSentinelRace:
    """The _DONE sentinel must not overtake a delayed flush clause."""

    def test_delayed_flush_clause_is_not_lost(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=2)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)
            # Replace the chunker with the delayed-flush variant.
            consumer._chunker = DelayedFlushChunker()

            consumer.start()
            consumer.on_delta("Some text that buffers inside the chunker. ")
            # Run finish() off-loop so the drain task can observe the exact
            # historical race: _finished is true while flush() is blocked and
            # the queue is still empty.
            finish_task = asyncio.create_task(asyncio.to_thread(consumer.finish))

            # Wait for flush() to block, then give the drain loop longer than
            # its queue.get timeout.  The old `_finished and queue.empty()`
            # escape hatch would terminate here and lose the tail clause.
            await asyncio.wait_for(
                asyncio.to_thread(consumer._chunker._flushed.wait, 2.0),
                timeout=2.0,
            )
            await asyncio.sleep(0.2)
            consumer._chunker.allow_flush.set()
            await finish_task

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is True
            # The tail clause must have been synthesised and written.
            assert len(adapter.written_chunks) > 0
            assert adapter.finish_count == 1

        _run_test(run)


# ---------------------------------------------------------------------------
# Adapter finish failure (#60671)
# ---------------------------------------------------------------------------


class FinishFailingAdapter(FakeVoiceAdapter):
    """Adapter whose finish_streaming_tts() always raises."""

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        raise RuntimeError("adapter finish failure")


# ---------------------------------------------------------------------------
# 断点续播: a clause whose provider connection dies mid-playback is re-synthesised
# over a fresh connection instead of killing the whole utterance
# ---------------------------------------------------------------------------


class FlakeyClauseStreamer(FakeStreamer):
    """Deterministic failure injection for resume tests.

    ``always_fail`` = endpoint dead: raises BEFORE any PCM (real connect failure).
    ``fail_once={clause text}`` = mid-stream drop: yields a partial chunk first,
    then raises (real behaviour: the player already heard the half)."""

    def __init__(self, fail_once=(), always_fail=False, **kwargs):
        super().__init__(**kwargs)
        self.fail_once = set(fail_once)
        self.always_fail = always_fail
        self.attempts = {}

    def stream(self, text):
        self.attempts[text] = self.attempts.get(text, 0) + 1
        if self.always_fail:
            raise RuntimeError(f"endpoint dead ({text!r})")
        if text in self.fail_once and self.attempts[text] == 1:
            yield b"partial-"
            raise RuntimeError(f"connection dropped mid-clause ({text!r})")
        yield from super().stream(text)


class TestClauseFailureResume:
    """Mid-playback clause failure: retry the clause, never the whole utterance."""

    def test_failed_clause_is_retried_and_story_continues(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FlakeyClauseStreamer(
                fail_once={"This is the second failing sentence."}, chunks_per_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("This is the first long sentence. ")
            consumer.on_delta("This is the second failing sentence. ")
            consumer.on_delta("This is the third long sentence. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is True
            assert adapter.abort_count == 0
            assert adapter.finish_count == 1
            # The failed clause was synthesised twice (fresh connection), the rest once.
            # (The chunker strips trailing spaces from clause keys.)
            assert streamer.attempts == {
                "This is the first long sentence.": 1,
                "This is the second failing sentence.": 2,
                "This is the third long sentence.": 1,
            }
            # Retry is requeued at the HEAD, so the failed clause plays in place; its
            # half-written PCM prefix stays (chunks played before the drop remain).
            # Failed attempts don't call super().stream() so they don't consume a chunk
            # number: first=1, retry(second)=2, third=3.
            assert adapter.written_chunks == [
                b"chunk-1-0", b"partial-", b"chunk-2-0", b"chunk-3-0",
            ]

        _run_test(run)

    def test_same_clause_failing_twice_is_skipped_and_story_continues(self):
        """In-clause retry budget exhausted while other audio plays: drop the clause, keep going."""
        async def run(loop):
            class _OneDeadClause(FlakeyClauseStreamer):
                def stream(self, text):
                    if text == "This is the always dead sentence.":
                        self.attempts[text] = self.attempts.get(text, 0) + 1
                        raise RuntimeError("dead clause")
                    yield from super().stream(text)

            adapter = FakeVoiceAdapter()
            streamer = _OneDeadClause(
                fail_once={"This is the flaky first sentence."}, chunks_per_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("This is the flaky first sentence. ")
            consumer.on_delta("This is the always dead sentence. ")
            consumer.on_delta("This is the final tail sentence. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is False  # a clause was dropped: not full completion
            assert adapter.abort_count == 0
            assert adapter.finish_count == 1
            assert consumer.suppress_whole_file is True  # no replay from the start
            assert streamer.attempts["This is the always dead sentence."] == 3  # 1 + 2 retries, then skipped
            assert streamer.attempts["This is the flaky first sentence."] == 2  # failed once, retried
            assert streamer.attempts["This is the final tail sentence."] == 1
            # Order (retry plays in place at the head): flaky-first's half (before the
            # drop), then flaky-first's full retry, then tail.  Failed attempts don't
            # call super().stream() so _clause_count only counts successes: retry=1, tail=2.
            assert adapter.written_chunks == [b"partial-", b"chunk-1-0", b"chunk-2-0"]

        _run_test(run)

    def test_dead_endpoint_pre_audio_falls_back_whole_file(self):
        """Endpoint dead from the start: retries + budget exhaust pre-audio -> whole-file fallback
        (legacy semantics), not a silent partial."""
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FlakeyClauseStreamer(always_fail=True, chunks_per_clause=1)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("This is the only dying sentence. ")
            consumer.on_delta("This is a second dying sentence. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is False
            assert consumer.suppress_whole_file is False  # whole-file TTS may replay
            assert adapter.begin_count == 0
            assert adapter.abort_count == 0
            # Endpoint dead pre-audio: clause 1 fails 3x (1 + 2 retries) with no PCM
            # ever written — legacy whole-file fallback; clause 2 is never synthesised
            # (drain loop exits, finally drains the queue).
            assert streamer.attempts["This is the only dying sentence."] == 3
            assert streamer.attempts.get("This is a second dying sentence.", 0) == 0

        _run_test(run)

    def test_dead_endpoint_mid_stream_drops_clause_and_keeps_playing(self):
        """Endpoint dies after audio is playing: retries exhaust, clause skipped, story continues
        to the end instead of aborting the utterance (the 西游记 断线事故 fix)."""
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = FakeStreamer(chunks_per_clause=1)
            # Clause 1 plays; clause 2's connection dies forever (always_fail from clause 2 on);
            # clause 3 plays.
            class _DieFromClause2(FakeStreamer):
                def __init__(self):
                    super().__init__(chunks_per_clause=1)
                    self.attempts = {}
                def stream(self, text):
                    if text.startswith("This is the second"):
                        self.attempts["second"] = self.attempts.get("second", 0) + 1
                        raise RuntimeError("WS connection timed out")
                    yield text.encode() + b"0"

            streamer2 = _DieFromClause2()
            consumer = _make_consumer(adapter, "chat1", loop, streamer2)

            consumer.start()
            consumer.on_delta("This is the first sentence here. ")
            consumer.on_delta("This is the second dying sentence. ")
            consumer.on_delta("This is the third sentence here. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is False  # clause 2 was dropped
            assert adapter.abort_count == 0  # the story was NOT aborted mid-air
            assert adapter.finish_count == 1
            assert consumer.suppress_whole_file is True
            assert consumer.partial is True
            # Clause 2 was attempted 3x (1 + 2 retries) then skipped; clauses 1 and 3
            # still played (the "西游记 断线事故" fix: no mid-air abort).  The stream()
            # text is stripped, so the written chunks carry no trailing space.
            assert streamer2.attempts["second"] == 3
            assert adapter.written_chunks == [
                b"This is the first sentence here.0",
                b"This is the third sentence here.0",
            ]

        _run_test(run)


class TestAdapterFinishFailure:
    """If finish_streaming_tts() raises, never report full completion."""

    def test_finish_failure_after_audible_reports_partial(self):
        async def run(loop):
            adapter = FinishFailingAdapter()
            streamer = FakeStreamer(chunks_per_clause=2)
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("A sentence that produces audio. ")
            consumer.finish()

            completed = await consumer.wait_complete(timeout=5.0)
            assert completed is False
            assert consumer.partial is True
            assert consumer.suppress_whole_file is True
            assert len(adapter.written_chunks) > 0

        _run_test(run)


# ---------------------------------------------------------------------------
# Post-audio timeout: clean abort, no later background completion (#60671)
# ---------------------------------------------------------------------------


class TestPostAudioTimeoutAbort:
    """On finalisation timeout after audible audio, abort the consumer."""

    def test_timeout_after_audible_aborts_and_preserves_suppression(self):
        async def run(loop):
            adapter = FakeVoiceAdapter()
            streamer = BlockingSecondChunkStreamer()
            consumer = _make_consumer(adapter, "chat1", loop, streamer)

            consumer.start()
            consumer.on_delta("This is a sentence with a delayed tail. ")
            consumer.finish()

            await asyncio.wait_for(
                asyncio.to_thread(streamer.first_chunk_written.wait, 2.0),
                timeout=2.0,
            )
            assert consumer.audible is True
            assert consumer.suppress_whole_file is True

            # Stale beyond the staleness window (grace=0): the consumer should be aborted, not
            # left running.  (The gateway default keeps grace while progressing; this test wants
            # the fixed staleness window.)
            completed = await consumer.wait_complete(timeout=0.01, grace=0.0)
            assert completed is False
            assert consumer.suppress_whole_file is True

            # Simulate the outer loop's abort-on-timeout behaviour.
            consumer.abort("streaming TTS finalisation timeout")
            await asyncio.sleep(0.05)

            # The consumer must not complete later in the background.
            assert consumer.completed is False
            assert consumer._aborted is True
            # Post-abort: progressing is False, so cleanup uses the fixed timeout.
            assert await consumer.wait_complete(timeout=0.5) is False
            # Release the blocking streamer so the drain task can observe the abort and exit
            # cleanly before the loop closes (no "Task was destroyed but it is pending").
            streamer.allow_remaining_chunks.set()
            for _ in range(50):
                if consumer.done:
                    break
                await asyncio.sleep(0.1)
            assert consumer.done is True

        _run_test(run)


# ---------------------------------------------------------------------------
# Real gateway regression: no _streaming_tts_consumer NameError (#60671)
# ---------------------------------------------------------------------------




class TestEndpointReportedRate:
    """Issue #76466: the adapter handle opens with the rate the provider learned from the
    endpoint's response, not the construction-time default."""

    def test_begin_uses_rate_learned_on_first_chunk(self):
        class _Learns(FakeStreamer):
            def stream(self, text):
                self.sample_rate = 44100
                yield from super().stream(text)

        async def run(loop):
            adapter = FakeVoiceAdapter()
            consumer = _make_consumer(adapter, "chat1", loop, _Learns(chunks_per_clause=2))
            assert consumer._audio_format.sample_rate == 24000  # provisional
            consumer.start()
            consumer.on_delta("A sentence. ")
            consumer.finish()
            assert await consumer.wait_complete(timeout=5.0) is True
            assert adapter.begin_count == 1
            assert adapter.handle.audio_format.sample_rate == 44100
            assert len(adapter.written_chunks) == 2

        _run_test(run)


# ---------------------------------------------------------------------------
# Progress-aware finalisation: slow streams are waited on, stuck ones give up
# ---------------------------------------------------------------------------


class _SlowPerClauseStreamer(FakeStreamer):
    """Synthesises each clause with a delay; stamps progress while working (as the real
    consumer stamps clause-fetch + PCM-write)."""

    def __init__(self, delay: float = 0.5, consumer: Any = None):
        super().__init__(chunks_per_clause=1)
        self.delay = delay
        self._consumer = consumer

    def stream(self, text: str):
        import time as _time
        if self._consumer is not None:
            self._consumer._last_progress = _time.monotonic()
        _time.sleep(self.delay)
        if self._consumer is not None:
            self._consumer._last_progress = _time.monotonic()
        yield b"pcm"


def test_slow_stream_is_waited_out_not_aborted():
    """A reply whose clauses take longer than the 10 s hard cap to synthesise must still be
    waited out to completion — this is the bug where long replies were cut to a few fragments
    (finalisation aborted after a fixed 10 s)."""
    async def run(loop):
        adapter = FakeVoiceAdapter()
        consumer = _make_consumer(adapter, "chat1", loop, None)
        streamer = _SlowPerClauseStreamer(delay=0.5, consumer=consumer)
        consumer._streamer = streamer
        consumer.start()
        # Each sentence is > 20 chars so the chunker (min_len=20) emits one clause per feed:
        # 5 clauses x 0.5 s = 2.5 s total, > the 1 s staleness window.
        for i in range(5):
            consumer.on_delta(f"Number {i} is the spoken sentence. ")
        consumer.finish()
        completed = await consumer.wait_complete(timeout=1.0)
        assert completed is True
        assert consumer.completed is True
        assert len(adapter.written_chunks) == 5

    _run_test(run, timeout=15.0)


def test_stuck_stream_gives_up_after_grace():
    """A consumer that never makes progress must still be given up on (bounded), so a dead
    provider cannot hold the turn forever."""
    async def run(loop):
        adapter = FakeVoiceAdapter()
        consumer = _make_consumer(adapter, "chat1", loop, FakeStreamer(chunks_per_clause=1))
        consumer.start()
        consumer.on_delta("One sentence. ")
        # Do NOT finish() and never make progress: the drain task waits on the queue.
        started = time.monotonic()
        completed = await consumer.wait_complete(timeout=0.5, grace=0.5)
        assert completed is False
        assert consumer.done is False
        assert time.monotonic() - started < 3.0
        consumer.abort("test cleanup")
        await consumer.wait_complete(timeout=1.0)
        assert consumer._aborted is True
        # Let the drain task drain the _ABORT sentinel and exit before the loop closes.
        for _ in range(50):
            if consumer.done:
                break
            await asyncio.sleep(0.1)
        assert consumer.done is True

    _run_test(run, timeout=15.0)
