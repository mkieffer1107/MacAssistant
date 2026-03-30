from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from unittest.mock import patch
from pathlib import Path

import numpy as np


RUNTIME_DIR = Path(__file__).resolve().parents[1]
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

import agent_runtime_host as runtime_host

HAS_MLX_AUDIO = importlib.util.find_spec("mlx_audio") is not None


class TestRuntimeHost(runtime_host.RuntimeHost):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, object]] = []

    def emit(self, payload: dict[str, object]) -> None:
        self.events.append(dict(payload))

    async def warm_model(self, model_id: str, arguments: dict[str, object]) -> None:
        return None

    def require_model(self, model_id: str, model: object) -> None:
        return None

    async def refresh_tool_definitions(self, force: bool = False) -> None:
        return None


def assistant_segments(events: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered: list[dict[str, object]] = []
    by_id: dict[str, dict[str, object]] = {}
    for event in events:
        if event.get("type") != "assistant_delta":
            continue
        segment_id = str(event.get("assistantSegmentID") or event.get("turnID"))
        segment = by_id.get(segment_id)
        if segment is None:
            segment = {
                "segment_id": segment_id,
                "text": "",
                "is_final": None,
            }
            by_id[segment_id] = segment
            ordered.append(segment)
        segment["text"] += str(event.get("text", ""))
        if "isFinalSegment" in event:
            segment["is_final"] = bool(event["isFinalSegment"])
    return ordered


class BufferedPreviewSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_continuous_audio_emits_partial_before_finish(self) -> None:
        partials: list[str] = []
        errors: list[str] = []

        async def preview_transcribe(pcm_bytes: bytes) -> str:
            await asyncio.sleep(0.01)
            return f"preview-{len(pcm_bytes)}"

        async def final_transcribe(pcm_bytes: bytes) -> str:
            return f"final-{len(pcm_bytes)}"

        session = runtime_host.BufferedPreviewSTTSession(
            turn_id="turn-1",
            preview_transcribe=preview_transcribe,
            final_transcribe=final_transcribe,
            emit_partial=partials.append,
            emit_error=errors.append,
            min_preview_pcm_bytes=2,
            preview_poll_interval_s=0.005,
        )
        try:
            for _ in range(5):
                session.append_pcm_bytes(b"\x00\x01")
                await asyncio.sleep(0.003)

            await asyncio.sleep(0.06)
            self.assertGreaterEqual(len(partials), 1)
            self.assertEqual(errors, [])

            final_text = await session.finish()
            self.assertEqual(final_text, "final-10")
        finally:
            session.cancel()


class RuntimeHostStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def wait_for_condition(
        self,
        predicate,
        *,
        timeout: float = 0.5,
        interval: float = 0.005,
    ) -> None:
        async def waiter() -> None:
            while not predicate():
                await asyncio.sleep(interval)

        await asyncio.wait_for(waiter(), timeout=timeout)

    async def test_stream_assistant_text_preserves_markdown_whitespace(self) -> None:
        class RecordingHost(runtime_host.RuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.events: list[dict[str, object]] = []

            def emit(self, payload: dict[str, object]) -> None:
                self.events.append(payload)

        host = RecordingHost()
        markdown = (
            "Here's a breakdown:\n\n"
            "- **CPU:** 10 cores\n"
            "- **RAM:** 64 GB\n\n"
            "```swift\n"
            "print(\"hello\")\n"
            "```\n"
        )

        with mock.patch.object(runtime_host.asyncio, "sleep", new=mock.AsyncMock()):
            await host.stream_assistant_text("turn-1", markdown)

        assistant_deltas = [
            event["text"]
            for event in host.events
            if event.get("type") == "assistant_delta"
        ]

        self.assertEqual("".join(assistant_deltas), markdown)

    async def test_cancel_turn_stops_further_stream_output_and_finishes_once(self) -> None:
        class StreamingHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.agent_model = object()
                self.agent_processor = object()
                self.tts_model = object()

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                return runtime_host.AssistantStepResult(
                    text="Hello world from the autonomous loop",
                    tool_calls=[],
                )

        host = StreamingHost()
        host.model_states["agent_model"] = runtime_host.ModelState(
            installed=True,
            warm=True,
            path="/tmp/agent",
            last_error=None,
        )
        host.model_states["tts_model"] = runtime_host.ModelState(
            installed=True,
            warm=True,
            path="/tmp/tts",
            last_error=None,
        )
        agent_model = host.agent_model
        tts_model = host.tts_model

        await host.handle_command(
            {
                "type": "send_text",
                "turnID": "turn-1",
                "text": "Hello",
                "speakReply": False,
            }
        )
        await self.wait_for_condition(
            lambda: len(
                [
                    event
                    for event in host.events
                    if event.get("type") == "assistant_delta"
                ]
            )
            >= 2
        )

        await host.handle_command({"type": "cancel", "turnID": "turn-1"})
        await self.wait_for_condition(
            lambda: any(
                event.get("type") == "turn_finished"
                and event.get("turnID") == "turn-1"
                for event in host.events
            )
        )

        assistant_delta_count = len(
            [
                event
                for event in host.events
                if event.get("type") == "assistant_delta"
            ]
        )
        await asyncio.sleep(0.05)

        self.assertEqual(
            len(
                [
                    event
                    for event in host.events
                    if event.get("type") == "assistant_delta"
                ]
            ),
            assistant_delta_count,
        )
        self.assertLess(assistant_delta_count, 7)
        self.assertEqual(
            [
                event
                for event in host.events
                if event.get("type") == "turn_finished"
                and event.get("turnID") == "turn-1"
            ],
            [
                {
                    "type": "turn_finished",
                    "turnID": "turn-1",
                    "status": "cancelled",
                    "isFinal": True,
                }
            ],
        )
        self.assertIs(host.agent_model, agent_model)
        self.assertIs(host.tts_model, tts_model)
        self.assertTrue(host.model_states["agent_model"].warm)
        self.assertTrue(host.model_states["tts_model"].warm)

    async def test_cancel_turn_suppresses_followup_tool_output(self) -> None:
        class ToolHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.agent_model = object()
                self.agent_processor = object()

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                return runtime_host.AssistantStepResult(
                    text="",
                    tool_calls=[
                        runtime_host.ToolCallRequest(
                            name="get_scripting_tips",
                            arguments={"search_term": "Finder"},
                        )
                    ],
                )

            async def execute_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> str:
                await asyncio.sleep(0.05)
                return "Tool result"

        host = ToolHost()

        await host.handle_command(
            {
                "type": "send_text",
                "turnID": "turn-tool",
                "text": "Help with Finder",
                "speakReply": False,
            }
        )
        await self.wait_for_condition(
            lambda: any(event.get("type") == "tool_started" for event in host.events)
        )

        await host.handle_command({"type": "cancel", "turnID": "turn-tool"})
        await self.wait_for_condition(
            lambda: any(
                event.get("type") == "turn_finished"
                and event.get("turnID") == "turn-tool"
                for event in host.events
            )
        )
        await asyncio.sleep(0.05)

        self.assertFalse(
            any(event.get("type") == "tool_output" for event in host.events)
        )
        self.assertFalse(
            any(event.get("type") == "assistant_delta" for event in host.events)
        )
        self.assertEqual(
            [
                event
                for event in host.events
                if event.get("type") == "turn_finished"
                and event.get("turnID") == "turn-tool"
            ],
            [
                {
                    "type": "turn_finished",
                    "turnID": "turn-tool",
                    "status": "cancelled",
                    "isFinal": True,
                }
            ],
        )

    async def test_cancel_turn_stops_followup_audio_chunks(self) -> None:
        class SpeakingHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.agent_model = object()
                self.agent_processor = object()
                self.tts_model = object()

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                return runtime_host.AssistantStepResult(text="Hello", tool_calls=[])

            def iter_tts_pcm_chunks(
                self,
                *,
                text: str,
                voice_preset: str,
                cancel_event: threading.Event | None = None,
            ):
                for index in range(3):
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    yield f"chunk-{index}".encode("utf-8")
                    time.sleep(0.03)

        host = SpeakingHost()

        await host.handle_command(
            {
                "type": "send_text",
                "turnID": "turn-audio",
                "text": "Say hello",
                "speakReply": True,
            }
        )
        await self.wait_for_condition(
            lambda: any(event.get("type") == "audio_chunk" for event in host.events)
        )

        await host.handle_command({"type": "cancel", "turnID": "turn-audio"})
        await self.wait_for_condition(
            lambda: any(
                event.get("type") == "turn_finished"
                and event.get("turnID") == "turn-audio"
                for event in host.events
            )
        )

        audio_chunk_count = len(
            [event for event in host.events if event.get("type") == "audio_chunk"]
        )
        await asyncio.sleep(0.05)

        self.assertEqual(
            len(
                [event for event in host.events if event.get("type") == "audio_chunk"]
            ),
            audio_chunk_count,
        )
        self.assertLess(audio_chunk_count, 3)
        self.assertEqual(
            [
                event
                for event in host.events
                if event.get("type") == "turn_finished"
                and event.get("turnID") == "turn-audio"
            ],
            [
                {
                    "type": "turn_finished",
                    "turnID": "turn-audio",
                    "status": "cancelled",
                    "isFinal": True,
                }
            ],
        )

    async def test_speak_text_emits_streamed_pcm_chunks_with_metadata(self) -> None:
        class SpeakingHost(runtime_host.RuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.events: list[dict[str, object]] = []
                self.tts_model = object()

            def emit(self, payload: dict[str, object]) -> None:
                self.events.append(dict(payload))

            async def warm_model(self, model_id: str, arguments: dict[str, object]) -> None:
                return

            def require_model(self, model_id: str, model: object) -> None:
                return

            def iter_tts_pcm_chunks(
                self,
                *,
                text: str,
                voice_preset: str,
                cancel_event: threading.Event | None = None,
            ):
                yield b"a"
                yield b"b"
                yield b"c"

        host = SpeakingHost()
        await host.handle_command(
            {
                "type": "speak_text",
                "turnID": "turn-stream",
                "text": "Hello there",
                "arguments": {"voice_preset": "casual_male"},
            }
        )
        await self.wait_for_condition(
            lambda: any(
                event.get("type") == "turn_finished"
                and event.get("turnID") == "turn-stream"
                for event in host.events
            )
        )

        audio_events = [
            event for event in host.events if event.get("type") == "audio_chunk"
        ]
        self.assertEqual(len(audio_events), 3)
        self.assertEqual(
            [base64.b64decode(event["chunkBase64"]) for event in audio_events],
            [b"a", b"b", b"c"],
        )
        self.assertTrue(all(event.get("turnID") == "turn-stream" for event in audio_events))
        self.assertTrue(all(event.get("sampleRate") == 24_000 for event in audio_events))
        self.assertTrue(all(event.get("channels") == 1 for event in audio_events))
        self.assertTrue(all(event.get("audioEncoding") == "pcm_s16le" for event in audio_events))

    async def test_send_text_with_spoken_reply_uses_streamed_pcm_chunk_path(self) -> None:
        class ReplyHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.tts_model = object()

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                return runtime_host.AssistantStepResult(text="Hello", tool_calls=[])

            def iter_tts_pcm_chunks(
                self,
                *,
                text: str,
                voice_preset: str,
                cancel_event: threading.Event | None = None,
            ):
                yield b"one"
                yield b"two"

        host = ReplyHost()
        await host.handle_command(
            {
                "type": "send_text",
                "turnID": "turn-reply-stream",
                "text": "Say hello",
                "speakReply": True,
                "arguments": {"voice_preset": "casual_male"},
            }
        )
        await self.wait_for_condition(
            lambda: any(
                event.get("type") == "turn_finished"
                and event.get("turnID") == "turn-reply-stream"
                for event in host.events
            )
        )

        audio_events = [
            event for event in host.events if event.get("type") == "audio_chunk"
        ]
        self.assertEqual(len(audio_events), 2)
        self.assertEqual(
            [base64.b64decode(event["chunkBase64"]) for event in audio_events],
            [b"one", b"two"],
        )

    @unittest.skipUnless(HAS_MLX_AUDIO, "mlx_audio is required for tokenizer patch coverage.")
    def test_patch_tts_runtime_replaces_broken_tokenizer_with_tekken_shim(self) -> None:
        class BrokenTokenizer:
            def encode(self, text: str) -> list[int]:
                return [1]

            def encode_speech_request(self, request: object) -> object:
                raise AssertionError(
                    "voice_num_audio_tokens must be set in audio config to use voice-based speech requests"
                )

        class ShimTokenizer:
            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

        class FakeModel:
            def __init__(self) -> None:
                self.tokenizer = BrokenTokenizer()
                self.sync_calls = 0

            def _sync_prompt_token_ids_from_tokenizer(self) -> None:
                self.sync_calls += 1

        host = runtime_host.RuntimeHost()
        model = FakeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            (model_path / "tekken.json").write_text(json.dumps({"audio": {}}), encoding="utf-8")
            shim_tokenizer = ShimTokenizer()
            with patch(
                "mlx_audio.tts.models.voxtral_tts.tekken.TekkenTokenizer.from_file",
                return_value=shim_tokenizer,
            ) as mock_from_file:
                host.patch_tts_runtime(model, model_path)

        self.assertIs(model.tokenizer, shim_tokenizer)
        self.assertEqual(model.sync_calls, 1)
        mock_from_file.assert_called_once()

    @unittest.skipUnless(HAS_MLX_AUDIO, "mlx_audio is required for tokenizer recovery coverage.")
    async def test_speak_text_recovers_when_loaded_tts_tokenizer_lacks_voice_metadata(self) -> None:
        class BrokenTokenizer:
            def encode(self, text: str) -> list[int]:
                return [1]

            def encode_speech_request(self, request: object) -> object:
                raise AssertionError(
                    "voice_num_audio_tokens must be set in audio config to use voice-based speech requests"
                )

        class ShimTokenizer:
            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

        class FakeAudioResult:
            def __init__(self) -> None:
                self.audio = [0.0]

        class FakeTTSModel:
            def __init__(self) -> None:
                self.tokenizer = BrokenTokenizer()
                self.sync_calls = 0
                self.used_fallback_path = False

            def _sync_prompt_token_ids_from_tokenizer(self) -> None:
                self.sync_calls += 1

            def generate(self, text: str, voice: str):
                if hasattr(self.tokenizer, "encode_speech_request"):
                    self.tokenizer.encode_speech_request(object())
                else:
                    self.used_fallback_path = True
                yield FakeAudioResult()

        class SpeakingHost(TestRuntimeHost):
            def __init__(self, model_path: Path) -> None:
                super().__init__()
                self.tts_model = FakeTTSModel()
                self.model_states["tts_model"] = runtime_host.ModelState(
                    installed=True,
                    warm=False,
                    path=str(model_path),
                    last_error=None,
                )

            async def warm_model(self, model_id: str, arguments: dict[str, object]) -> None:
                self.patch_tts_runtime(self.tts_model, Path(self.model_states["tts_model"].path))

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                return runtime_host.AssistantStepResult(text="Hello", tool_calls=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            (model_path / "tekken.json").write_text(json.dumps({"audio": {}}), encoding="utf-8")
            shim_tokenizer = ShimTokenizer()
            host = SpeakingHost(model_path)

            with patch(
                "mlx_audio.tts.models.voxtral_tts.tekken.TekkenTokenizer.from_file",
                return_value=shim_tokenizer,
            ):
                await host.handle_command(
                    {
                        "type": "speak_text",
                        "turnID": "turn-speak-text",
                        "text": "Hello there",
                        "arguments": {"voice_preset": "casual_male"},
                    }
                )
                await self.wait_for_condition(
                    lambda: any(event.get("type") == "turn_finished" for event in host.events)
                )

        self.assertTrue(
            any(event.get("type") == "audio_chunk" for event in host.events)
        )
        self.assertFalse(
            any(event.get("type") == "error" for event in host.events)
        )
        self.assertIs(host.tts_model.tokenizer, shim_tokenizer)
        self.assertEqual(host.tts_model.sync_calls, 1)
        self.assertTrue(host.tts_model.used_fallback_path)

    @unittest.skipUnless(HAS_MLX_AUDIO, "mlx_audio is required for tokenizer recovery coverage.")
    async def test_send_text_with_spoken_reply_recovers_when_loaded_tts_tokenizer_lacks_voice_metadata(self) -> None:
        class BrokenTokenizer:
            def encode(self, text: str) -> list[int]:
                return [1]

            def encode_speech_request(self, request: object) -> object:
                raise AssertionError(
                    "voice_num_audio_tokens must be set in audio config to use voice-based speech requests"
                )

        class ShimTokenizer:
            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

        class FakeAudioResult:
            def __init__(self) -> None:
                self.audio = [0.0]

        class FakeTTSModel:
            def __init__(self) -> None:
                self.tokenizer = BrokenTokenizer()
                self.sync_calls = 0
                self.used_fallback_path = False

            def _sync_prompt_token_ids_from_tokenizer(self) -> None:
                self.sync_calls += 1

            def generate(self, text: str, voice: str):
                if hasattr(self.tokenizer, "encode_speech_request"):
                    self.tokenizer.encode_speech_request(object())
                else:
                    self.used_fallback_path = True
                yield FakeAudioResult()

        class ReplyHost(TestRuntimeHost):
            def __init__(self, model_path: Path) -> None:
                super().__init__()
                self.tts_model = FakeTTSModel()
                self.model_states["tts_model"] = runtime_host.ModelState(
                    installed=True,
                    warm=False,
                    path=str(model_path),
                    last_error=None,
                )

            async def warm_model(self, model_id: str, arguments: dict[str, object]) -> None:
                if model_id == "tts_model":
                    self.patch_tts_runtime(self.tts_model, Path(self.model_states["tts_model"].path))

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                return runtime_host.AssistantStepResult(text="Hello", tool_calls=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            (model_path / "tekken.json").write_text(json.dumps({"audio": {}}), encoding="utf-8")
            shim_tokenizer = ShimTokenizer()
            host = ReplyHost(model_path)

            with patch(
                "mlx_audio.tts.models.voxtral_tts.tekken.TekkenTokenizer.from_file",
                return_value=shim_tokenizer,
            ):
                await host.handle_command(
                    {
                        "type": "send_text",
                        "turnID": "turn-speak-reply",
                        "text": "Say hello",
                        "speakReply": True,
                        "arguments": {"voice_preset": "casual_male"},
                    }
                )
                await self.wait_for_condition(
                    lambda: any(event.get("type") == "turn_finished" for event in host.events)
                )

        self.assertTrue(
            any(event.get("type") == "audio_chunk" for event in host.events)
        )
        self.assertFalse(
            any(event.get("type") == "error" for event in host.events)
        )
        self.assertIs(host.tts_model.tokenizer, shim_tokenizer)
        self.assertEqual(host.tts_model.sync_calls, 1)
        self.assertTrue(host.tts_model.used_fallback_path)

    def test_apply_model_compatibility_fixes_preserves_voice_num_audio_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            tekken_path = model_path / "tekken.json"
            tekken_path.write_text(
                json.dumps(
                    {
                        "audio": {
                            "sampling_rate": 24000,
                            "frame_rate": 12.5,
                            "chunk_length_s": 30.0,
                            "audio_encoding_config": {
                                "num_mel_bins": 128,
                                "hop_length": 160,
                                "window_size": 400,
                            },
                            "voice_num_audio_tokens": {"casual_male": 147},
                        }
                    }
                ),
                encoding="utf-8",
            )

            runtime_host.RuntimeHost.apply_model_compatibility_fixes("tts_model", model_path)
            updated = json.loads(tekken_path.read_text(encoding="utf-8"))

        self.assertEqual(
            updated["audio"]["voice_num_audio_tokens"]["casual_male"],
            147,
        )

    def test_streaming_processor_matches_demo_chunk_math(self) -> None:
        processor = runtime_host.VoxtralStreamingProcessor(
            sample_rate=16_000,
            hop_length=160,
            window_size=400,
            frame_rate=12.5,
            num_delay_tokens=6,
        )

        self.assertEqual(processor.audio_length_per_tok, 8)
        self.assertEqual(processor.num_mel_frames_first_audio_chunk, 56)
        self.assertEqual(processor.num_samples_first_audio_chunk, 9_000)
        self.assertEqual(processor.num_samples_per_audio_chunk, 1_680)
        self.assertEqual(processor.samples_per_token, 1_280)

    def test_streaming_processor_batches_follow_on_audio_like_demo(self) -> None:
        processor = runtime_host.VoxtralStreamingProcessor(
            sample_rate=16_000,
            hop_length=160,
            window_size=400,
            frame_rate=12.5,
            num_delay_tokens=6,
        )

        self.assertIsNone(
            processor.next_chunk(audio_length=8_999, finalize=False)
        )

        first_chunk = processor.next_chunk(audio_length=9_000, finalize=False)
        self.assertEqual(
            first_chunk,
            runtime_host.VoxtralStreamingChunk(0, 9_000, True),
        )

        processor.advance(mel_frames=56, is_first_audio_chunk=True)

        self.assertIsNone(
            processor.next_chunk(audio_length=10_439, finalize=False)
        )

        next_chunk = processor.next_chunk(audio_length=13_000, finalize=False)
        self.assertEqual(
            next_chunk,
            runtime_host.VoxtralStreamingChunk(8_760, 13_000, False),
        )

    def test_streaming_mel_matches_voxtral_filter_layout(self) -> None:
        class FakeAudioEncodingArgs:
            global_log_mel_max = 1.5

        class FakeConfig:
            audio_encoding_args = FakeAudioEncodingArgs()

        class FakeModel:
            config = FakeConfig()
            _mel_filters = np.ones((201, 4), dtype=np.float32)

        session = runtime_host.VoxtralRealtimeSTTSession.__new__(
            runtime_host.VoxtralRealtimeSTTSession
        )
        session._mx = np
        session._model = FakeModel()
        session._window_size = 400
        session._hop_length = 160

        mel = session._compute_streaming_mel_spectrogram(
            np.ones(1_600, dtype=np.float32),
            is_first_audio_chunk=True,
        )

        self.assertEqual(mel.shape[0], 4)
        self.assertGreater(mel.shape[1], 0)

    def test_offline_left_pad_prefix_produces_expected_frame_count(self) -> None:
        class FakeAudioEncodingArgs:
            global_log_mel_max = 1.5

        class FakeConfig:
            audio_encoding_args = FakeAudioEncodingArgs()

        class FakeModel:
            config = FakeConfig()
            _mel_filters = np.ones((201, 4), dtype=np.float32)

        session = runtime_host.VoxtralRealtimeSTTSession.__new__(
            runtime_host.VoxtralRealtimeSTTSession
        )
        session._mx = np
        session._model = FakeModel()
        session._window_size = 400
        session._hop_length = 160

        mel = session._compute_offline_mel_spectrogram(
            np.zeros(32 * 1_280, dtype=np.float32)
        )

        self.assertEqual(mel.shape, (4, 256))

    def test_short_non_first_finalize_tail_is_ignored(self) -> None:
        class FakeProcessor:
            def next_chunk(self, *, audio_length: int, finalize: bool):
                return runtime_host.VoxtralStreamingChunk(0, 360, False)

            def advance(
                self,
                *,
                mel_frames: int,
                is_first_audio_chunk: bool,
            ) -> None:
                raise AssertionError("Short finalize tail should not advance processor state.")

        session = runtime_host.VoxtralRealtimeSTTSession.__new__(
            runtime_host.VoxtralRealtimeSTTSession
        )
        session._final_audio_buffer = np.zeros(360, dtype=np.float32)
        session._audio_buffer = np.zeros(0, dtype=np.float32)
        session._audio_lock = threading.Lock()
        session._streaming_processor = FakeProcessor()
        session._compute_streaming_mel_spectrogram = (
            lambda audio_chunk, is_first_audio_chunk: np.zeros((4, 1), dtype=np.float32)
        )

        self.assertIsNone(session._next_streaming_mel_chunk(finalize=True))

    def test_reset_decoder_state_can_preserve_stream_buffers(self) -> None:
        session = runtime_host.VoxtralRealtimeSTTSession.__new__(
            runtime_host.VoxtralRealtimeSTTSession
        )
        session._decoder_cache = object()
        session._prefilled = True
        session._next_token = object()
        session._decode_position = 123
        session._prefix_len = 39
        session._generated_tokens = [1, 2, 3]
        session._audio_embed_buffer = "audio-buffer"
        session._conv1_tail = "conv1-tail"
        session._conv2_tail = "conv2-tail"
        session._encoder_cache = "encoder-cache"
        session._downsample_buffer = "downsample-buffer"
        session._encoder_position = 456

        session._reset_decoder_state(preserve_stream_state=True)

        self.assertIsNone(session._decoder_cache)
        self.assertFalse(session._prefilled)
        self.assertIsNone(session._next_token)
        self.assertEqual(session._decode_position, 39)
        self.assertEqual(session._generated_tokens, [])
        self.assertEqual(session._audio_embed_buffer, "audio-buffer")
        self.assertEqual(session._conv1_tail, "conv1-tail")
        self.assertEqual(session._conv2_tail, "conv2-tail")
        self.assertEqual(session._encoder_cache, "encoder-cache")
        self.assertEqual(session._downsample_buffer, "downsample-buffer")
        self.assertEqual(session._encoder_position, 456)

    async def test_finish_uses_final_fallback_when_live_text_is_empty(self) -> None:
        session = runtime_host.VoxtralRealtimeSTTSession.__new__(
            runtime_host.VoxtralRealtimeSTTSession
        )
        session._stopped = False
        session._stop_sentinel = object()
        session._audio_queue = runtime_host.queue.Queue()
        session._updated = asyncio.Event()
        session._worker_task = asyncio.create_task(asyncio.sleep(0))
        session._preview_task = None
        session._pcm_bytes = bytearray(b"\x00\x01" * 10)
        session._cancelled = False
        session._final_transcribe = None
        session._sample_rate = 16_000
        session._preview_text = ""
        session._full_text = ""
        session._current_segment_text = ""
        session._live_partial_seen = False
        session._transcribe_snapshot = lambda pcm_bytes, transcription_delay_ms: "rescued"

        final_text = await session.finish()

        self.assertEqual(final_text, "rescued")
        self.assertEqual(session._preview_text, "rescued")

    def test_final_text_prefers_preview_until_live_text_exists(self) -> None:
        session = runtime_host.VoxtralRealtimeSTTSession.__new__(
            runtime_host.VoxtralRealtimeSTTSession
        )
        session._preview_text = "preview text"
        session._full_text = ""
        session._current_segment_text = ""
        session._live_partial_seen = False

        self.assertEqual(session._final_text(), "preview text")

        session._full_text = "live text"
        session._live_partial_seen = True
        self.assertEqual(session._final_text(), "live text")

    async def test_stop_recording_uses_live_session_when_present(self) -> None:
        class FakeRealtimeSession(runtime_host.BaseSTTSession):
            def __init__(self) -> None:
                self.appended: list[bytes] = []
                self.finish_called = False
                self.cancel_called = False

            def append_pcm_bytes(self, pcm_bytes: bytes) -> None:
                self.appended.append(pcm_bytes)

            async def finish(self) -> str:
                self.finish_called = True
                return "live transcript"

            def cancel(self) -> None:
                self.cancel_called = True

        class FakeRealtimeConfig:
            model_type = "voxtral_realtime"

        class FakeRealtimeModel:
            config = FakeRealtimeConfig()
            encoder = object()
            decoder = object()
            _tokenizer = object()
            _mel_filters = object()

            def _ensure_ada_scales(self, transcription_delay_ms: int) -> None:
                self.transcription_delay_ms = transcription_delay_ms

        class RecordingHost(runtime_host.RuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.session = FakeRealtimeSession()
                self.events: list[dict[str, object]] = []
                self.run_turn_calls: list[tuple[str, str, list[dict[str, object]], bool, str]] = []
                self.transcribe_calls: list[tuple[int, int]] = []
                self.stt_model = FakeRealtimeModel()
                self.model_states["stt_model"] = runtime_host.ModelState(
                    installed=True,
                    warm=True,
                    path="/tmp/stt",
                    last_error=None,
                )

            def emit(self, payload: dict[str, object]) -> None:
                self.events.append(payload)

            async def warm_model(self, model_id: str, arguments: dict[str, object]) -> None:
                return None

            async def transcribe_buffer(
                self,
                pcm_bytes: bytes,
                *,
                transcription_delay_ms: int,
            ) -> str:
                self.transcribe_calls.append((len(pcm_bytes), transcription_delay_ms))
                return "batch transcript"

            async def run_turn(
                self,
                turn_id: str,
                user_text: str,
                attachments: list[dict[str, object]] | None,
                speak_reply: bool,
                voice_preset: str,
            ) -> None:
                self.run_turn_calls.append((turn_id, user_text, attachments or [], speak_reply, voice_preset))

            def create_stt_session(self, turn_id: str) -> runtime_host.BaseSTTSession:
                return self.session

        host = RecordingHost()
        chunk = b"\x01\x02\x03\x04"

        await host.start_recording(
            {
                "turnID": "turn-1",
                "speakReply": True,
                "attachments": [
                    {"type": "image", "filePath": "/tmp/image.png", "displayName": "image.png"}
                ],
                "arguments": {"voice_preset": "casual_male"},
            }
        )
        await host.append_audio_chunk(
            {
                "turnID": "turn-1",
                "chunkBase64": base64.b64encode(chunk).decode("utf-8"),
            }
        )
        await host.stop_recording("turn-1")

        self.assertEqual(host.session.appended, [chunk])
        self.assertTrue(host.session.finish_called)
        self.assertEqual(host.transcribe_calls, [])
        self.assertIn(
            {
                "type": "transcript_final",
                "turnID": "turn-1",
                "text": "live transcript",
            },
            host.events,
        )
        self.assertEqual(
            host.run_turn_calls,
            [
                (
                    "turn-1",
                    "live transcript",
                    [{"type": "image", "file_path": "/tmp/image.png", "display_name": "image.png"}],
                    True,
                    "casual_male",
                )
            ],
        )

    async def test_fallback_session_still_streams_preview_and_finalizes_batch(self) -> None:
        class FakeFallbackConfig:
            model_type = "whisper"

        class FakeFallbackModel:
            config = FakeFallbackConfig()

        class FallbackHost(runtime_host.RuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.events: list[dict[str, object]] = []
                self.run_turn_calls: list[tuple[str, str, list[dict[str, object]], bool, str]] = []
                self.transcribe_calls: list[tuple[int, int]] = []
                self.stt_model = FakeFallbackModel()
                self.model_states["stt_model"] = runtime_host.ModelState(
                    installed=True,
                    warm=True,
                    path="/tmp/stt",
                    last_error=None,
                )

            def emit(self, payload: dict[str, object]) -> None:
                self.events.append(payload)

            async def warm_model(self, model_id: str, arguments: dict[str, object]) -> None:
                return None

            async def transcribe_buffer(
                self,
                pcm_bytes: bytes,
                *,
                transcription_delay_ms: int,
            ) -> str:
                self.transcribe_calls.append((len(pcm_bytes), transcription_delay_ms))
                await asyncio.sleep(0.005)
                if transcription_delay_ms == runtime_host.PREVIEW_TRANSCRIPTION_DELAY_MS:
                    return f"preview-{len(pcm_bytes)}"
                return f"final-{len(pcm_bytes)}"

            async def run_turn(
                self,
                turn_id: str,
                user_text: str,
                attachments: list[dict[str, object]] | None,
                speak_reply: bool,
                voice_preset: str,
            ) -> None:
                self.run_turn_calls.append((turn_id, user_text, attachments or [], speak_reply, voice_preset))

        host = FallbackHost()
        chunk = b"\x01\x02" * 1600

        await host.start_recording(
            {
                "turnID": "turn-2",
                "speakReply": False,
                "arguments": {"voice_preset": "casual_male"},
            }
        )
        self.assertIsInstance(
            host.turns["turn-2"].stt_session,
            runtime_host.BufferedPreviewSTTSession,
        )

        for _ in range(4):
            await host.append_audio_chunk(
                {
                    "turnID": "turn-2",
                    "chunkBase64": base64.b64encode(chunk).decode("utf-8"),
                }
            )
            await asyncio.sleep(0.003)

        await asyncio.sleep(0.05)
        preview_events = [event for event in host.events if event.get("type") == "transcript_delta"]
        self.assertGreaterEqual(len(preview_events), 1)

        await host.stop_recording("turn-2")

        self.assertTrue(
            any(
                delay == runtime_host.FINAL_TRANSCRIPTION_DELAY_MS
                for _, delay in host.transcribe_calls
            )
        )
        self.assertIn(
            {
                "type": "transcript_final",
                "turnID": "turn-2",
                "text": "final-12800",
            },
            host.events,
        )
        self.assertEqual(
            host.run_turn_calls,
            [("turn-2", "final-12800", [], False, "casual_male")],
        )

    def test_realtime_session_detection_is_specific_to_voxtral(self) -> None:
        class FakeRealtimeConfig:
            model_type = "voxtral_realtime"

        class FakeRealtimeModel:
            config = FakeRealtimeConfig()
            encoder = object()
            decoder = object()
            _tokenizer = object()
            _mel_filters = object()

            def _ensure_ada_scales(self, transcription_delay_ms: int) -> None:
                self.transcription_delay_ms = transcription_delay_ms

        class FakeNonRealtimeConfig:
            model_type = "whisper"

        class FakeNonRealtimeModel:
            config = FakeNonRealtimeConfig()

        host = runtime_host.RuntimeHost()
        host.stt_model = FakeRealtimeModel()
        self.assertTrue(host.uses_realtime_stt_session())

        host.stt_model = FakeNonRealtimeModel()
        self.assertFalse(host.uses_realtime_stt_session())


class SamplingPresetTests(unittest.TestCase):
    def test_instruct_reasoning_preset_matches_requested_qwen_values(self) -> None:
        host = runtime_host.RuntimeHost()

        preset = host.sampling_preset(
            enable_thinking=False,
            task_profile="reasoning",
        )

        self.assertEqual(preset.temperature, 1.0)
        self.assertEqual(preset.top_p, 0.95)
        self.assertEqual(preset.top_k, 20)
        self.assertEqual(preset.min_p, 0.0)
        self.assertEqual(preset.presence_penalty, 1.5)
        self.assertEqual(preset.repetition_penalty, 1.0)

    def test_instruct_general_preset_matches_requested_qwen_values(self) -> None:
        host = runtime_host.RuntimeHost()

        preset = host.sampling_preset(
            enable_thinking=False,
            task_profile="general",
        )

        self.assertEqual(preset.temperature, 0.7)
        self.assertEqual(preset.top_p, 0.8)
        self.assertEqual(preset.top_k, 20)
        self.assertEqual(preset.min_p, 0.0)
        self.assertEqual(preset.presence_penalty, 1.5)
        self.assertEqual(preset.repetition_penalty, 1.0)

    def test_thinking_coding_preset_is_available_for_future_use(self) -> None:
        host = runtime_host.RuntimeHost()

        preset = host.sampling_preset(
            enable_thinking=True,
            task_profile="coding",
        )

        self.assertEqual(preset.temperature, 0.6)
        self.assertEqual(preset.top_p, 0.95)
        self.assertEqual(preset.top_k, 20)
        self.assertEqual(preset.min_p, 0.0)
        self.assertEqual(preset.presence_penalty, 0.0)
        self.assertEqual(preset.repetition_penalty, 1.0)


class TurnRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_only_question_runs_through_agent_loop(self) -> None:
        class RecordingHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.generated_messages: list[dict[str, object]] = []

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                self.generated_messages = json.loads(json.dumps(messages))
                return runtime_host.AssistantStepResult(text="Direct reply", tool_calls=[])

        host = RecordingHost()

        await host.run_turn(
            turn_id="turn-1",
            user_text="What is polymorphism?",
            attachments=[],
            speak_reply=False,
            voice_preset="casual_male",
        )

        self.assertEqual(len(host.generated_messages), 2)
        self.assertEqual(host.generated_messages[0]["role"], "system")
        self.assertEqual(host.generated_messages[1], {"role": "user", "content": "What is polymorphism?"})
        self.assertEqual(host.history[-1], {"role": "assistant", "content": "Direct reply"})
        self.assertEqual(host.events[-1], {"type": "turn_finished", "turnID": "turn-1", "status": "finished", "isFinal": True})

    async def test_image_turn_preserves_multimodal_user_message(self) -> None:
        class RecordingHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.generated_messages: list[dict[str, object]] = []

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                self.generated_messages = json.loads(json.dumps(messages))
                return runtime_host.AssistantStepResult(text="Planned reply", tool_calls=[])

        host = RecordingHost()

        await host.run_turn(
            turn_id="turn-2",
            user_text="Describe this image.",
            attachments=[{"type": "image", "file_path": "/tmp/test.png", "display_name": "test.png"}],
            speak_reply=False,
            voice_preset="casual_male",
        )

        self.assertEqual(len(host.generated_messages), 2)
        self.assertEqual(host.generated_messages[1]["role"], "user")
        self.assertEqual(
            host.generated_messages[1]["content"],
            [
                {"type": "input_image", "image_url": "/tmp/test.png"},
                {"type": "text", "text": "Describe this image."},
            ],
        )


class SequentialToolLoopTests(unittest.IsolatedAsyncioTestCase):
    async def wait_for_condition(
        self,
        predicate,
        *,
        timeout: float = 0.5,
        interval: float = 0.005,
    ) -> None:
        async def waiter() -> None:
            while not predicate():
                await asyncio.sleep(interval)

        await asyncio.wait_for(waiter(), timeout=timeout)

    async def test_tool_failure_is_fed_back_and_model_retries_in_same_turn(self) -> None:
        class RetryHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.step_calls = 0
                self.tool_attempts = 0
                self.message_snapshots: list[list[dict[str, object]]] = []

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                self.message_snapshots.append(json.loads(json.dumps(messages)))
                if self.step_calls == 0:
                    self.step_calls += 1
                    return runtime_host.AssistantStepResult(
                        text="Checking Finder.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="get_scripting_tips",
                                arguments={"search_term": "Finder"},
                            )
                        ],
                    )
                if self.step_calls == 1:
                    self.step_calls += 1
                    assert "Tool failed: temporary failure" in messages[-1]["content"]
                    assert "The task is still unresolved." in messages[-1]["content"]
                    assert "continue with a corrected tool call" in messages[-1]["content"]
                    return runtime_host.AssistantStepResult(
                        text="Retrying with the tool output in mind.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="get_scripting_tips",
                                arguments={"search_term": "Finder"},
                            )
                        ],
                    )
                assert "Recovered result" in messages[-1]["content"]
                return runtime_host.AssistantStepResult(text="All set.", tool_calls=[])

            async def execute_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> str:
                self.tool_attempts += 1
                if self.tool_attempts == 1:
                    raise RuntimeError("temporary failure")
                return "Recovered result"

        host = RetryHost()

        await host.run_turn(
            turn_id="retry-turn",
            user_text="Help with Finder",
            attachments=[],
            speak_reply=False,
            voice_preset="casual_male",
        )

        self.assertEqual(host.tool_attempts, 2)
        segments = assistant_segments(host.events)
        self.assertEqual(
            [segment["text"] for segment in segments],
            ["Checking Finder.", "Retrying with the tool output in mind.", "All set."],
        )
        self.assertEqual([segment["is_final"] for segment in segments], [False, False, True])
        self.assertEqual(
            host.history,
            [
                {"role": "user", "content": "Help with Finder"},
                {"role": "assistant", "content": "All set."},
            ],
        )

    async def test_script_syntax_failure_keeps_retry_guidance_in_tool_response(self) -> None:
        class SyntaxRetryHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.step_calls = 0

            def classify_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> tuple[str, str, str]:
                return ("readOnly", "notRequired", "proposed")

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                if self.step_calls == 0:
                    self.step_calls += 1
                    return runtime_host.AssistantStepResult(
                        text="Trying AppleScript first.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="execute_script",
                                arguments={"script_content": "broken applescript", "language": "applescript"},
                            )
                        ],
                    )

                assert "syntax error" in messages[-1]["content"]
                assert "The task is still unresolved." in messages[-1]["content"]
                assert "Prefer corrected local app automation" in messages[-1]["content"]
                return runtime_host.AssistantStepResult(text="Retrying with a corrected script.", tool_calls=[])

            async def execute_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> str:
                raise RuntimeError("35:36: syntax error")

        host = SyntaxRetryHost()

        await host.run_turn(
            turn_id="syntax-turn",
            user_text="Read my latest mail",
            attachments=[],
            speak_reply=False,
            voice_preset="casual_male",
        )

        self.assertEqual(host.history[-1], {"role": "assistant", "content": "Retrying with a corrected script."})

    async def test_shell_command_failure_keeps_retry_guidance_in_tool_response(self) -> None:
        class ShellRetryHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.step_calls = 0

            def classify_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> tuple[str, str, str]:
                return ("readOnly", "notRequired", "proposed")

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                if self.step_calls == 0:
                    self.step_calls += 1
                    return runtime_host.AssistantStepResult(
                        text="Trying the browser automation.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="execute_script",
                                arguments={"script_content": "bad shell usage", "language": "applescript"},
                            )
                        ],
                    )

                assert "command not found" in messages[-1]["content"]
                assert "continue with a corrected tool call" in messages[-1]["content"]
                assert "Only stop without another tool call" in messages[-1]["content"]
                return runtime_host.AssistantStepResult(text="Retrying with corrected browser automation.", tool_calls=[])

            async def execute_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> str:
                raise RuntimeError("sh: google: command not found (127)")

        host = ShellRetryHost()

        await host.run_turn(
            turn_id="shell-turn",
            user_text="Search in Safari",
            attachments=[],
            speak_reply=False,
            voice_preset="casual_male",
        )

        self.assertEqual(host.history[-1], {"role": "assistant", "content": "Retrying with corrected browser automation."})

    async def test_multi_step_loop_runs_until_no_tool_call_remains(self) -> None:
        class MultiStepHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.step_calls = 0
                self.tool_runs: list[str] = []

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                if self.step_calls == 0:
                    self.step_calls += 1
                    return runtime_host.AssistantStepResult(
                        text="Looking up the first step.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="get_scripting_tips",
                                arguments={"search_term": "Finder"},
                            )
                        ],
                    )
                if self.step_calls == 1:
                    self.step_calls += 1
                    assert "tool_response" in messages[-1]["content"]
                    return runtime_host.AssistantStepResult(
                        text="That worked. Checking one more thing.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="get_scripting_tips",
                                arguments={"search_term": "Mail"},
                            )
                        ],
                    )
                self.step_calls += 1
                assert "Mail" in messages[-1]["content"]
                return runtime_host.AssistantStepResult(text="Finished after both checks.", tool_calls=[])

            async def execute_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> str:
                search_term = str(tool_arguments["search_term"])
                self.tool_runs.append(search_term)
                return f"result:{search_term}"

        host = MultiStepHost()

        await host.run_turn(
            turn_id="multi-turn",
            user_text="Check Finder and Mail",
            attachments=[],
            speak_reply=False,
            voice_preset="casual_male",
        )

        self.assertEqual(host.step_calls, 3)
        self.assertEqual(host.tool_runs, ["Finder", "Mail"])
        self.assertEqual(
            host.history,
            [
                {"role": "user", "content": "Check Finder and Mail"},
                {"role": "assistant", "content": "Finished after both checks."},
            ],
        )

    async def test_multiple_tool_calls_in_one_step_run_serially_before_next_generation(self) -> None:
        class SerialHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.step_calls = 0
                self.tool_runs: list[str] = []
                self.followup_messages: list[dict[str, object]] = []

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                if self.step_calls == 0:
                    self.step_calls += 1
                    return runtime_host.AssistantStepResult(
                        text="Checking both apps.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="get_scripting_tips",
                                arguments={"search_term": "Finder"},
                            ),
                            runtime_host.ToolCallRequest(
                                name="get_scripting_tips",
                                arguments={"search_term": "Mail"},
                            ),
                        ],
                    )
                self.followup_messages = json.loads(json.dumps(messages))
                return runtime_host.AssistantStepResult(text="Done with both.", tool_calls=[])

            async def execute_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> str:
                search_term = str(tool_arguments["search_term"])
                self.tool_runs.append(search_term)
                return f"result:{search_term}"

        host = SerialHost()

        await host.run_turn(
            turn_id="serial-turn",
            user_text="Check both tools",
            attachments=[],
            speak_reply=False,
            voice_preset="casual_male",
        )

        self.assertEqual(host.tool_runs, ["Finder", "Mail"])
        self.assertEqual(host.followup_messages[2]["role"], "assistant")
        self.assertEqual(len(host.followup_messages[2]["tool_calls"]), 2)
        self.assertIn("result:Finder", host.followup_messages[3]["content"])
        self.assertIn("result:Mail", host.followup_messages[4]["content"])

    async def test_intermediate_assistant_segments_are_emitted_before_tool_steps(self) -> None:
        class SegmentHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.step_calls = 0

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                if self.step_calls == 0:
                    self.step_calls += 1
                    return runtime_host.AssistantStepResult(
                        text="First update.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="get_scripting_tips",
                                arguments={"search_term": "Finder"},
                            )
                        ],
                    )
                if self.step_calls == 1:
                    self.step_calls += 1
                    return runtime_host.AssistantStepResult(
                        text="Second update.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="get_scripting_tips",
                                arguments={"search_term": "Mail"},
                            )
                        ],
                    )
                return runtime_host.AssistantStepResult(text="Done.", tool_calls=[])

            async def execute_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> str:
                return "ok"

        host = SegmentHost()

        await host.run_turn(
            turn_id="segment-turn",
            user_text="Check updates",
            attachments=[],
            speak_reply=False,
            voice_preset="casual_male",
        )

        segments = assistant_segments(host.events)
        self.assertEqual(
            [segment["text"] for segment in segments],
            ["First update.", "Second update.", "Done."],
        )
        self.assertEqual([segment["is_final"] for segment in segments], [False, False, True])
        self.assertEqual(len({segment["segment_id"] for segment in segments}), 3)

    async def test_reply_speech_starts_only_after_terminal_assistant_step(self) -> None:
        class SpeakingLoopHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.tts_model = object()
                self.step_calls = 0

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                if self.step_calls == 0:
                    self.step_calls += 1
                    return runtime_host.AssistantStepResult(
                        text="Working on it.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="get_scripting_tips",
                                arguments={"search_term": "Finder"},
                            )
                        ],
                    )
                return runtime_host.AssistantStepResult(text="Finished.", tool_calls=[])

            async def execute_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> str:
                return "ok"

            def iter_tts_pcm_chunks(
                self,
                *,
                text: str,
                voice_preset: str,
                cancel_event: threading.Event | None = None,
            ):
                yield b"audio"

        host = SpeakingLoopHost()

        await host.run_turn(
            turn_id="speech-turn",
            user_text="Check and speak",
            attachments=[],
            speak_reply=True,
            voice_preset="casual_male",
        )

        events = host.events
        audio_index = next(index for index, event in enumerate(events) if event.get("type") == "audio_chunk")
        final_segment_index = max(
            index
            for index, event in enumerate(events)
            if event.get("type") == "assistant_delta" and event.get("isFinalSegment") is True
        )
        intermediate_segment_index = max(
            index
            for index, event in enumerate(events)
            if event.get("type") == "assistant_delta" and event.get("isFinalSegment") is False
        )
        self.assertGreater(audio_index, final_segment_index)
        self.assertGreater(final_segment_index, intermediate_segment_index)

    async def test_approval_required_tools_resume_the_same_loop_after_approval(self) -> None:
        class ApprovalHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.step_calls = 0

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                if self.step_calls == 0:
                    self.step_calls += 1
                    return runtime_host.AssistantStepResult(
                        text="Need approval before I can continue.",
                        tool_calls=[
                            runtime_host.ToolCallRequest(
                                name="execute_script",
                                arguments={"script_content": "return 1", "language": "applescript"},
                            )
                        ],
                    )
                self.step_calls += 1
                assert "Approved result" in messages[-1]["content"]
                return runtime_host.AssistantStepResult(text="Done after approval.", tool_calls=[])

            async def execute_tool(
                self,
                tool_name: str,
                tool_arguments: dict[str, object],
            ) -> str:
                return "Approved result"

        host = ApprovalHost()

        await host.handle_command(
            {
                "type": "send_text",
                "turnID": "approval-turn",
                "text": "Run the script",
                "speakReply": False,
            }
        )
        await self.wait_for_condition(
            lambda: any(event.get("type") == "tool_proposed" for event in host.events)
        )
        proposed_event = next(event for event in host.events if event.get("type") == "tool_proposed")

        await host.handle_command(
            {
                "type": "approve_tool",
                "turnID": "approval-turn",
                "toolCallID": proposed_event["toolCallID"],
            }
        )
        await self.wait_for_condition(
            lambda: any(
                event.get("type") == "turn_finished"
                and event.get("turnID") == "approval-turn"
                for event in host.events
            )
        )

        self.assertEqual(host.step_calls, 2)
        self.assertEqual(
            host.history,
            [
                {"role": "user", "content": "Run the script"},
                {"role": "assistant", "content": "Done after approval."},
            ],
        )
        self.assertEqual(host.pending_tools, {})


class ResetConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_reset_conversation_clears_history_and_keeps_warm_models_loaded(self) -> None:
        class RecordingHost(runtime_host.RuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.events: list[dict[str, object]] = []

            def emit(self, payload: dict[str, object]) -> None:
                self.events.append(payload)

        host = RecordingHost()
        model_object = object()
        processor_object = object()
        host.agent_model = model_object
        host.agent_processor = processor_object
        host.model_states["agent_model"] = runtime_host.ModelState(
            installed=True,
            warm=True,
            path="/tmp/agent-model",
            last_error=None,
        )
        host.history = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "/tmp/old-image.png"},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {"role": "assistant", "content": "Old reply"},
        ]
        host.pending_tools["tool-1"] = {
            "turn_id": "turn-1",
            "tool_name": "execute_script",
            "tool_arguments": {},
        }
        host.turns["turn-1"] = runtime_host.TurnBuffer(
            turn_id="turn-1",
            speak_reply=False,
            voice_preset="casual_male",
        )
        host.cancelled_turns.add("stale-turn")

        await host.handle_command({"type": "reset_conversation"})

        self.assertEqual(host.history, [])
        self.assertEqual(host.pending_tools, {})
        self.assertEqual(host.turns, {})
        self.assertEqual(host.cancelled_turns, set())
        self.assertIs(host.agent_model, model_object)
        self.assertIs(host.agent_processor, processor_object)
        self.assertTrue(host.model_states["agent_model"].installed)
        self.assertTrue(host.model_states["agent_model"].warm)

    async def test_reset_conversation_removes_prior_image_context_from_next_text_turn(self) -> None:
        class RecordingHost(TestRuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.generated_messages: list[dict[str, object]] = []

            async def generate_assistant_step(
                self,
                messages: list[dict[str, object]],
                *,
                tool_schemas: list[dict[str, object]],
                turn_id: str,
            ) -> runtime_host.AssistantStepResult:
                self.generated_messages = json.loads(json.dumps(messages))
                return runtime_host.AssistantStepResult(text="Direct reply", tool_calls=[])

        host = RecordingHost()
        host.history = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "/tmp/old-image.png"},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {"role": "assistant", "content": "Old reply"},
        ]

        await host.handle_command({"type": "reset_conversation"})
        await host.run_turn(
            turn_id="turn-3",
            user_text="What is polymorphism?",
            attachments=[],
            speak_reply=False,
            voice_preset="casual_male",
        )

        self.assertEqual(len(host.generated_messages), 2)
        self.assertEqual(host.generated_messages[0]["role"], "system")
        self.assertEqual(host.generated_messages[1], {"role": "user", "content": "What is polymorphism?"})


class AgentRuntimeDependencyTests(unittest.IsolatedAsyncioTestCase):
    def test_missing_agent_runtime_dependencies_reports_qwen_requirements(self) -> None:
        host = runtime_host.RuntimeHost()
        real_find_spec = runtime_host.importlib.util.find_spec

        def fake_find_spec(module_name: str):
            if module_name in {"torch", "torchvision"}:
                return None
            return real_find_spec(module_name)

        with mock.patch.object(runtime_host.importlib.util, "find_spec", side_effect=fake_find_spec):
            with self.assertRaises(RuntimeError) as context:
                host.ensure_agent_runtime_dependencies()

        message = str(context.exception)
        self.assertIn("Qwen image processor", message)
        self.assertIn("PyTorch", message)
        self.assertIn("Torchvision", message)
        self.assertIn("refresh the runtime packages", message)

    def test_incompatible_mistral_common_reports_qwen_requirements(self) -> None:
        host = runtime_host.RuntimeHost()
        real_find_spec = runtime_host.importlib.util.find_spec
        real_import_module = runtime_host.importlib.import_module

        class IncompatibleMistralRequestModule:
            ChatCompletionRequest = object

        def fake_find_spec(module_name: str):
            if module_name == "mistral_common":
                return object()
            return real_find_spec(module_name)

        def fake_import_module(module_name: str, package: str | None = None):
            if module_name == "mistral_common.protocol.instruct.request":
                return IncompatibleMistralRequestModule()
            return real_import_module(module_name, package)

        with mock.patch.object(runtime_host.importlib.util, "find_spec", side_effect=fake_find_spec):
            with mock.patch.object(runtime_host.importlib, "import_module", side_effect=fake_import_module):
                with self.assertRaises(RuntimeError) as context:
                    host.ensure_agent_runtime_dependencies()

        message = str(context.exception)
        self.assertIn("mistral-common", message)
        self.assertIn("ReasoningEffort", message)
        self.assertIn("refresh the runtime packages", message)

    async def test_warm_model_reports_missing_agent_runtime_dependencies_before_load(self) -> None:
        class RecordingHost(runtime_host.RuntimeHost):
            def __init__(self) -> None:
                super().__init__()
                self.events: list[dict[str, object]] = []

            def emit(self, payload: dict[str, object]) -> None:
                self.events.append(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            original_manifest = runtime_host.MANIFEST_PATH
            try:
                runtime_host.MANIFEST_PATH = Path(temp_dir) / "manifest.json"
                host = RecordingHost()
                host.model_states["agent_model"] = runtime_host.ModelState(
                    installed=True,
                    warm=False,
                    path="/tmp/agent-model",
                    last_error=None,
                )
                real_find_spec = runtime_host.importlib.util.find_spec

                def fake_find_spec(module_name: str):
                    if module_name in {"torch", "torchvision"}:
                        return None
                    return real_find_spec(module_name)

                with mock.patch.object(runtime_host.importlib.util, "find_spec", side_effect=fake_find_spec):
                    with mock.patch.object(runtime_host.asyncio, "to_thread", new=mock.AsyncMock()) as to_thread:
                        await host.warm_model("agent_model", {})

                self.assertFalse(host.model_states["agent_model"].warm)
                self.assertIn("Qwen image processor", host.model_states["agent_model"].last_error)
                self.assertEqual(host.events[-1]["type"], "model_state")
                self.assertEqual(host.events[-1]["warmState"], "error")
                to_thread.assert_not_awaited()
            finally:
                runtime_host.MANIFEST_PATH = original_manifest

    def test_missing_tool_capable_chat_template_reports_reinstall_error(self) -> None:
        host = runtime_host.RuntimeHost()
        host.agent_processor = type("Processor", (), {"chat_template": ""})()
        host.agent_tokenizer = type(
            "Tokenizer",
            (),
            {
                "has_chat_template": True,
                "has_tool_calling": True,
                "tool_parser": object(),
            },
        )()

        with self.assertRaises(RuntimeError) as context:
            host.assert_agent_tool_template(Path("/tmp/qwen-model"))

        message = str(context.exception)
        self.assertIn("missing its bundled Qwen chat template", message)
        self.assertIn("Reinstall or refresh the local agent model/runtime", message)


class MultimodalHistoryTests(unittest.TestCase):
    def test_normalize_multimodal_messages_preserves_image_turn_order(self) -> None:
        host = runtime_host.RuntimeHost()
        messages = [
            {"role": "system", "content": "Be helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "/tmp/one.png"},
                    {"type": "text", "text": "Describe the first image."},
                ],
            },
            {"role": "assistant", "content": "It is a chart."},
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "/tmp/two.png"},
                    {"type": "text", "text": "Now compare it to this one."},
                ],
            },
        ]

        with mock.patch.object(runtime_host.Path, "exists", return_value=True):
            normalized, images = host.normalize_multimodal_messages(messages)

        self.assertEqual(images, ["/tmp/one.png", "/tmp/two.png"])
        self.assertEqual(normalized[1]["content"][0]["type"], "input_image")
        self.assertEqual(normalized[3]["content"][0]["image_url"], "/tmp/two.png")

    def test_build_user_message_supports_attachment_only_turns(self) -> None:
        host = runtime_host.RuntimeHost()

        message = host.build_user_message(
            "",
            [{"type": "image", "file_path": "/tmp/only-image.png", "display_name": "only-image.png"}],
        )

        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"], [{"type": "input_image", "image_url": "/tmp/only-image.png"}])


class ManifestCompatibilityTests(unittest.TestCase):
    def test_load_manifest_invalidates_outdated_agent_model_repo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            model_dir = Path(temp_dir) / "agent_model"
            model_dir.mkdir()

            manifest_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "agent_model": {
                                "installed": True,
                                "warm": False,
                                "path": str(model_dir),
                                "repo": "Qwen/Qwen3-4B-MLX-4bit",
                                "last_error": None,
                            }
                        }
                    }
                )
            )

            original_manifest = runtime_host.MANIFEST_PATH
            try:
                runtime_host.MANIFEST_PATH = manifest_path
                host = runtime_host.RuntimeHost()
                manifest = host._load_manifest()
            finally:
                runtime_host.MANIFEST_PATH = original_manifest

            self.assertFalse(manifest["agent_model"].installed)
            self.assertIn("Reinstall required", manifest["agent_model"].last_error)
            self.assertFalse(model_dir.exists())


if __name__ == "__main__":
    unittest.main()
