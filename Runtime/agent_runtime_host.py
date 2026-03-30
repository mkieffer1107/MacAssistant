#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import importlib
import importlib.util
import io
import json
import math
import os
import queue
import re
import signal
import shutil
import struct
import sys
import tempfile
import threading
import time
import wave
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
from tqdm.auto import tqdm as base_tqdm


APP_SUPPORT = Path(os.environ.get("MAC_ASSISTANT_APP_SUPPORT", Path.home() / "Library/Application Support/MacAssistant"))
MODELS_ROOT = APP_SUPPORT / "models"
MANIFEST_PATH = MODELS_ROOT / "manifest.json"
LOGS_ROOT = APP_SUPPORT / "logs"
CACHE_ROOT = APP_SUPPORT / "cache"
RUNTIME_ROOT = APP_SUPPORT / "runtime"
MCP_ROOT = RUNTIME_ROOT / "mcp"
BUN_CACHE_ROOT = CACHE_ROOT / "bun"
MACOS_AUTOMATOR_PACKAGE = os.environ.get(
    "MAC_ASSISTANT_MACOS_AUTOMATOR_PACKAGE",
    "@steipete/macos-automator-mcp@0.4.1",
)
PREVIEW_TRANSCRIPTION_DELAY_MS = 160
FINAL_TRANSCRIPTION_DELAY_MS = 480
PREVIEW_POLL_INTERVAL_S = 0.1
MIN_PREVIEW_PCM_BYTES = 6_400
REALTIME_TRANSCRIPTION_DELAY_MS = FINAL_TRANSCRIPTION_DELAY_MS
TTS_PCM_SAMPLE_RATE = 24_000
TTS_PCM_CHANNELS = 1
TTS_PCM_ENCODING = "pcm_s16le"
TTS_STREAM_FRAME_BATCH = 8

MODEL_SPECS = {
    "agent_model": {
        "repo": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "family": "llm",
    },
    "tts_model": {
        "repo": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
        "family": "tts",
    },
    "stt_model": {
        "repo": "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
        "family": "stt",
    },
}

INSTALLABLES = {
    "voice_pack": ["stt_model", "tts_model"],
    "agent_model": ["agent_model"],
}

LEGACY_MODEL_IDS = ("qwen_agent", "voxtral_tts", "voxtral_realtime")
AGENT_RUNTIME_DEPENDENCIES = {
    "mistral_common": "mistral-common",
    "torch": "PyTorch",
    "torchvision": "Torchvision",
}

READ_ONLY_TOOL_NAMES = {"get_scripting_tips"}
SYSTEM_INFO_SCRIPT = "return system info"
NULL_TQDM_STREAM = open(os.devnull, "w", encoding="utf-8")


def runtime_log(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


class SilentTqdm(base_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def make_download_progress_tqdm(model_id: str) -> type[base_tqdm]:
    class DownloadProgressTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            progress_name = kwargs.pop("name", None)
            self._is_download_bar = progress_name == "huggingface_hub.snapshot_download"
            self._emit_lock = threading.Lock()
            self._last_emit_at = 0.0
            self._last_emitted_bytes = -1
            self._last_emitted_total = -1
            self._started_at = time.monotonic()
            self._closed = False
            kwargs["disable"] = False
            kwargs.setdefault("file", NULL_TQDM_STREAM)
            kwargs.setdefault("leave", False)
            super().__init__(*args, **kwargs)
            if self._is_download_bar:
                self._emit_progress(force=True)

        def update(self, n: int | float = 1) -> None:
            super().update(n)
            if self._is_download_bar:
                self._emit_progress(force=False)

        def refresh(self, *args, **kwargs) -> None:
            super().refresh(*args, **kwargs)
            if self._is_download_bar:
                self._emit_progress(force=False)

        def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
            super().set_description(desc, refresh=refresh)
            if self._is_download_bar:
                self._emit_progress(force=True)

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            if self._is_download_bar:
                self._emit_progress(force=True)
            super().close()

        def _emit_progress(self, force: bool) -> None:
            with self._emit_lock:
                completed_bytes = int(self.n or 0)
                total_bytes = int(self.total or 0)
                now = time.monotonic()
                is_final = total_bytes > 0 and completed_bytes >= total_bytes
                total_changed = total_bytes != self._last_emitted_total
                completed_changed = completed_bytes != self._last_emitted_bytes
                if not force:
                    if not total_changed and not completed_changed:
                        return
                    if not is_final and now - self._last_emit_at < 0.25:
                        return

                elapsed = max(now - self._started_at, 0.001)
                speed_bytes_per_second = completed_bytes / elapsed if completed_bytes > 0 else None
                eta_seconds = None
                if total_bytes > 0 and speed_bytes_per_second and speed_bytes_per_second > 0:
                    eta_seconds = max((total_bytes - completed_bytes) / speed_bytes_per_second, 0.0)

                payload = {
                    "type": "download_progress",
                    "modelID": model_id,
                    "bytesDownloaded": completed_bytes,
                    "bytesTotal": total_bytes,
                    "speedBytesPerSecond": speed_bytes_per_second,
                    "etaSeconds": eta_seconds,
                }
                print(json.dumps(payload), flush=True)
                self._last_emit_at = now
                self._last_emitted_bytes = completed_bytes
                self._last_emitted_total = total_bytes

    return DownloadProgressTqdm


@dataclass
class ModelState:
    installed: bool = False
    warm: bool = False
    path: str | None = None
    last_error: str | None = None


@dataclass
class TurnBuffer:
    turn_id: str
    speak_reply: bool
    voice_preset: str
    attachments: list[dict[str, Any]] = field(default_factory=list)
    pcm_bytes: bytearray = field(default_factory=bytearray)
    stopped: bool = False
    stt_session: "BaseSTTSession | None" = None


@dataclass
class TurnExecutionState:
    turn_id: str
    task: asyncio.Task[None] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    finished_emitted: bool = False


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class SamplingPreset:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float


QWEN_TEXT_ONLY_SAMPLING_PRESETS = {
    "thinking_general": SamplingPreset(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
    ),
    "thinking_coding": SamplingPreset(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    ),
    "instruct_general": SamplingPreset(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
    ),
    "instruct_reasoning": SamplingPreset(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
    ),
}


class BaseSTTSession:
    def append_pcm_bytes(self, pcm_bytes: bytes) -> None:
        raise NotImplementedError

    async def finish(self) -> str:
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError


class BufferedPreviewSTTSession(BaseSTTSession):
    def __init__(
        self,
        *,
        turn_id: str,
        preview_transcribe: Callable[[bytes], Awaitable[str]],
        final_transcribe: Callable[[bytes], Awaitable[str]],
        emit_partial: Callable[[str], None],
        emit_error: Callable[[str], None],
        min_preview_pcm_bytes: int = MIN_PREVIEW_PCM_BYTES,
        preview_poll_interval_s: float = PREVIEW_POLL_INTERVAL_S,
    ) -> None:
        self.turn_id = turn_id
        self._preview_transcribe = preview_transcribe
        self._final_transcribe = final_transcribe
        self._emit_partial = emit_partial
        self._emit_error = emit_error
        self._min_preview_pcm_bytes = min_preview_pcm_bytes
        self._preview_poll_interval_s = preview_poll_interval_s
        self._pcm_bytes = bytearray()
        self._last_text = ""
        self._dirty = False
        self._stopped = False
        self._updated = asyncio.Event()
        self._preview_task = asyncio.create_task(self._preview_loop())

    def append_pcm_bytes(self, pcm_bytes: bytes) -> None:
        if self._stopped or not pcm_bytes:
            return
        self._pcm_bytes.extend(pcm_bytes)
        self._dirty = True
        self._updated.set()

    async def finish(self) -> str:
        if self._stopped:
            return await self._final_transcribe(bytes(self._pcm_bytes))

        self._stopped = True
        self._updated.set()
        if self._preview_task:
            await self._preview_task
        return await self._final_transcribe(bytes(self._pcm_bytes))

    def cancel(self) -> None:
        self._stopped = True
        self._updated.set()
        if self._preview_task:
            self._preview_task.cancel()

    async def _preview_loop(self) -> None:
        while not self._stopped:
            try:
                await asyncio.wait_for(
                    self._updated.wait(),
                    timeout=self._preview_poll_interval_s,
                )
            except asyncio.TimeoutError:
                pass
            self._updated.clear()
            if not self._dirty or len(self._pcm_bytes) < self._min_preview_pcm_bytes:
                continue

            snapshot = bytes(self._pcm_bytes)
            self._dirty = False
            try:
                text = (await self._preview_transcribe(snapshot)).strip()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._emit_error(f"Preview transcription failed: {exc}")
                continue

            if text and text != self._last_text:
                self._last_text = text
                self._emit_partial(text)


@dataclass
class VoxtralStreamingChunk:
    start_sample: int
    end_sample: int
    is_first_audio_chunk: bool


class VoxtralStreamingProcessor:
    def __init__(
        self,
        *,
        sample_rate: int,
        hop_length: int,
        window_size: int,
        frame_rate: float,
        num_delay_tokens: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.window_size = window_size
        self.win_half = window_size // 2
        self.raw_audio_length_per_tok = int(sample_rate / frame_rate)
        self.audio_length_per_tok = self.raw_audio_length_per_tok // hop_length
        self.num_delay_tokens = num_delay_tokens
        self.num_mel_frames_first_audio_chunk = (
            (num_delay_tokens + 1) * self.audio_length_per_tok
        )
        self.num_samples_first_audio_chunk = (
            (self.num_mel_frames_first_audio_chunk - 1) * hop_length
            + self.win_half
        )
        self.num_samples_per_audio_chunk = (
            self.audio_length_per_tok * hop_length + window_size
        )
        self._dispatched_first_chunk = False
        self._mel_frame_idx = 0
        self._next_start_sample = 0

    @property
    def samples_per_token(self) -> int:
        return self.audio_length_per_tok * self.hop_length

    def next_chunk(
        self,
        *,
        audio_length: int,
        finalize: bool,
    ) -> VoxtralStreamingChunk | None:
        if not self._dispatched_first_chunk:
            if audio_length < self.num_samples_first_audio_chunk and not finalize:
                return None
            chunk_end = min(audio_length, self.num_samples_first_audio_chunk)
            if chunk_end <= 0:
                return None
            return VoxtralStreamingChunk(
                start_sample=0,
                end_sample=chunk_end,
                is_first_audio_chunk=True,
            )

        if self._next_start_sample >= audio_length:
            return None

        end_needed = self._next_start_sample + self.num_samples_per_audio_chunk
        if audio_length < end_needed and not finalize:
            return None

        chunk_end = min(audio_length, end_needed)
        while chunk_end + self.samples_per_token <= audio_length:
            chunk_end += self.samples_per_token

        if chunk_end <= self._next_start_sample:
            return None

        return VoxtralStreamingChunk(
            start_sample=self._next_start_sample,
            end_sample=chunk_end,
            is_first_audio_chunk=False,
        )

    def advance(
        self,
        *,
        mel_frames: int,
        is_first_audio_chunk: bool,
    ) -> None:
        if mel_frames <= 0:
            return
        if is_first_audio_chunk:
            self._dispatched_first_chunk = True
            self._mel_frame_idx = mel_frames
        else:
            self._mel_frame_idx += mel_frames
        self._next_start_sample = self._mel_frame_idx * self.hop_length - self.win_half


class VoxtralRealtimeSTTSession(BaseSTTSession):
    def __init__(
        self,
        *,
        turn_id: str,
        model: Any,
        emit_partial: Callable[[str], None],
        emit_error: Callable[[str], None],
        transcription_delay_ms: int = REALTIME_TRANSCRIPTION_DELAY_MS,
        min_preview_pcm_bytes: int = MIN_PREVIEW_PCM_BYTES,
        preview_poll_interval_s: float = PREVIEW_POLL_INTERVAL_S,
    ) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.cache import RotatingKVCache

        self.turn_id = turn_id
        self._mx = mx
        self._nn = nn
        self._RotatingKVCache = RotatingKVCache
        self._model = model
        self._emit_partial = emit_partial
        self._emit_error = emit_error
        self._transcription_delay_ms = transcription_delay_ms
        self._min_preview_pcm_bytes = min_preview_pcm_bytes
        self._preview_poll_interval_s = preview_poll_interval_s
        self._loop = asyncio.get_running_loop()
        self._audio_queue: queue.Queue[object] = queue.Queue()
        self._stop_sentinel = object()
        self._model_lock = threading.Lock()
        self._audio_lock = threading.Lock()
        self._stopped = False
        self._cancelled = False
        self._full_text = ""
        self._current_segment_text = ""
        self._preview_text = ""
        self._last_emitted_text = ""
        self._pcm_bytes = bytearray()
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._final_audio_buffer: np.ndarray | None = None
        self._session_started_at = time.monotonic()
        self._first_partial_at: float | None = None
        self._eos_reset_count = 0
        self._audio_seconds_ingested = 0.0
        self._dirty = False
        self._updated = asyncio.Event()
        self._live_partial_seen = False

        config = model.config
        audio_config = config.audio_encoding_args
        self._sample_rate = int(audio_config.sampling_rate)
        self._hop_length = int(audio_config.hop_length)
        self._window_size = int(audio_config.window_size)
        self._frame_rate = float(audio_config.frame_rate)
        self._raw_audio_length_per_tok = int(self._sample_rate / self._frame_rate)
        self._samples_per_token = self._raw_audio_length_per_tok
        self._n_left_pad_tokens = int(config.n_left_pad_tokens)
        self._n_delay_tokens = self._num_delay_tokens(
            delay_ms=transcription_delay_ms,
            sample_rate=self._sample_rate,
            hop_length=self._hop_length,
            frame_rate=self._frame_rate,
        )
        self._n_right_pad_tokens = self._n_delay_tokens + 11
        self._streaming_processor = VoxtralStreamingProcessor(
            sample_rate=self._sample_rate,
            hop_length=self._hop_length,
            window_size=self._window_size,
            frame_rate=self._frame_rate,
            num_delay_tokens=self._n_delay_tokens,
        )
        self._prefix_token_ids = [config.bos_token_id] + [
            config.streaming_pad_token_id
        ] * (self._n_left_pad_tokens + self._n_delay_tokens)
        self._prefix_len = len(self._prefix_token_ids)
        self._eos_token_id = int(config.eos_token_id)
        self._temperature = 0.0

        prompt_ids = self._mx.array(self._prefix_token_ids)
        self._prompt_text_embeds = model.decoder.embed_tokens(prompt_ids)
        self._mx.eval(self._prompt_text_embeds)
        model._ensure_ada_scales(transcription_delay_ms)

        self._conv1_tail = None
        self._conv2_tail = None
        self._encoder_cache: list[Any] | None = None
        self._decoder_cache: list[Any] | None = None
        self._downsample_buffer = None
        self._audio_embed_buffer = None
        self._encoder_position = 0
        self._prefilled = False
        self._next_token = None
        self._decode_position = self._prefix_len
        self._generated_tokens: list[int] = []
        self._preview_fallback_after_s = 1.5

        self._log_debug(
            "selected stt path=realtime "
            f"delay_ms={transcription_delay_ms} prefix_len={self._prefix_len} "
            f"first_chunk_samples={self._streaming_processor.num_samples_first_audio_chunk} "
            f"chunk_samples={self._streaming_processor.num_samples_per_audio_chunk}"
        )
        with self._model_lock:
            self._prime_left_pad_prefix()
        self._worker_task = asyncio.create_task(asyncio.to_thread(self._run_worker))
        self._preview_task = asyncio.create_task(self._preview_loop())

    def append_pcm_bytes(self, pcm_bytes: bytes) -> None:
        if self._stopped or self._cancelled or not pcm_bytes:
            return
        self._pcm_bytes.extend(pcm_bytes)
        self._audio_seconds_ingested = len(self._pcm_bytes) / 2.0 / self._sample_rate
        self._dirty = True
        self._updated.set()
        audio_chunk = self._pcm_bytes_to_audio_array(pcm_bytes)
        with self._audio_lock:
            if self._audio_buffer.size == 0:
                self._audio_buffer = audio_chunk
            else:
                self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])
        self._audio_queue.put(object())

    async def finish(self) -> str:
        if self._stopped:
            return self._final_text()

        self._stopped = True
        self._audio_queue.put(self._stop_sentinel)
        self._updated.set()
        try:
            await self._worker_task
            if self._preview_task is not None:
                await self._preview_task
        except asyncio.CancelledError:
            raise
        final_text = self._final_text()
        if not final_text and self._pcm_bytes:
            try:
                fallback_text = (
                    await asyncio.to_thread(
                        self._transcribe_snapshot,
                        bytes(self._pcm_bytes),
                        FINAL_TRANSCRIPTION_DELAY_MS,
                    )
                ).strip()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._emit_error(f"Realtime final transcription fallback failed: {exc}")
            else:
                if fallback_text:
                    self._preview_text = fallback_text
                    final_text = fallback_text
        return final_text

    def cancel(self) -> None:
        self._cancelled = True
        self._stopped = True
        self._audio_queue.put(self._stop_sentinel)
        self._updated.set()
        if self._worker_task:
            self._worker_task.cancel()
        if self._preview_task:
            self._preview_task.cancel()

    def _run_worker(self) -> None:
        try:
            while True:
                try:
                    item = self._audio_queue.get(timeout=0.02)
                except queue.Empty:
                    item = None

                if item is self._stop_sentinel:
                    break

                with self._model_lock:
                    self._advance_stream(finalize=False)
                if self._cancelled:
                    return

            with self._model_lock:
                self._flush_stream()
        except Exception as exc:  # noqa: BLE001
            self._schedule_error(f"Realtime transcription failed: {exc}")

    async def _preview_loop(self) -> None:
        while not self._stopped:
            try:
                await asyncio.wait_for(
                    self._updated.wait(),
                    timeout=self._preview_poll_interval_s,
                )
            except asyncio.TimeoutError:
                pass
            self._updated.clear()
            if self._live_partial_seen:
                return
            if time.monotonic() - self._session_started_at < self._preview_fallback_after_s:
                continue
            if not self._dirty or len(self._pcm_bytes) < self._min_preview_pcm_bytes:
                continue

            snapshot = bytes(self._pcm_bytes)
            self._dirty = False
            try:
                preview_text = (
                    await asyncio.to_thread(
                        self._transcribe_snapshot,
                        snapshot,
                        PREVIEW_TRANSCRIPTION_DELAY_MS,
                    )
                ).strip()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._emit_error(f"Realtime preview fallback failed: {exc}")
                continue

            if preview_text:
                self._update_preview_text(preview_text)

    def _flush_stream(self) -> None:
        if self._cancelled:
            return

        if self._final_audio_buffer is None:
            with self._audio_lock:
                final_audio = self._audio_buffer.copy()
            right_pad = np.zeros(
                self._n_right_pad_tokens * self._samples_per_token,
                dtype=np.float32,
            )
            self._final_audio_buffer = np.concatenate([final_audio, right_pad])
        self._advance_stream(finalize=True)
        self._emit_cumulative_text(force=True)
        self._log_debug(
            "finish "
            f"audio_s={self._audio_seconds_ingested:.2f} "
            f"first_partial_s={self._first_partial_latency():.2f} "
            f"eos_resets={self._eos_reset_count} "
            f"final_chars={len(self._final_text())}"
        )

    def _advance_stream(self, *, finalize: bool) -> None:
        while True:
            mel = self._next_streaming_mel_chunk(finalize=finalize)
            if mel is None:
                break
            new_embeds = self._encode_mel_chunk(mel)
            if new_embeds is not None:
                self._append_audio_embeds(new_embeds)
            self._decode_available()

    def _prime_left_pad_prefix(self) -> None:
        silence = np.zeros(
            self._n_left_pad_tokens * self._raw_audio_length_per_tok,
            dtype=np.float32,
        )
        mel = self._compute_offline_mel_spectrogram(silence)
        encoder = self._model.encoder
        conv_out = encoder.conv_stem(mel)
        if int(conv_out.shape[0]) <= int(encoder.config.sliding_window):
            embeds = encoder.encode_full(conv_out)
        else:
            chunks = [
                encoder.downsample_and_project(chunk)
                for chunk in encoder.encode_chunks(conv_out)
            ]
            embeds = self._mx.concatenate(chunks, axis=0) if chunks else None
        if embeds is None or int(embeds.shape[0]) != self._n_left_pad_tokens:
            raise RuntimeError(
                "Failed to prime Voxtral left-pad streaming prefix."
            )
        self._audio_embed_buffer = embeds
        self._encoder_position = int(conv_out.shape[0])
        self._mx.eval(self._audio_embed_buffer)

    def _next_streaming_mel_chunk(self, *, finalize: bool) -> Any | None:
        audio_source = self._final_audio_buffer if finalize else self._audio_buffer
        if audio_source is None:
            return None

        if finalize:
            source = audio_source
        else:
            with self._audio_lock:
                source = self._audio_buffer

        chunk = self._streaming_processor.next_chunk(
            audio_length=int(source.shape[0]),
            finalize=finalize,
        )
        if chunk is None:
            return None

        audio_chunk = source[chunk.start_sample : chunk.end_sample].copy()
        mel = self._compute_streaming_mel_spectrogram(
            audio_chunk,
            is_first_audio_chunk=chunk.is_first_audio_chunk,
        )
        mel_frames = int(mel.shape[1])
        if mel_frames == 0:
            return None
        if not chunk.is_first_audio_chunk and mel_frames < 2:
            return None
        self._streaming_processor.advance(
            mel_frames=mel_frames,
            is_first_audio_chunk=chunk.is_first_audio_chunk,
        )
        return mel

    def _encode_mel_chunk(self, mel: Any) -> Any | None:
        if mel.shape[1] == 0:
            return None

        encoder = self._model.encoder
        mel_input = mel.T[None, :, :].astype(encoder.conv_layers_0_conv.conv.weight.dtype)

        conv1 = encoder.conv_layers_0_conv
        if self._conv1_tail is not None:
            x = self._mx.concatenate([self._conv1_tail, mel_input], axis=1)
        else:
            x = self._mx.pad(mel_input, [(0, 0), (conv1.padding, 0), (0, 0)])
        self._conv1_tail = mel_input[:, -conv1.padding :, :] if conv1.padding > 0 else None
        x = self._nn.gelu(conv1.conv(x))

        conv2 = encoder.conv_layers_1_conv
        if self._conv2_tail is not None:
            x_input = self._mx.concatenate([self._conv2_tail, x], axis=1)
        else:
            x_input = self._mx.pad(x, [(0, 0), (conv2.padding, 0), (0, 0)])
        self._conv2_tail = x[:, -conv2.padding :, :] if conv2.padding > 0 else None
        x = self._nn.gelu(conv2.conv(x)).squeeze(0)

        if self._encoder_cache is None:
            sliding_window = int(encoder.config.sliding_window)
            self._encoder_cache = [
                self._RotatingKVCache(max_size=sliding_window, keep=0)
                for _ in encoder.transformer_layers
            ]

        chunk_len = int(x.shape[0])
        if chunk_len == 0:
            return None

        positions = self._mx.arange(
            self._encoder_position,
            self._encoder_position + chunk_len,
        )
        rope_cos, rope_sin = self._compute_rope_freqs(
            positions,
            head_dim=int(encoder.config.head_dim),
            theta=float(encoder.config.rope_theta),
        )
        sliding_window = int(encoder.config.sliding_window)
        for index, layer in enumerate(encoder.transformer_layers):
            mask = self._encoder_cache[index].make_mask(
                chunk_len,
                window_size=sliding_window,
            )
            x = layer(x, rope_cos, rope_sin, mask, cache=self._encoder_cache[index])
        x = encoder.transformer_norm(x)
        self._encoder_position += chunk_len

        if self._downsample_buffer is not None:
            x = self._mx.concatenate([self._downsample_buffer, x], axis=0)

        downsample = int(encoder.config.downsample_factor)
        n_complete = (int(x.shape[0]) // downsample) * downsample
        if n_complete == 0:
            self._downsample_buffer = x
            return None

        self._downsample_buffer = x[n_complete:] if int(x.shape[0]) > n_complete else None
        embeds = encoder.downsample_and_project(x[:n_complete])
        self._mx.eval(embeds)
        return embeds

    def _append_audio_embeds(self, new_embeds: Any) -> None:
        if new_embeds is None or int(new_embeds.shape[0]) == 0:
            return
        if self._audio_embed_buffer is None:
            self._audio_embed_buffer = new_embeds
        else:
            self._audio_embed_buffer = self._mx.concatenate(
                [self._audio_embed_buffer, new_embeds],
                axis=0,
            )
        self._mx.eval(self._audio_embed_buffer)

    def _decode_available(self) -> bool:
        progressed = False

        if not self._prefilled:
            if (
                self._audio_embed_buffer is None
                or int(self._audio_embed_buffer.shape[0]) < self._prefix_len
            ):
                return False
            prefix_embeds = self._audio_embed_buffer[: self._prefix_len] + self._prompt_text_embeds
            hidden, self._decoder_cache = self._model.decoder.forward(prefix_embeds, start_pos=0)
            logits = self._model.decoder.logits(hidden[-1])
            cache_arrays = [tensor for layer_cache in self._decoder_cache for tensor in layer_cache[:2]]
            self._mx.eval(logits, *cache_arrays)
            self._next_token = self._sample(logits)
            self._mx.async_eval(self._next_token)
            self._audio_embed_buffer = (
                self._audio_embed_buffer[self._prefix_len :]
                if int(self._audio_embed_buffer.shape[0]) > self._prefix_len
                else None
            )
            self._prefilled = True
            self._decode_position = self._prefix_len
            progressed = True

        while self._prefilled and self._audio_embed_buffer is not None and int(self._audio_embed_buffer.shape[0]) > 0:
            token_id = int(self._next_token.item())
            if token_id == self._eos_token_id:
                self._eos_reset_count += 1
                self._commit_current_segment()
                self._reset_decoder_state(preserve_stream_state=True)
                progressed = True
                if self._audio_embed_buffer is None:
                    break
                continue

            self._generated_tokens.append(token_id)
            decoded = self._model._tokenizer.decode(self._generated_tokens)
            if decoded and decoded != self._current_segment_text:
                self._current_segment_text = decoded
                self._emit_cumulative_text()

            step_embed = self._audio_embed_buffer[0] + self._model.decoder.embed_token(token_id)
            hidden, self._decoder_cache = self._model.decoder.forward(
                step_embed[None, :],
                start_pos=self._decode_position,
                cache=self._decoder_cache,
            )
            logits = self._model.decoder.logits(hidden.squeeze(0))
            self._next_token = self._sample(logits)
            self._mx.async_eval(self._next_token)
            self._audio_embed_buffer = (
                self._audio_embed_buffer[1:]
                if int(self._audio_embed_buffer.shape[0]) > 1
                else None
            )
            self._decode_position += 1
            progressed = True

            if len(self._generated_tokens) % 256 == 0:
                self._mx.clear_cache()

        return progressed

    def _commit_current_segment(self) -> None:
        segment = self._current_segment_text.strip()
        if segment:
            self._full_text = f"{self._full_text} {segment}".strip()
        self._current_segment_text = ""
        self._generated_tokens = []
        self._emit_cumulative_text(force=True)

    def _reset_decoder_state(self, *, preserve_stream_state: bool) -> None:
        self._decoder_cache = None
        self._prefilled = False
        self._next_token = None
        self._decode_position = self._prefix_len
        self._generated_tokens = []
        if not preserve_stream_state:
            self._audio_embed_buffer = None
            self._conv1_tail = None
            self._conv2_tail = None
            self._encoder_cache = None
            self._downsample_buffer = None
            self._encoder_position = 0

    def _emit_cumulative_text(self, *, force: bool = False) -> None:
        combined = self._final_text()
        if (force or combined != self._last_emitted_text) and combined:
            self._last_emitted_text = combined
            self._live_partial_seen = True
            if self._first_partial_at is None:
                self._first_partial_at = time.monotonic()
                self._log_debug(
                    "first partial "
                    f"latency_s={self._first_partial_latency():.2f} "
                    f"audio_s={self._audio_seconds_ingested:.2f}"
                )
            self._schedule_partial(combined)

    def _final_text(self) -> str:
        live_parts = [self._full_text.strip(), self._current_segment_text.strip()]
        live_text = " ".join(part for part in live_parts if part).strip()
        if self._live_partial_seen and live_text:
            return live_text
        return max((live_text, self._preview_text.strip()), key=len).strip()

    def _schedule_partial(self, text: str) -> None:
        self._loop.call_soon_threadsafe(self._emit_partial, text)

    def _schedule_error(self, message: str) -> None:
        self._loop.call_soon_threadsafe(self._emit_error, message)

    def _first_partial_latency(self) -> float:
        if self._first_partial_at is None:
            return -1.0
        return self._first_partial_at - self._session_started_at

    def _log_debug(self, message: str) -> None:
        runtime_log(f"[stt:{self.turn_id}] {message}")

    def _update_preview_text(self, text: str) -> None:
        if self._live_partial_seen or not text:
            return
        if len(text) < len(self._preview_text):
            return
        self._preview_text = text
        if self._first_partial_at is None:
            self._first_partial_at = time.monotonic()
            self._log_debug(
                "first partial via preview "
                f"latency_s={self._first_partial_latency():.2f} "
                f"audio_s={self._audio_seconds_ingested:.2f}"
            )
        if text != self._last_emitted_text:
            self._last_emitted_text = text
            self._schedule_partial(text)

    def _transcribe_snapshot(self, pcm_bytes: bytes, transcription_delay_ms: int) -> str:
        audio_array = self._pcm_bytes_to_audio_array(pcm_bytes)
        with self._model_lock:
            result = self._model.generate(
                audio_array,
                transcription_delay_ms=transcription_delay_ms,
            )
        return getattr(result, "text", "").strip()

    def _sample(self, logits: Any) -> Any:
        if self._temperature <= 0:
            return self._mx.argmax(logits)
        return self._mx.random.categorical(logits * (1.0 / self._temperature))

    def _compute_streaming_mel_spectrogram(
        self,
        audio_chunk: np.ndarray,
        *,
        is_first_audio_chunk: bool,
    ) -> Any:
        if is_first_audio_chunk:
            combined = np.concatenate(
                [np.zeros(self._window_size // 2, dtype=np.float32), audio_chunk]
            )
        else:
            combined = audio_chunk
        return self._compute_mel_from_audio(
            combined,
            drop_last_frame=False,
        )

    def _compute_offline_mel_spectrogram(self, audio_chunk: np.ndarray) -> Any:
        pad_size = self._window_size // 2
        padded = np.pad(audio_chunk, (pad_size, pad_size), mode="constant")
        return self._compute_mel_from_audio(
            padded,
            drop_last_frame=True,
        )

    def _compute_mel_from_audio(
        self,
        audio_chunk: np.ndarray,
        *,
        drop_last_frame: bool,
    ) -> Any:
        audio_mx = self._mx.array(audio_chunk)
        n_frames = 1 + (int(audio_mx.shape[0]) - self._window_size) // self._hop_length
        if n_frames <= 0:
            return self._mx.zeros((self._model._mel_filters.shape[1], 0))

        window = self._mx.array(np.hanning(self._window_size + 1)[:-1].astype(np.float32))
        t = self._mx.arange(self._window_size)[None, :]
        starts = (self._mx.arange(n_frames) * self._hop_length)[:, None]
        indices = starts + t
        frames = audio_mx[indices] * window[None, :]
        n_freqs = self._window_size // 2 + 1
        k = self._mx.arange(n_freqs).astype(self._mx.float32)[:, None]
        n = self._mx.arange(self._window_size).astype(self._mx.float32)[None, :]
        angles = -2.0 * math.pi * (k @ n) / self._window_size
        dft_real = self._mx.cos(angles)
        dft_imag = self._mx.sin(angles)
        spec_real = frames @ dft_real.T
        spec_imag = frames @ dft_imag.T
        magnitudes = spec_real**2 + spec_imag**2
        if drop_last_frame and int(magnitudes.shape[0]) > 0:
            magnitudes = magnitudes[:-1, :]
        mel_spec = magnitudes @ self._model._mel_filters
        log_spec = self._mx.log10(self._mx.maximum(mel_spec, 1e-10))
        global_log_mel_max = float(self._model.config.audio_encoding_args.global_log_mel_max)
        log_spec = self._mx.maximum(log_spec, global_log_mel_max - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.T

    @staticmethod
    def _pcm_bytes_to_audio_array(pcm_bytes: bytes) -> np.ndarray:
        if not pcm_bytes:
            return np.array([], dtype=np.float32)
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    def _compute_rope_freqs(self, positions: Any, *, head_dim: int, theta: float) -> tuple[Any, Any]:
        freqs = 1.0 / (
            theta ** (self._mx.arange(0, head_dim, 2, dtype=self._mx.float32) / head_dim)
        )
        angles = positions[:, None].astype(self._mx.float32) * freqs[None, :]
        return (self._mx.cos(angles), self._mx.sin(angles))

    @staticmethod
    def _num_delay_tokens(
        *,
        delay_ms: int,
        sample_rate: int,
        hop_length: int,
        frame_rate: float,
    ) -> int:
        raw_audio_length_per_tok = int(sample_rate / frame_rate)
        audio_length_per_tok = raw_audio_length_per_tok // hop_length
        delay_len = int(delay_ms / 1000.0 * sample_rate)
        if delay_len % hop_length != 0:
            audio_len = math.ceil(delay_len / hop_length - 1)
        else:
            audio_len = delay_len // hop_length
        return math.ceil(audio_len / audio_length_per_tok)


class MacAutomatorClient:
    def __init__(self) -> None:
        self._session = None
        self._session_cm = None
        self._stdio_cm = None
        self._initialized = False

    async def ensure_ready(self) -> None:
        if self._initialized:
            return
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = os.environ.copy()
        env.setdefault("LOG_LEVEL", "ERROR")
        env.setdefault("KB_PARSING", "lazy")
        env.setdefault("NO_COLOR", "1")
        env.setdefault("BUN_INSTALL_CACHE_DIR", str(BUN_CACHE_ROOT))

        server_params = StdioServerParameters(
            command="bun",
            args=["x", "--silent", MACOS_AUTOMATOR_PACKAGE],
            env=env,
            cwd=str(MCP_ROOT),
        )
        self._stdio_cm = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read_stream, write_stream)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        self._initialized = True

    async def close(self) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self._session = None
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None
        self._initialized = False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        await self.ensure_ready()
        result = await self._session.call_tool(name, arguments=arguments)
        content = []
        for item in getattr(result, "content", []):
            text = getattr(item, "text", None)
            if text:
                content.append(text)
        if not content and getattr(result, "structuredContent", None):
            content.append(json.dumps(result.structuredContent))
        return "\n".join(content) or "Tool completed without text output."

    async def list_tools(self) -> list[ToolDefinition]:
        await self.ensure_ready()
        result = await self._session.list_tools()
        return [
            ToolDefinition(
                name=tool.name,
                description=getattr(tool, "description", "") or "",
                input_schema=getattr(tool, "inputSchema", {}) or {},
            )
            for tool in getattr(result, "tools", [])
        ]


class RuntimeHost:
    def __init__(self) -> None:
        self.model_states: dict[str, ModelState] = {}
        self.turns: dict[str, TurnBuffer] = {}
        self.turn_states: dict[str, TurnExecutionState] = {}
        self.history: list[dict[str, Any]] = []
        self.pending_tools: dict[str, dict[str, Any]] = {}
        self.install_tasks: dict[str, asyncio.Task[None]] = {}
        self.install_processes: dict[str, asyncio.subprocess.Process] = {}
        self.agent_model = None
        self.agent_processor = None
        self.tts_model = None
        self.stt_model = None
        self.mcp_client = MacAutomatorClient()
        self.tool_definitions: dict[str, ToolDefinition] = {}
        self.cancelled_turns: set[str] = set()
        self.should_exit = False

    def ensure_turn_state(self, turn_id: str) -> TurnExecutionState:
        state = self.turn_states.get(turn_id)
        if state is None:
            state = TurnExecutionState(turn_id=turn_id)
            self.turn_states[turn_id] = state
        return state

    def finish_turn_task(self, turn_id: str, task: asyncio.Task[None]) -> None:
        state = self.turn_states.get(turn_id)
        if state is not None and state.task is task:
            state.task = None
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        self.cleanup_turn_state(turn_id)

    def cleanup_turn_state(self, turn_id: str) -> None:
        state = self.turn_states.get(turn_id)
        if state is None:
            return
        has_pending_tool = any(
            pending.get("turn_id") == turn_id
            for pending in self.pending_tools.values()
        )
        has_buffer = turn_id in self.turns
        if state.task is None and not has_pending_tool and not has_buffer:
            self.turn_states.pop(turn_id, None)
            self.cancelled_turns.discard(turn_id)

    def turn_cancel_event(self, turn_id: str | None) -> threading.Event | None:
        if turn_id is None:
            return None
        state = self.turn_states.get(turn_id)
        return state.cancel_event if state is not None else None

    def is_turn_cancelled(self, turn_id: str) -> bool:
        state = self.turn_states.get(turn_id)
        return turn_id in self.cancelled_turns or (
            state is not None and state.cancel_event.is_set()
        )

    def raise_if_turn_cancelled(self, turn_id: str) -> None:
        if self.is_turn_cancelled(turn_id):
            raise asyncio.CancelledError

    def emit_turn_finished(
        self,
        turn_id: str,
        *,
        status: str,
        message: str | None = None,
        is_final: bool = True,
    ) -> None:
        state = self.ensure_turn_state(turn_id)
        if state.finished_emitted:
            return
        state.finished_emitted = True
        payload: dict[str, Any] = {
            "type": "turn_finished",
            "turnID": turn_id,
            "status": status,
            "isFinal": is_final,
        }
        if message is not None:
            payload["message"] = message
        self.emit(payload)

    def clear_pending_tools_for_turn(self, turn_id: str) -> None:
        for tool_call_id, pending in list(self.pending_tools.items()):
            if pending.get("turn_id") == turn_id:
                self.pending_tools.pop(tool_call_id, None)

    def start_turn_task(
        self,
        turn_id: str,
        task_factory: Callable[[], Awaitable[None]],
    ) -> None:
        state = self.ensure_turn_state(turn_id)
        state.finished_emitted = False
        state.cancel_event.clear()
        self.cancelled_turns.discard(turn_id)
        existing_task = state.task
        if existing_task is not None and not existing_task.done():
            existing_task.cancel()

        async def run_task() -> None:
            try:
                await task_factory()
            except asyncio.CancelledError:
                self.emit_turn_finished(turn_id, status="cancelled", is_final=True)
            except Exception as exc:  # noqa: BLE001
                self.emit({"type": "error", "message": str(exc)})
                self.emit_turn_finished(
                    turn_id,
                    status="failed",
                    message=str(exc),
                    is_final=True,
                )

        task = asyncio.create_task(run_task())
        state.task = task
        task.add_done_callback(
            lambda finished_task, turn_id=turn_id: self.finish_turn_task(
                turn_id, finished_task
            )
        )

    def sampling_preset(
        self,
        *,
        enable_thinking: bool,
        task_profile: str,
    ) -> SamplingPreset:
        if enable_thinking:
            if task_profile == "coding":
                return QWEN_TEXT_ONLY_SAMPLING_PRESETS["thinking_coding"]
            if task_profile in {"general", "reasoning"}:
                return QWEN_TEXT_ONLY_SAMPLING_PRESETS["thinking_general"]
        else:
            if task_profile == "reasoning":
                return QWEN_TEXT_ONLY_SAMPLING_PRESETS["instruct_reasoning"]
            if task_profile in {"general", "coding"}:
                return QWEN_TEXT_ONLY_SAMPLING_PRESETS["instruct_general"]
        raise ValueError(f"Unsupported sampling task profile: {task_profile}")

    async def run(self) -> None:
        self._ensure_dirs()
        self._purge_legacy_models()
        self.model_states = self._load_manifest()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        while not reader.at_eof():
            line = await reader.readline()
            if not line:
                break
            command: dict[str, Any] | None = None
            payload = line.decode("utf-8").strip()
            if not payload:
                continue
            try:
                command = json.loads(payload)
                await self.handle_command(command)
                if self.should_exit:
                    break
            except Exception as exc:  # noqa: BLE001
                self.emit({"type": "error", "message": str(exc)})
                if isinstance(command, dict) and command.get("turnID"):
                    self.emit(
                        {
                            "type": "turn_finished",
                            "turnID": command["turnID"],
                            "status": "failed",
                            "message": str(exc),
                            "isFinal": True,
                        }
                    )

    async def handle_command(self, command: dict[str, Any]) -> None:
        command_type = command["type"]
        if command_type == "bootstrap":
            await self.bootstrap()
        elif command_type == "download_model":
            self.start_installable_download(command["installableID"])
        elif command_type == "delete_model":
            await self.delete_installable(command["installableID"])
        elif command_type == "delete_underlying_model":
            await self.delete_underlying_model(command["modelID"])
        elif command_type == "warm_model":
            await self.warm_model(command["modelID"], command.get("arguments") or {})
        elif command_type == "start_recording":
            await self.start_recording(command)
        elif command_type == "append_audio_chunk":
            await self.append_audio_chunk(command)
        elif command_type == "stop_recording":
            turn_id = command["turnID"]
            self.start_turn_task(
                turn_id,
                lambda turn_id=turn_id: self.stop_recording(turn_id),
            )
        elif command_type == "send_text":
            turn_id = command["turnID"]
            attachments = self.normalize_attachments(command.get("attachments"))
            speak_reply = bool(command.get("speakReply"))
            voice_preset = (command.get("arguments") or {}).get(
                "voice_preset",
                "casual_male",
            )
            self.start_turn_task(
                turn_id,
                lambda turn_id=turn_id, attachments=attachments, speak_reply=speak_reply, voice_preset=voice_preset: self.run_turn(
                    turn_id=turn_id,
                    user_text=command.get("text", ""),
                    attachments=attachments,
                    speak_reply=speak_reply,
                    voice_preset=voice_preset,
                ),
            )
        elif command_type == "speak_text":
            turn_id = command["turnID"]
            voice_preset = (command.get("arguments") or {}).get(
                "voice_preset",
                "casual_male",
            )
            self.start_turn_task(
                turn_id,
                lambda turn_id=turn_id, text=command.get("text", ""), voice_preset=voice_preset: self.run_speech_turn(
                    turn_id=turn_id,
                    text=text,
                    voice_preset=voice_preset,
                ),
            )
        elif command_type == "approve_tool":
            turn_id = command["turnID"]
            tool_call_id = command["toolCallID"]
            self.start_turn_task(
                turn_id,
                lambda turn_id=turn_id, tool_call_id=tool_call_id: self.resume_tool(
                    turn_id,
                    tool_call_id,
                    approved=True,
                ),
            )
        elif command_type == "deny_tool":
            turn_id = command["turnID"]
            tool_call_id = command["toolCallID"]
            self.start_turn_task(
                turn_id,
                lambda turn_id=turn_id, tool_call_id=tool_call_id: self.resume_tool(
                    turn_id,
                    tool_call_id,
                    approved=False,
                ),
            )
        elif command_type == "cancel":
            turn_id = command.get("turnID")
            if turn_id:
                await self.cancel_turn(turn_id)
            else:
                await self.cancel_all()
        elif command_type == "replace_history":
            await self.replace_history(command.get("history") or [])
        elif command_type == "reset_conversation":
            await self.reset_conversation()
        elif command_type == "shutdown":
            await self.shutdown()
        else:
            self.emit({"type": "error", "message": f"Unhandled command type: {command_type}"})

    def start_installable_download(self, installable_id: str) -> None:
        existing_task = self.install_tasks.get(installable_id)
        if existing_task is not None and not existing_task.done():
            return

        task = asyncio.create_task(self.download_installable(installable_id))
        self.install_tasks[installable_id] = task
        task.add_done_callback(lambda finished_task, installable_id=installable_id: self.finish_installable_download(installable_id, finished_task))

    def finish_installable_download(self, installable_id: str, task: asyncio.Task[None]) -> None:
        if self.install_tasks.get(installable_id) is task:
            self.install_tasks.pop(installable_id, None)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            self.emit({"type": "error", "message": f"Failed to download {installable_id}: {exc}"})

    async def bootstrap(self) -> None:
        self.emit(
            {
                "type": "bootstrap_progress",
                "stage": "Checking runtime prerequisites",
                "message": "Verifying Bun, runtime directories, and local support files.",
            }
        )
        if shutil.which("bun") is None:
            self.emit({"type": "error", "message": "Bun is required to run macos-automator-mcp."})
        await asyncio.sleep(0.05)
        self.emit(
            {
                "type": "bootstrap_progress",
                "stage": "Connecting automation tools",
                "message": "Loading the live MCP tool schema from macos-automator-mcp.",
            }
        )
        try:
            await self.refresh_tool_definitions(force=True)
        except Exception as exc:  # noqa: BLE001
            self.emit({"type": "error", "message": f"Failed to connect to macOS automation tools: {exc}"})
        installed_models = sum(1 for state in self.model_states.values() if state.installed)
        self.emit(
            {
                "type": "bootstrap_progress",
                "stage": "Checking installed models",
                "message": (
                    "Looking for previously downloaded local model weights."
                    if installed_models > 0
                    else "No local models found yet. You will be prompted to download them next."
                ),
            }
        )
        for model_id in MODEL_SPECS:
            self.emit_model_state(model_id, message=self.model_states.get(model_id, ModelState()).last_error)
        self.emit(
            {
                "type": "bootstrap_progress",
                "stage": "Runtime ready",
                "message": (
                    "Runtime connected. Warming installed models next."
                    if installed_models > 0
                    else "Runtime connected. Waiting for model downloads."
                ),
            }
        )

    async def download_installable(self, installable_id: str) -> None:
        model_ids = INSTALLABLES[installable_id]
        planned_download_sizes = await asyncio.to_thread(self.plan_installable_download, installable_id)
        installable_total_bytes = sum(planned_download_sizes.values())
        completed_bytes = 0
        try:
            for index, model_id in enumerate(model_ids):
                self.model_states.setdefault(model_id, ModelState())
                self.model_states[model_id].installed = False
                self.model_states[model_id].warm = False
                self.model_states[model_id].last_error = None
                self.emit_model_state(model_id, install_state="downloading", warm_state="cold")
                spec = MODEL_SPECS[model_id]
                self.emit(
                    {
                        "type": "bootstrap_progress",
                        "stage": f"Downloading {model_id}",
                        "message": f"Fetching {spec['repo']}…",
                    }
                )

                try:
                    local_dir, observed_model_total = await self.run_download_process(
                        installable_id,
                        model_id,
                        completed_bytes=completed_bytes,
                        installable_total_bytes=installable_total_bytes,
                    )
                    self.apply_model_compatibility_fixes(model_id, Path(local_dir))
                    self.model_states[model_id] = ModelState(installed=True, warm=False, path=str(local_dir), last_error=None)
                    completed_bytes += max(planned_download_sizes.get(model_id, 0), observed_model_total)
                    installable_total_bytes = max(installable_total_bytes, completed_bytes)
                    self.emit_download_progress(
                        installable_id,
                        model_id,
                        bytes_downloaded=completed_bytes,
                        bytes_total=installable_total_bytes,
                        eta_seconds=0.0 if installable_total_bytes > 0 else None,
                    )
                except asyncio.CancelledError:
                    for cancelled_model_id in model_ids[index:]:
                        await self.reset_model_install(cancelled_model_id)
                        self.emit_model_state(cancelled_model_id)
                    self._save_manifest()
                    raise
                except Exception as exc:  # noqa: BLE001
                    await self.reset_model_install(model_id, last_error=str(exc))

                self.emit_model_state(
                    model_id,
                    install_state="installed" if self.model_states[model_id].installed else "failed",
                    warm_state="cold",
                    message=self.model_states[model_id].last_error,
                )

            self._save_manifest()
        finally:
            await self.stop_download_process(installable_id)

    async def delete_installable(self, installable_id: str) -> None:
        await self.cancel_installable_download(installable_id)
        for model_id in INSTALLABLES[installable_id]:
            await self.delete_model_files(model_id)
        self._save_manifest()

    async def delete_underlying_model(self, model_id: str) -> None:
        for installable_id, model_ids in INSTALLABLES.items():
            if model_id in model_ids:
                await self.cancel_installable_download(installable_id)
        await self.delete_model_files(model_id)
        self._save_manifest()

    async def delete_model_files(self, model_id: str) -> None:
        await self.unload_model(model_id)
        model_dir = MODELS_ROOT / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        self.model_states[model_id] = ModelState(installed=False, warm=False, path=None, last_error=None)
        self.emit_model_state(model_id)

    async def warm_model(self, model_id: str, arguments: dict[str, Any]) -> None:
        state = self.model_states.setdefault(model_id, ModelState())
        if not state.installed or not state.path:
            self.emit_model_state(model_id, install_state="missing", warm_state="cold", message="Model is not installed.")
            return
        if state.warm:
            self.emit_model_state(model_id)
            return

        self.emit_model_state(model_id, warm_state="warming")
        try:
            self.apply_model_compatibility_fixes(model_id, Path(state.path))
            if model_id == "agent_model":
                self.ensure_agent_runtime_dependencies()
                from mlx_vlm import load

                self.agent_model, self.agent_processor = await asyncio.to_thread(load, state.path)
            elif model_id == "tts_model":
                from mlx_audio.tts.utils import load

                self.tts_model = await asyncio.to_thread(load, state.path)
                self.patch_tts_runtime(self.tts_model, Path(state.path))
            elif model_id == "stt_model":
                from mlx_audio.stt.utils import load

                self.stt_model = await asyncio.to_thread(load, state.path)
            state.warm = True
            state.last_error = None
        except Exception as exc:  # noqa: BLE001
            state.warm = False
            state.last_error = str(exc)
        self.emit_model_state(
            model_id,
            install_state="installed" if state.installed else "missing",
            warm_state="warm" if state.warm else ("error" if state.last_error else "cold"),
            message=state.last_error,
        )
        self._save_manifest()

    async def unload_model(self, model_id: str) -> None:
        if model_id == "agent_model":
            self.agent_model = None
            self.agent_processor = None
        elif model_id == "tts_model":
            self.tts_model = None
        elif model_id == "stt_model":
            self.stt_model = None
        state = self.model_states.setdefault(model_id, ModelState())
        state.warm = False

    async def start_recording(self, command: dict[str, Any]) -> None:
        turn_id = command["turnID"]
        arguments = command.get("arguments") or {}
        await self.warm_model("stt_model", {})
        self.require_model("stt_model", self.stt_model)
        buffer = TurnBuffer(
            turn_id=turn_id,
            speak_reply=bool(command.get("speakReply")),
            voice_preset=str(arguments.get("voice_preset", "casual_male")),
            attachments=self.normalize_attachments(command.get("attachments")),
        )
        buffer.stt_session = self.create_stt_session(turn_id)
        self.turns[turn_id] = buffer

    async def append_audio_chunk(self, command: dict[str, Any]) -> None:
        turn_id = command["turnID"]
        buffer = self.turns.get(turn_id)
        if not buffer:
            return
        payload = base64.b64decode(command["chunkBase64"])
        buffer.pcm_bytes.extend(payload)
        if buffer.stt_session is not None:
            buffer.stt_session.append_pcm_bytes(payload)

    async def stop_recording(self, turn_id: str) -> None:
        buffer = self.turns.get(turn_id)
        if not buffer:
            return
        buffer.stopped = True
        try:
            transcript = (
                await buffer.stt_session.finish()
                if buffer.stt_session is not None
                else await self.transcribe_buffer(
                    bytes(buffer.pcm_bytes),
                    transcription_delay_ms=FINAL_TRANSCRIPTION_DELAY_MS,
                )
            )
            self.raise_if_turn_cancelled(turn_id)
            self.emit({"type": "transcript_final", "turnID": turn_id, "text": transcript})
            await self.run_turn(
                turn_id=turn_id,
                user_text=transcript,
                attachments=buffer.attachments,
                speak_reply=buffer.speak_reply,
                voice_preset=buffer.voice_preset,
            )
        finally:
            self.turns.pop(turn_id, None)
            self.cleanup_turn_state(turn_id)

    async def transcribe_buffer(self, pcm_bytes: bytes, *, transcription_delay_ms: int) -> str:
        await self.warm_model("stt_model", {})
        self.require_model("stt_model", self.stt_model)
        audio_array = self.pcm_bytes_to_audio_array(pcm_bytes)
        result = await asyncio.to_thread(
            self.stt_model.generate,
            audio_array,
            transcription_delay_ms=transcription_delay_ms,
        )
        return getattr(result, "text", "").strip()

    async def run_turn(
        self,
        turn_id: str,
        user_text: str,
        attachments: list[dict[str, Any]] | None,
        speak_reply: bool,
        voice_preset: str,
    ) -> None:
        attachments = self.normalize_attachments(attachments)
        resolved_user_text = user_text.strip()
        if not resolved_user_text and attachments:
            resolved_user_text = "Describe this image."

        if not resolved_user_text and not attachments:
            self.emit_turn_finished(
                turn_id,
                status="failed",
                message="No text captured.",
                is_final=True,
            )
            return

        self.raise_if_turn_cancelled(turn_id)
        user_message = self.build_user_message(resolved_user_text, attachments)
        direct_tool_call = self.direct_tool_call_for_user_text(resolved_user_text)
        if direct_tool_call is not None:
            tool_plan = {"assistant_message": "", "tool_call": direct_tool_call}
            tool_call = direct_tool_call
        elif self.should_use_tool_planner(resolved_user_text, attachments):
            tool_plan = await self.generate_agent_plan(user_message, turn_id=turn_id)
            tool_call = self.choose_tool_call(resolved_user_text, tool_plan)
        else:
            assistant_message = await self.generate_direct_answer(
                user_message,
                turn_id=turn_id,
            )
            self.raise_if_turn_cancelled(turn_id)
            self.history.extend(
                [
                    user_message,
                    {"role": "assistant", "content": assistant_message},
                ]
            )
            if speak_reply:
                await self.speak(turn_id, assistant_message, voice_preset)
            self.raise_if_turn_cancelled(turn_id)
            self.emit_turn_finished(turn_id, status="finished", is_final=True)
            return
        self.raise_if_turn_cancelled(turn_id)

        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name", "")
            tool_arguments = tool_call.get("arguments") or {}
            tool_call_id = f"tool-{turn_id}"
            safety_class, approval_state, execution_state = self.classify_tool(tool_name, tool_arguments)
            summary = f"{tool_name} • {approval_state.replace('_', ' ')}"
            self.pending_tools[tool_call_id] = {
                "turn_id": turn_id,
                "user_text": resolved_user_text,
                "user_message": user_message,
                "tool_name": tool_name,
                "tool_arguments": tool_arguments,
                "voice_preset": voice_preset,
                "speak_reply": speak_reply,
            }
            self.emit(
                {
                    "type": "tool_proposed",
                    "turnID": turn_id,
                    "toolCallID": tool_call_id,
                    "toolName": tool_name,
                    "toolSummary": summary,
                    "toolArguments": tool_arguments,
                    "safetyClass": safety_class,
                    "approvalState": approval_state,
                    "executionState": execution_state,
                }
            )
            if approval_state == "notRequired":
                await self.resume_tool(turn_id, tool_call_id, approved=True)
            return

        await self.stream_assistant_text(turn_id, tool_plan.get("assistant_message", ""))
        self.raise_if_turn_cancelled(turn_id)
        self.history.extend(
            [
                user_message,
                {"role": "assistant", "content": tool_plan.get("assistant_message", "")},
            ]
        )
        if speak_reply:
            await self.speak(turn_id, tool_plan.get("assistant_message", ""), voice_preset)
        self.raise_if_turn_cancelled(turn_id)
        self.emit_turn_finished(turn_id, status="finished", is_final=True)

    async def run_speech_turn(self, turn_id: str, text: str, voice_preset: str) -> None:
        if not text.strip():
            self.emit_turn_finished(turn_id, status="finished", is_final=True)
            return
        await self.speak(turn_id, text, voice_preset)
        self.raise_if_turn_cancelled(turn_id)
        self.emit_turn_finished(turn_id, status="finished", is_final=True)

    async def resume_tool(self, turn_id: str, tool_call_id: str, approved: bool) -> None:
        pending = self.pending_tools.pop(tool_call_id, None)
        if not pending:
            return
        self.raise_if_turn_cancelled(turn_id)
        if not approved:
            tool_result = "Tool execution was denied by the user."
            self.emit(
                {
                    "type": "tool_finished",
                    "turnID": turn_id,
                    "toolCallID": tool_call_id,
                    "executionState": "finished",
                    "output": tool_result,
                }
            )
        else:
            self.emit({"type": "tool_started", "turnID": turn_id, "toolCallID": tool_call_id, "executionState": "running"})
            try:
                tool_result = await self.execute_tool(pending["tool_name"], pending["tool_arguments"])
                self.raise_if_turn_cancelled(turn_id)
                self.emit({"type": "tool_output", "turnID": turn_id, "toolCallID": tool_call_id, "output": tool_result})
                self.emit({"type": "tool_finished", "turnID": turn_id, "toolCallID": tool_call_id, "executionState": "finished", "output": tool_result})
            except Exception as exc:  # noqa: BLE001
                if self.is_turn_cancelled(turn_id):
                    raise
                tool_result = f"Tool failed: {exc}"
                self.emit({"type": "tool_finished", "turnID": turn_id, "toolCallID": tool_call_id, "executionState": "failed", "output": tool_result})

        final_plan = await self.generate_followup_answer(
            pending["user_message"],
            pending["tool_name"],
            pending["tool_arguments"],
            tool_result,
            turn_id=turn_id,
        )
        await self.stream_assistant_text(turn_id, final_plan.get("assistant_message", ""))
        self.raise_if_turn_cancelled(turn_id)
        self.history.extend(
            [
                pending["user_message"],
                {"role": "assistant", "content": final_plan.get("assistant_message", "")},
            ]
        )
        if pending["speak_reply"]:
            await self.speak(
                turn_id,
                final_plan.get("assistant_message", ""),
                pending["voice_preset"],
            )
        self.raise_if_turn_cancelled(turn_id)
        self.emit_turn_finished(turn_id, status="finished", is_final=True)

    async def execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> str:
        result = await self.mcp_client.call_tool(tool_name, tool_arguments)
        filtered_lines = [
            line for line in result.splitlines()
            if not line.startswith("MacOS Automator MCP v")
        ]
        return "\n".join(filtered_lines).strip() or result.strip()

    async def generate_agent_plan(
        self,
        user_message: dict[str, Any],
        *,
        turn_id: str,
    ) -> dict[str, Any]:
        await self.refresh_tool_definitions()
        await self.warm_model("agent_model", {})
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a local macOS voice agent. "
                    "Return strict JSON with keys assistant_message and tool_call. "
                    "tool_call must be null or an object with name and arguments. "
                    "Use at most one tool call. "
                    "assistant_message may contain concise Markdown such as paragraphs, bullet lists, emphasis, inline code, and fenced code blocks when helpful. "
                    "Only use the exact tool names and argument keys listed below.\n\n"
                    f"{self.tool_planner_guide()}\n\n"
                    "Rules:\n"
                    "- Use get_scripting_tips only for discovery. Never call it with empty arguments. Prefer search_term for natural-language queries.\n"
                    "- Use execute_script for concrete actions or live state lookups on this Mac. It can run AppleScript or JavaScript for Automation and is how you open apps, click UI, read state, and automate the Mac.\n"
                    "- For requests to open or launch an app, prefer execute_script with AppleScript such as {\"script_content\": \"tell application \\\"Mail\\\" to activate\", \"language\": \"applescript\"}.\n"
                    "- For current-machine questions like computer name, macOS version, chip, RAM, hostname, model, specs, or general questions about this Mac/computer, prefer execute_script with {\"script_content\": \"return system info\", \"language\": \"applescript\"}.\n"
                    "- Do not claim you lack access to the user's computer details when the user is asking about this Mac. Inspect the machine with execute_script first, then answer from the live result.\n"
                    "- Requests like 'tell me about my computer', 'what Mac is this', 'what are my specs', or 'how much RAM do I have' should trigger a live execute_script lookup, not a generic refusal.\n"
                    "- If you choose execute_script, provide one of kb_script_id, script_content, or script_path.\n"
                    "- If you set output_format_mode for execute_script, only use auto, human_readable, structured_error, structured_output_and_error, or direct. Prefer omitting it unless you need a specific format.\n"
                    "- If the user asks for a direct explanation that does not require live macOS state or automation, set tool_call to null.\n"
                    "- Keep assistant_message short when a tool will run.\n"
                    "- The outer response must still be valid JSON even when assistant_message contains Markdown."
                ),
            }
        ]
        messages.extend(self.history[-6:])
        messages.append(user_message)
        return await self.generate_json(
            messages,
            turn_id=turn_id,
            enable_thinking=False,
            task_profile="reasoning",
            max_tokens=384,
        )

    async def generate_followup_answer(
        self,
        user_message: dict[str, Any],
        tool_name: str,
        tool_arguments: dict[str, Any],
        tool_result: str,
        *,
        turn_id: str,
    ) -> dict[str, Any]:
        await self.warm_model("agent_model", {})
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a local macOS voice agent. "
                    "Return strict JSON with keys assistant_message and tool_call. "
                    "tool_call must be null. "
                    "assistant_message may contain concise Markdown such as paragraphs, bullet lists, emphasis, inline code, and fenced code blocks when helpful. "
                    "Summarize the completed tool result clearly."
                ),
            },
        ]
        messages.extend(self.history[-6:])
        messages.extend(
            [
                user_message,
            {
                "role": "assistant",
                "content": (
                    f"Tool {tool_name} was executed with arguments {json.dumps(tool_arguments)}.\n"
                    f"Tool result:\n{tool_result}"
                ),
            },
            ]
        )
        return await self.generate_json(
            messages,
            turn_id=turn_id,
            enable_thinking=False,
            task_profile="general",
            max_tokens=512,
        )

    async def generate_direct_answer(
        self,
        user_message: dict[str, Any],
        *,
        turn_id: str,
    ) -> str:
        await self.warm_model("agent_model", {})
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a local macOS assistant. "
                    "Answer the user's request directly and concisely in Markdown. "
                    "Use the conversation context when relevant. "
                    "Do not invent live macOS state or claim you performed an action."
                ),
            },
        ]
        messages.extend(self.history[-6:])
        messages.append(user_message)
        return await self.generate_text(
            messages,
            turn_id=turn_id,
            enable_thinking=False,
            task_profile="general",
            max_tokens=512,
            on_chunk=lambda chunk: self.emit(
                {
                    "type": "assistant_delta",
                    "turnID": turn_id,
                    "text": chunk,
                }
            ),
        )

    async def generate_json(
        self,
        messages: list[dict[str, Any]],
        *,
        turn_id: str,
        enable_thinking: bool,
        task_profile: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        raw = await self.generate_text(
            messages,
            turn_id=turn_id,
            enable_thinking=enable_thinking,
            task_profile=task_profile,
            max_tokens=max_tokens,
        )
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"assistant_message": raw, "tool_call": None}
        return self.normalize_generation_payload(parsed, raw)

    async def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        turn_id: str,
        enable_thinking: bool,
        task_profile: str,
        max_tokens: int,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import get_chat_template

        self.require_model("agent_model", self.agent_model)
        self.require_model("agent_model", self.agent_processor)
        preset = self.sampling_preset(
            enable_thinking=enable_thinking,
            task_profile=task_profile,
        )
        normalized_messages, images = self.normalize_multimodal_messages(messages)
        prompt = get_chat_template(
            self.agent_processor,
            normalized_messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        cancel_event = self.turn_cancel_event(turn_id)
        queue_items: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def stream_tokens() -> None:
            try:
                for result in stream_generate(
                    model=self.agent_model,
                    processor=self.agent_processor,
                    prompt=prompt,
                    image=images or None,
                    max_tokens=max_tokens,
                    temperature=preset.temperature,
                    top_p=preset.top_p,
                    min_p=preset.min_p,
                    top_k=preset.top_k,
                    repetition_penalty=preset.repetition_penalty,
                    verbose=False,
                ):
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    text = getattr(result, "text", "")
                    if not text:
                        continue
                    loop.call_soon_threadsafe(queue_items.put_nowait, ("chunk", text))
                    if cancel_event is not None and cancel_event.is_set():
                        break
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue_items.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue_items.put_nowait, ("done", None))

        threading.Thread(target=stream_tokens, daemon=True).start()

        collected_chunks: list[str] = []
        try:
            while True:
                item_type, value = await queue_items.get()
                if item_type == "chunk":
                    self.raise_if_turn_cancelled(turn_id)
                    text_chunk = str(value)
                    collected_chunks.append(text_chunk)
                    if on_chunk is not None:
                        on_chunk(text_chunk)
                elif item_type == "error":
                    raise value
                else:
                    break
            self.raise_if_turn_cancelled(turn_id)
        except asyncio.CancelledError:
            if cancel_event is not None:
                cancel_event.set()
            raise

        return "".join(collected_chunks)

    @staticmethod
    def assistant_stream_chunks(text: str) -> list[str]:
        if not text:
            return []
        return [match.group(0) for match in re.finditer(r"\s+|\S+", text)]

    async def stream_assistant_text(self, turn_id: str, text: str) -> None:
        chunks = self.assistant_stream_chunks(text)
        if not chunks:
            return
        for chunk in chunks:
            self.raise_if_turn_cancelled(turn_id)
            self.emit({"type": "assistant_delta", "turnID": turn_id, "text": chunk})
            await asyncio.sleep(0.025)

    async def speak(self, turn_id: str, text: str, voice_preset: str) -> None:
        if not text.strip():
            return
        await self.warm_model("tts_model", {})
        self.require_model("tts_model", self.tts_model)
        cancel_event = self.turn_cancel_event(turn_id)
        queue_items: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def stream_audio() -> None:
            try:
                for pcm_bytes in self.iter_tts_pcm_chunks(
                    text=text,
                    voice_preset=voice_preset,
                    cancel_event=cancel_event,
                ):
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue_items.put_nowait, ("chunk", pcm_bytes))
                    if cancel_event is not None and cancel_event.is_set():
                        break
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue_items.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue_items.put_nowait, ("done", None))

        threading.Thread(target=stream_audio, daemon=True).start()

        try:
            while True:
                item_type, value = await queue_items.get()
                if item_type == "chunk":
                    self.raise_if_turn_cancelled(turn_id)
                    self.emit(
                        {
                            "type": "audio_chunk",
                            "turnID": turn_id,
                            "chunkBase64": base64.b64encode(value).decode("utf-8"),
                            "sampleRate": TTS_PCM_SAMPLE_RATE,
                            "channels": TTS_PCM_CHANNELS,
                            "audioEncoding": TTS_PCM_ENCODING,
                        }
                    )
                elif item_type == "error":
                    raise value
                else:
                    break
            self.raise_if_turn_cancelled(turn_id)
        except asyncio.CancelledError:
            if cancel_event is not None:
                cancel_event.set()
            raise

    def iter_tts_pcm_chunks(
        self,
        *,
        text: str,
        voice_preset: str,
        cancel_event: threading.Event | None = None,
    ):
        model = self.tts_model
        if model is None:
            raise RuntimeError("TTS model is not loaded.")
        if self.can_stream_voxtral_tts_model(model):
            yield from self.stream_voxtral_tts_pcm_chunks(
                model,
                text=text,
                voice_preset=voice_preset,
                cancel_event=cancel_event,
            )
            return

        for result in model.generate(text=text, voice=voice_preset):
            if cancel_event is not None and cancel_event.is_set():
                break
            yield self.audio_array_to_pcm_bytes(result.audio)

    @staticmethod
    def can_stream_voxtral_tts_model(model: Any) -> bool:
        return all(
            hasattr(model, attribute)
            for attribute in (
                "_encode_text",
                "_build_input_embeddings",
                "language_model",
                "acoustic_transformer",
                "audio_tokenizer",
                "_codes_to_global_indices",
                "audio_codebook_embeddings",
                "config",
            )
        )

    def stream_voxtral_tts_pcm_chunks(
        self,
        model: Any,
        *,
        text: str,
        voice_preset: str,
        cancel_event: threading.Event | None,
        chunk_frames: int = TTS_STREAM_FRAME_BATCH,
        max_tokens: int = 4096,
    ):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        if model.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded. Ensure post_load_hook was called."
            )

        input_ids = model._encode_text(text, voice_preset)
        input_ids_mx = mx.array(input_ids)[None, :]
        input_embeddings = model._build_input_embeddings(input_ids_mx, voice_preset)
        lm_backbone = model.language_model.model.model
        cache = make_prompt_cache(model.language_model.model)
        lm_backbone(
            input_ids_mx,
            cache=cache,
            input_embeddings=input_embeddings,
        )

        audio_token_id = model.config.audio_token_id
        audio_tok_emb = model.language_model.embed_tokens(mx.array([[audio_token_id]]))
        hidden = lm_backbone(
            mx.array([[audio_token_id]]),
            cache=cache,
            input_embeddings=audio_tok_emb,
        )

        all_codes: list[Any] = []
        emitted_samples = 0
        frames_since_emit = 0

        for index in range(max_tokens):
            if cancel_event is not None and cancel_event.is_set():
                break

            hidden_frame = hidden[:, -1, :]
            codes = model.acoustic_transformer.decode_one_frame(hidden_frame)
            semantic_code = codes[0, 0].item()
            if semantic_code <= 1:
                break

            all_codes.append(codes[:, None, :])
            frames_since_emit += 1

            global_codes = model._codes_to_global_indices(codes)
            code_embeddings = model.audio_codebook_embeddings["embeddings"](global_codes)
            next_embedding = code_embeddings.sum(axis=1, keepdims=True)
            hidden = lm_backbone(
                mx.array([[audio_token_id]]),
                cache=cache,
                input_embeddings=next_embedding,
            )

            if frames_since_emit >= chunk_frames:
                chunk = self.decode_voxtral_tts_suffix(
                    model,
                    all_codes,
                    emitted_samples=emitted_samples,
                )
                if chunk is not None:
                    pcm_bytes, emitted_samples = chunk
                    yield pcm_bytes
                frames_since_emit = 0

            if index % 50 == 0:
                mx.clear_cache()

        if not all_codes:
            if cancel_event is not None and cancel_event.is_set():
                return
            raise RuntimeError("No audio frames generated")

        if cancel_event is None or not cancel_event.is_set():
            chunk = self.decode_voxtral_tts_suffix(
                model,
                all_codes,
                emitted_samples=emitted_samples,
            )
            if chunk is not None:
                pcm_bytes, _ = chunk
                yield pcm_bytes

        mx.clear_cache()

    def decode_voxtral_tts_suffix(
        self,
        model: Any,
        all_codes: list[Any],
        *,
        emitted_samples: int,
    ) -> tuple[bytes, int] | None:
        import mlx.core as mx

        audio_codes = mx.concatenate(all_codes, axis=1)
        waveform = model.audio_tokenizer.decode(audio_codes)
        waveform = np.asarray(waveform, dtype=np.float32)
        waveform = np.squeeze(waveform)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        total_samples = int(waveform.shape[0])
        if total_samples <= emitted_samples:
            return None
        suffix = waveform[emitted_samples:total_samples]
        if suffix.size == 0:
            return None
        return self.audio_array_to_pcm_bytes(suffix), total_samples

    async def cancel_turn(self, turn_id: str) -> None:
        state = self.ensure_turn_state(turn_id)
        self.cancelled_turns.add(turn_id)
        state.cancel_event.set()
        buffer = self.turns.get(turn_id)
        if buffer is not None:
            buffer.stopped = True
            if buffer.stt_session is not None:
                buffer.stt_session.cancel()
        self.clear_pending_tools_for_turn(turn_id)
        if state.task is not None and not state.task.done():
            state.task.cancel()
        self.emit_turn_finished(turn_id, status="cancelled", is_final=True)
        if state.task is None:
            self.turns.pop(turn_id, None)
            self.cleanup_turn_state(turn_id)

    async def cancel_all(self) -> None:
        active_turn_ids = set(self.turns.keys()) | set(self.turn_states.keys())
        for turn_id in list(active_turn_ids):
            await self.cancel_turn(turn_id)

    async def reset_conversation(self) -> None:
        await self.cancel_all()
        self.history.clear()
        self.cancelled_turns.clear()

    async def replace_history(self, messages: list[dict[str, Any]]) -> None:
        await self.cancel_all()
        self.history.clear()
        self.cancelled_turns.clear()
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            text = str(message.get("text", ""))
            attachments = self.normalize_attachments(message.get("attachments"))
            if role == "user":
                self.history.append(self.build_user_message(text, attachments))
            elif role == "assistant":
                self.history.append({"role": "assistant", "content": text})

    async def shutdown(self) -> None:
        await self.cancel_all()
        await self.cancel_installable_downloads()
        for model_id in MODEL_SPECS:
            await self.unload_model(model_id)
        self.history.clear()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass
        await self.mcp_client.close()
        self.emit({"type": "bootstrap_progress", "stage": "Shutting down runtime", "message": "Stopped local models and background automation services."})
        self.should_exit = True

    def normalize_generation_payload(self, parsed: Any, raw: str) -> dict[str, Any]:
        if isinstance(parsed, dict):
            payload = dict(parsed)
        elif isinstance(parsed, str):
            payload = {"assistant_message": parsed, "tool_call": None}
        else:
            payload = {"assistant_message": raw, "tool_call": None}

        payload["assistant_message"] = self.normalize_assistant_message(payload.get("assistant_message"))
        payload["tool_call"] = self.normalize_tool_call(payload.get("tool_call"), payload)
        return payload

    @staticmethod
    def normalize_assistant_message(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def normalize_tool_call(self, value: Any, payload: dict[str, Any]) -> dict[str, Any] | None:
        sibling_arguments = payload.get("tool_arguments")
        if sibling_arguments is None:
            sibling_arguments = payload.get("arguments")

        if isinstance(value, str):
            name = value.strip()
            if not name or name.lower() in {"null", "none", "false"}:
                return None
            if name not in self.available_tool_names:
                return None
            arguments = sibling_arguments if isinstance(sibling_arguments, dict) else {}
            return {"name": name, "arguments": self.sanitize_tool_arguments(name, arguments)}

        if not isinstance(value, dict):
            return None

        name = value.get("name") or value.get("tool_name")
        if not isinstance(name, str):
            return None
        name = name.strip()
        if name not in self.available_tool_names:
            return None

        arguments = value.get("arguments")
        if arguments is None:
            arguments = value.get("tool_arguments")
        if arguments is None and isinstance(sibling_arguments, dict):
            arguments = sibling_arguments
        if not isinstance(arguments, dict):
            arguments = {}

        return {"name": name, "arguments": self.sanitize_tool_arguments(name, arguments)}

    def classify_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> tuple[str, str, str]:
        if tool_name in READ_ONLY_TOOL_NAMES:
            return ("readOnly", "notRequired", "proposed")
        return ("mutating", "pending", "proposed")

    @property
    def available_tool_names(self) -> set[str]:
        if self.tool_definitions:
            return set(self.tool_definitions.keys())
        return {"get_scripting_tips", "execute_script"}

    async def refresh_tool_definitions(self, force: bool = False) -> None:
        if self.tool_definitions and not force:
            return
        tools = await self.mcp_client.list_tools()
        self.tool_definitions = {tool.name: tool for tool in tools}

    def tool_planner_guide(self) -> str:
        if not self.tool_definitions:
            return (
                "Available tools:\n"
                "- get_scripting_tips(search_term?, category?, list_categories?, refresh_database?, limit?) for discovery only.\n"
                "- execute_script(kb_script_id? | script_content? | script_path?, language?, arguments?, input_data?, timeout_seconds?) for actual execution."
            )

        lines = ["Live MCP tools:"]
        for name in sorted(self.tool_definitions):
            tool = self.tool_definitions[name]
            properties = tool.input_schema.get("properties", {}) if isinstance(tool.input_schema, dict) else {}
            property_names = ", ".join(sorted(properties.keys()))
            lines.append(f"- {name}({property_names})")
            if name == "get_scripting_tips":
                lines.append("  Use for discovery only. search_term is the normal way to query it.")
            elif name == "execute_script":
                lines.append("  Use for live actions or state lookups. One of kb_script_id, script_content, or script_path is required.")
                lines.append('  AppleScript example for opening Mail: {"script_content": "tell application \\"Mail\\" to activate", "language": "applescript"}')
        return "\n".join(lines)

    def choose_tool_call(self, user_text: str, tool_plan: dict[str, Any]) -> dict[str, Any] | None:
        proposed = tool_plan.get("tool_call")
        if isinstance(proposed, dict):
            name = proposed.get("name")
            arguments = proposed.get("arguments") or {}
            if isinstance(name, str):
                repaired = self.repair_tool_call(name, arguments, user_text)
                if repaired is not None:
                    return repaired
        return self.direct_tool_call_for_user_text(user_text)

    def direct_tool_call_for_user_text(self, user_text: str) -> dict[str, Any] | None:
        app_launch = self.app_activation_tool_call(user_text)
        if app_launch is not None:
            return app_launch
        return self.system_info_tool_call(user_text)

    def should_use_tool_planner(self, user_text: str, attachments: list[dict[str, Any]]) -> bool:
        if attachments:
            return True
        if self.direct_tool_call_for_user_text(user_text) is not None:
            return False

        text = user_text.strip().lower()
        if not text:
            return False
        if self.is_automation_discovery_request(user_text):
            return True

        imperative_prefixes = (
            "open ",
            "launch ",
            "start ",
            "close ",
            "quit ",
            "click ",
            "tap ",
            "press ",
            "type ",
            "paste ",
            "copy ",
            "move ",
            "drag ",
            "drop ",
            "scroll ",
            "select ",
            "choose ",
            "enable ",
            "disable ",
            "turn ",
            "set ",
            "rename ",
            "create ",
            "delete ",
            "remove ",
            "save ",
            "download ",
            "upload ",
            "install ",
            "uninstall ",
            "send ",
            "reply ",
            "compose ",
            "switch ",
            "focus ",
            "check ",
            "inspect ",
            "read ",
            "show ",
            "take ",
        )
        if any(text.startswith(prefix) for prefix in imperative_prefixes):
            return True

        tool_hint_terms = (
            " on my mac",
            " on this mac",
            " on my computer",
            " screenshot",
            " screen shot",
            " clipboard",
            " finder",
            " safari",
            " chrome",
            " mail",
            " messages",
            " notes",
            " calendar",
            " system settings",
            " window",
            " tab",
            " folder",
            " file",
            " desktop",
            " downloads",
            " documents",
            " application",
            " app ",
        )
        return any(term in text for term in tool_hint_terms)

    def repair_tool_call(self, name: str, arguments: dict[str, Any], user_text: str) -> dict[str, Any] | None:
        if name not in self.available_tool_names:
            return self.system_info_tool_call(user_text)

        arguments = self.sanitize_tool_arguments(name, arguments)
        if name == "get_scripting_tips":
            if not arguments:
                if self.is_automation_discovery_request(user_text):
                    arguments = {"search_term": user_text, "limit": 5}
                else:
                    return self.system_info_tool_call(user_text)
            if not any(key in arguments for key in ("search_term", "category", "list_categories", "refresh_database")):
                if self.is_automation_discovery_request(user_text):
                    arguments["search_term"] = user_text
                else:
                    return None
            return {"name": name, "arguments": arguments}

        if name == "execute_script":
            if not any(arguments.get(key) for key in ("kb_script_id", "script_content", "script_path")):
                return self.system_info_tool_call(user_text)
            return {"name": name, "arguments": arguments}

        return None

    def sanitize_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(arguments)
        if tool_name == "execute_script":
            if "script" in sanitized and "script_content" not in sanitized:
                sanitized["script_content"] = sanitized.pop("script")
            if "content" in sanitized and "script_content" not in sanitized:
                sanitized["script_content"] = sanitized.pop("content")
        elif tool_name == "get_scripting_tips":
            if "query" in sanitized and "search_term" not in sanitized:
                sanitized["search_term"] = sanitized.pop("query")
            if "search" in sanitized and "search_term" not in sanitized:
                sanitized["search_term"] = sanitized.pop("search")

        tool = self.tool_definitions.get(tool_name)
        if tool is None:
            return sanitized

        properties: dict[str, Any] = {}
        if isinstance(tool.input_schema, dict):
            properties = tool.input_schema.get("properties") or {}
        allowed_keys = set(properties.keys())
        if not allowed_keys:
            return sanitized
        sanitized = {key: value for key, value in sanitized.items() if key in allowed_keys}

        normalized: dict[str, Any] = {}
        for key, value in sanitized.items():
            normalized_value = self.normalize_argument_value(tool_name, key, value, properties.get(key) or {})
            if normalized_value is not None:
                normalized[key] = normalized_value
        return normalized

    def normalize_argument_value(self, tool_name: str, key: str, value: Any, schema: dict[str, Any]) -> Any:
        if value is None:
            return None

        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            if value in enum_values:
                pass
            else:
                remapped = self.remap_enum_value(tool_name, key, value, enum_values)
                if remapped is None:
                    return None
                value = remapped

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            schema_type = next((item for item in schema_type if isinstance(item, str)), None)

        if schema_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "1"}:
                    return True
                if lowered in {"false", "no", "0"}:
                    return False
            return None

        if schema_type == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return None
            return None

        if schema_type == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return None
            return None

        if schema_type == "array":
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                return [value]
            return None

        if schema_type == "object":
            return value if isinstance(value, dict) else None

        if schema_type == "string":
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            return None

        return value

    @staticmethod
    def remap_enum_value(tool_name: str, key: str, value: Any, enum_values: list[Any]) -> Any | None:
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")

            if tool_name == "execute_script" and key == "output_format_mode":
                aliases = {
                    "text": "human_readable",
                    "human": "human_readable",
                    "human_readable": "human_readable",
                    "readable": "human_readable",
                    "json": "structured_output_and_error",
                    "structured": "structured_output_and_error",
                    "structured_output": "structured_output_and_error",
                    "structured_output_and_errors": "structured_output_and_error",
                    "raw": "direct",
                }
                candidate = aliases.get(normalized)
                if candidate in enum_values:
                    return candidate

            for option in enum_values:
                if isinstance(option, str) and normalized == option.lower():
                    return option

        return None

    @staticmethod
    def is_automation_discovery_request(user_text: str) -> bool:
        text = user_text.lower()
        automation_terms = [
            "open ",
            "close ",
            "launch ",
            "click ",
            "type ",
            "press ",
            "finder",
            "safari",
            "chrome",
            "mail",
            "calendar",
            "notes",
            "script",
            "automation",
        ]
        return any(term in text for term in automation_terms)

    @staticmethod
    def system_info_tool_call(user_text: str) -> dict[str, Any] | None:
        text = user_text.lower()
        triggers = [
            "what computer am i on",
            "what mac am i on",
            "what machine am i on",
            "tell me about my computer",
            "tell me about this computer",
            "tell me about my mac",
            "tell me about this mac",
            "about my computer",
            "about this computer",
            "about my mac",
            "about this mac",
            "my computer specs",
            "my mac specs",
            "computer specs",
            "mac specs",
            "computer details",
            "mac details",
            "system specs",
            "system details",
            "what are my specs",
            "what are this computer specs",
            "what mac is this",
            "what computer is this",
            "what kind of computer is this",
            "computer name",
            "hostname",
            "host name",
            "system version",
            "macos version",
            "what version of macos",
            "what chip",
            "what processor",
            "how much memory",
            "how much ram",
            "serial number",
            "model identifier",
        ]
        if not any(trigger in text for trigger in triggers):
            return None
        return {
            "name": "execute_script",
            "arguments": {
                "script_content": SYSTEM_INFO_SCRIPT,
                "language": "applescript",
            },
        }

    @staticmethod
    def app_activation_tool_call(user_text: str) -> dict[str, Any] | None:
        match = re.match(
            r"^\s*(?:please\s+)?(?:open|launch|start)\s+(?:the\s+)?(?P<app>.+?)(?:\s+(?:app|application))?\s*[.!?]?\s*$",
            user_text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None

        raw_app_name = match.group("app").strip()
        if not raw_app_name:
            return None

        lowered = raw_app_name.lower()
        blocked_fragments = {"window", "folder", "file", "message", "email", "mailbox", "tab", "website"}
        if any(fragment in lowered for fragment in blocked_fragments):
            return None

        aliases = {
            "mail": "Mail",
            "email": "Mail",
            "emails": "Mail",
            "finder": "Finder",
            "safari": "Safari",
            "messages": "Messages",
            "notes": "Notes",
            "calendar": "Calendar",
            "music": "Music",
            "chrome": "Google Chrome",
            "google chrome": "Google Chrome",
        }
        app_name = aliases.get(lowered, " ".join(word.capitalize() for word in raw_app_name.split()))
        escaped_app_name = app_name.replace('"', '\\"')

        return {
            "name": "execute_script",
            "arguments": {
                "script_content": f'tell application "{escaped_app_name}" to activate',
                "language": "applescript",
            },
        }

    def emit_model_state(
        self,
        model_id: str,
        *,
        install_state: str | None = None,
        warm_state: str | None = None,
        message: str | None = None,
    ) -> None:
        state = self.model_states.setdefault(model_id, ModelState())
        install_state = install_state or ("installed" if state.installed else "missing")
        warm_state = warm_state or ("warm" if state.warm else "cold")
        self.emit(
            {
                "type": "model_state",
                "modelID": model_id,
                "installState": install_state,
                "warmState": warm_state,
                "message": message,
            }
        )

    def require_model(self, model_id: str, model: Any) -> None:
        if model is not None:
            return
        state = self.model_states.get(model_id, ModelState())
        if state.last_error:
            raise RuntimeError(f"{model_id} failed to load: {state.last_error}")
        raise RuntimeError(f"{model_id} is not ready.")

    @staticmethod
    def missing_agent_runtime_dependencies() -> list[str]:
        missing: list[str] = []
        for module_name, display_name in AGENT_RUNTIME_DEPENDENCIES.items():
            if importlib.util.find_spec(module_name) is None:
                missing.append(display_name)
        return missing

    def ensure_agent_runtime_dependencies(self) -> None:
        missing = self.missing_agent_runtime_dependencies()
        if missing:
            dependency_text = ", ".join(missing)
            raise RuntimeError(
                "The local runtime is missing bundled dependencies required by the Qwen image processor "
                f"({dependency_text}). Relaunch MacAssistant to refresh the runtime packages. "
                "If this keeps happening, delete ~/Library/Application Support/MacAssistant/runtime and relaunch."
            )
        try:
            request_module = importlib.import_module("mistral_common.protocol.instruct.request")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "The local runtime has an incomplete bundled mistral-common package required by the Qwen image processor. "
                "Relaunch MacAssistant to refresh the runtime packages. "
                "If this keeps happening, delete ~/Library/Application Support/MacAssistant/runtime and relaunch."
            ) from exc
        if not hasattr(request_module, "ReasoningEffort"):
            raise RuntimeError(
                "The local runtime has an incompatible bundled mistral-common package required by the Qwen image processor "
                "(missing ReasoningEffort support). Relaunch MacAssistant to refresh the runtime packages. "
                "If this keeps happening, delete ~/Library/Application Support/MacAssistant/runtime and relaunch."
            )

    def normalize_attachments(self, attachments: Any) -> list[dict[str, Any]]:
        if not isinstance(attachments, list):
            return []

        normalized: list[dict[str, Any]] = []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            attachment_type = attachment.get("type")
            file_path = attachment.get("filePath") or attachment.get("file_path")
            if attachment_type != "image" or not isinstance(file_path, str) or not file_path:
                continue
            normalized.append(
                {
                    "type": "image",
                    "file_path": file_path,
                    "display_name": attachment.get("displayName") or Path(file_path).name,
                }
            )
        return normalized

    def build_user_message(self, user_text: str, attachments: list[dict[str, Any]]) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for attachment in attachments:
            if attachment.get("type") != "image":
                continue
            file_path = attachment.get("file_path")
            if isinstance(file_path, str) and file_path:
                content.append({"type": "input_image", "image_url": file_path})

        text = user_text.strip()
        if text:
            content.append({"type": "text", "text": text})

        if not content:
            return {"role": "user", "content": text}
        if len(content) == 1 and content[0].get("type") == "text":
            return {"role": "user", "content": text}
        return {"role": "user", "content": content}

    def normalize_multimodal_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        normalized_messages: list[dict[str, Any]] = []
        images: list[str] = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role") or "user")
            content = message.get("content", "")

            if not isinstance(content, list):
                normalized_messages.append({"role": role, "content": content or ""})
                continue

            normalized_content: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type in {"input_image", "image_url"}:
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    if not isinstance(image_url, str) or not image_url:
                        continue
                    if not image_url.startswith("data:") and not Path(image_url).exists():
                        runtime_log(f"[agent] skipping missing image attachment: {image_url}")
                        continue
                    images.append(image_url)
                    normalized_content.append({"type": "input_image", "image_url": image_url})
                    continue

                if item_type in {"text", "input_text"}:
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        normalized_content.append({"type": "text", "text": str(text)})

            if len(normalized_content) == 1 and normalized_content[0].get("type") == "text":
                normalized_messages.append({"role": role, "content": normalized_content[0]["text"]})
            else:
                normalized_messages.append({"role": role, "content": normalized_content})

        return normalized_messages, images

    def create_stt_session(self, turn_id: str) -> BaseSTTSession:
        emit_partial = lambda text: self.emit(  # noqa: E731
            {"type": "transcript_delta", "turnID": turn_id, "text": text}
        )
        emit_error = lambda message: self.emit({"type": "error", "message": message})  # noqa: E731
        if self.uses_realtime_stt_session():
            return VoxtralRealtimeSTTSession(
                turn_id=turn_id,
                model=self.stt_model,
                emit_partial=emit_partial,
                emit_error=emit_error,
                transcription_delay_ms=REALTIME_TRANSCRIPTION_DELAY_MS,
            )
        return BufferedPreviewSTTSession(
            turn_id=turn_id,
            preview_transcribe=lambda pcm_bytes: self.transcribe_buffer(  # noqa: E731
                pcm_bytes,
                transcription_delay_ms=PREVIEW_TRANSCRIPTION_DELAY_MS,
            ),
            final_transcribe=lambda pcm_bytes: self.transcribe_buffer(  # noqa: E731
                pcm_bytes,
                transcription_delay_ms=FINAL_TRANSCRIPTION_DELAY_MS,
            ),
            emit_partial=emit_partial,
            emit_error=emit_error,
        )

    def uses_realtime_stt_session(self) -> bool:
        model = self.stt_model
        if model is None:
            return False
        config = getattr(model, "config", None)
        model_type = getattr(config, "model_type", None)
        if model_type != "voxtral_realtime":
            return False
        return all(
            hasattr(model, attribute)
            for attribute in ("encoder", "decoder", "_tokenizer", "_mel_filters", "_ensure_ada_scales")
        )

    def emit(self, payload: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    def emit_download_progress(
        self,
        installable_id: str,
        model_id: str,
        *,
        bytes_downloaded: int,
        bytes_total: int,
        speed_bytes_per_second: float | None = None,
        eta_seconds: float | None = None,
        message: str | None = None,
    ) -> None:
        fraction = None
        if bytes_total > 0:
            fraction = min(max(bytes_downloaded / bytes_total, 0.0), 1.0)

        self.emit(
            {
                "type": "download_progress",
                "installableID": installable_id,
                "modelID": model_id,
                "progress": fraction,
                "bytesDownloaded": bytes_downloaded,
                "bytesTotal": bytes_total,
                "speedBytesPerSecond": speed_bytes_per_second,
                "etaSeconds": eta_seconds,
                "message": message,
            }
        )

    def patch_tts_runtime(self, model: Any, model_path: Path) -> None:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return

        tekken_path = model_path / "tekken.json"
        if not self.tts_tokenizer_has_voice_metadata(tokenizer) and tekken_path.exists():
            from mlx_audio.tts.models.voxtral_tts.tekken import TekkenTokenizer

            model.tokenizer = TekkenTokenizer.from_file(tekken_path)
        sync_prompt_tokens = getattr(model, "_sync_prompt_token_ids_from_tokenizer", None)
        if callable(sync_prompt_tokens):
            sync_prompt_tokens()

    @staticmethod
    def tts_tokenizer_has_voice_metadata(tokenizer: Any) -> bool:
        try:
            instruct_tokenizer = getattr(tokenizer, "instruct_tokenizer", None)
            if instruct_tokenizer is not None:
                audio_encoder = getattr(instruct_tokenizer, "audio_encoder", None)
                if audio_encoder is not None:
                    audio_config = getattr(audio_encoder, "audio_config", None)
                    voice_cfg = getattr(audio_config, "voice_num_audio_tokens", None)
                    if isinstance(voice_cfg, dict) and voice_cfg:
                        return True
        except Exception:
            pass

        audio_cfg = getattr(tokenizer, "audio", None)
        if isinstance(audio_cfg, dict):
            voice_cfg = audio_cfg.get("voice_num_audio_tokens")
            if isinstance(voice_cfg, dict) and voice_cfg:
                return True

        return False

    @staticmethod
    def apply_model_compatibility_fixes(model_id: str, model_path: Path) -> None:
        return

    async def cancel_installable_downloads(self) -> None:
        for installable_id in list(self.install_tasks.keys()):
            await self.cancel_installable_download(installable_id)

    async def cancel_installable_download(self, installable_id: str) -> None:
        task = self.install_tasks.get(installable_id)
        if task is None or task.done():
            await self.stop_download_process(installable_id)
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop_download_process(installable_id)

    async def run_download_process(
        self,
        installable_id: str,
        model_id: str,
        *,
        completed_bytes: int,
        installable_total_bytes: int,
    ) -> tuple[str, int]:
        target_dir = MODELS_ROOT / model_id
        target_dir.mkdir(parents=True, exist_ok=True)

        environment = os.environ.copy()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            "--download-model",
            model_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
            start_new_session=True,
        )
        self.install_processes[installable_id] = process
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        progress_state = {"bytesDownloaded": 0, "bytesTotal": 0}
        stdout_task = asyncio.create_task(
            self.consume_download_stdout(
                stream=process.stdout,
                installable_id=installable_id,
                model_id=model_id,
                completed_bytes=completed_bytes,
                installable_total_bytes=installable_total_bytes,
                captured_lines=stdout_lines,
                progress_state=progress_state,
            )
        )
        stderr_task = asyncio.create_task(self.consume_process_stream(process.stderr, stderr_lines))

        try:
            await process.wait()
            await stdout_task
            await stderr_task
        except asyncio.CancelledError:
            await self.stop_download_process(installable_id)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise
        finally:
            if self.install_processes.get(installable_id) is process:
                self.install_processes.pop(installable_id, None)

        if process.returncode != 0:
            error_output = "\n".join(stderr_lines).strip()
            standard_output = "\n".join(stdout_lines).strip()
            message = error_output or standard_output or f"Download helper exited with code {process.returncode}."
            raise RuntimeError(message)

        observed_total = max(progress_state["bytesTotal"], progress_state["bytesDownloaded"])
        return (str(target_dir), observed_total)

    async def consume_download_stdout(
        self,
        *,
        stream: asyncio.StreamReader | None,
        installable_id: str,
        model_id: str,
        completed_bytes: int,
        installable_total_bytes: int,
        captured_lines: list[str],
        progress_state: dict[str, int],
    ) -> None:
        if stream is None:
            return

        while True:
            line = await stream.readline()
            if not line:
                break

            payload = line.decode("utf-8", errors="replace").strip()
            if not payload:
                continue

            captured_lines.append(payload)
            try:
                progress_payload = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if progress_payload.get("type") != "download_progress":
                continue

            model_bytes_downloaded = max(int(progress_payload.get("bytesDownloaded") or 0), 0)
            model_bytes_total = max(int(progress_payload.get("bytesTotal") or 0), 0)
            speed_bytes_per_second = progress_payload.get("speedBytesPerSecond")
            speed_bytes_per_second = float(speed_bytes_per_second) if speed_bytes_per_second is not None else None

            progress_state["bytesDownloaded"] = model_bytes_downloaded
            progress_state["bytesTotal"] = model_bytes_total

            effective_total = installable_total_bytes
            if effective_total <= 0 and model_bytes_total > 0:
                effective_total = completed_bytes + model_bytes_total

            total_downloaded = completed_bytes + model_bytes_downloaded
            if effective_total > 0:
                total_downloaded = min(total_downloaded, effective_total)

            eta_seconds = None
            if effective_total > 0 and speed_bytes_per_second and speed_bytes_per_second > 0:
                eta_seconds = max((effective_total - total_downloaded) / speed_bytes_per_second, 0.0)

            self.emit_download_progress(
                installable_id,
                model_id,
                bytes_downloaded=total_downloaded,
                bytes_total=effective_total,
                speed_bytes_per_second=speed_bytes_per_second,
                eta_seconds=eta_seconds,
            )

    async def consume_process_stream(self, stream: asyncio.StreamReader | None, captured_lines: list[str]) -> None:
        if stream is None:
            return

        while True:
            line = await stream.readline()
            if not line:
                break
            payload = line.decode("utf-8", errors="replace").strip()
            if payload:
                captured_lines.append(payload)

    async def stop_download_process(self, installable_id: str) -> None:
        process = self.install_processes.get(installable_id)
        if process is None:
            return
        if process.returncode is not None:
            self.install_processes.pop(installable_id, None)
            return

        self.install_processes.pop(installable_id, None)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            await process.wait()

    async def reset_model_install(self, model_id: str, *, last_error: str | None = None) -> None:
        await self.unload_model(model_id)
        model_dir = MODELS_ROOT / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        self.model_states[model_id] = ModelState(installed=False, warm=False, path=None, last_error=last_error)

    def plan_installable_download(self, installable_id: str) -> dict[str, int]:
        planned_sizes: dict[str, int] = {}
        for model_id in INSTALLABLES[installable_id]:
            try:
                planned_sizes[model_id] = self.estimate_model_download_size(model_id)
            except Exception:  # noqa: BLE001
                planned_sizes[model_id] = 0
        return planned_sizes

    @staticmethod
    def estimate_model_download_size(model_id: str) -> int:
        if model_id not in MODEL_SPECS:
            return 0

        from huggingface_hub import snapshot_download

        spec = MODEL_SPECS[model_id]
        target_dir = MODELS_ROOT / model_id
        dry_run_files = snapshot_download(
            repo_id=spec["repo"],
            local_dir=str(target_dir),
            tqdm_class=SilentTqdm,
            dry_run=True,
        )
        return sum(
            int(file_info.file_size)
            for file_info in dry_run_files
            if getattr(file_info, "will_download", False) and getattr(file_info, "file_size", None) is not None
        )

    def _ensure_dirs(self) -> None:
        for path in [APP_SUPPORT, MODELS_ROOT, LOGS_ROOT, CACHE_ROOT, RUNTIME_ROOT, MCP_ROOT, BUN_CACHE_ROOT]:
            path.mkdir(parents=True, exist_ok=True)

    def _purge_legacy_models(self) -> None:
        manifest_changed = False
        manifest_data: dict[str, Any] = {}

        if MANIFEST_PATH.exists():
            try:
                manifest_data = json.loads(MANIFEST_PATH.read_text())
            except json.JSONDecodeError:
                manifest_data = {}

            models = manifest_data.get("models")
            if isinstance(models, dict):
                for legacy_model_id in LEGACY_MODEL_IDS:
                    if models.pop(legacy_model_id, None) is not None:
                        manifest_changed = True
            elif "models" in manifest_data:
                manifest_data["models"] = {}
                manifest_changed = True

        for legacy_model_id in LEGACY_MODEL_IDS:
            legacy_dir = MODELS_ROOT / legacy_model_id
            if legacy_dir.exists():
                shutil.rmtree(legacy_dir, ignore_errors=True)

        if manifest_changed:
            manifest_data["models"] = {
                model_id: model_state
                for model_id, model_state in (manifest_data.get("models") or {}).items()
                if model_id in MODEL_SPECS
            }
            MANIFEST_PATH.write_text(json.dumps(manifest_data, indent=2))

    def _load_manifest(self) -> dict[str, ModelState]:
        if not MANIFEST_PATH.exists():
            return {model_id: ModelState() for model_id in MODEL_SPECS}
        try:
            data = json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            return {model_id: ModelState() for model_id in MODEL_SPECS}
        manifest = {}
        for model_id, spec in MODEL_SPECS.items():
            item = data.get("models", {}).get(model_id, {})
            model_path = item.get("path")
            last_error = item.get("last_error")
            installed_repo = item.get("repo")
            compatible_repo = installed_repo == spec["repo"]

            if model_path and Path(model_path).exists() and not compatible_repo:
                shutil.rmtree(model_path, ignore_errors=True)
                model_path = None
                last_error = f"Installed {model_id} no longer matches {spec['repo']}. Reinstall required."

            installed = bool(item.get("installed")) and bool(model_path) and compatible_repo and Path(model_path).exists()
            manifest[model_id] = ModelState(
                installed=installed,
                warm=False,
                path=model_path if installed else None,
                last_error=last_error,
            )
        return manifest

    def _save_manifest(self) -> None:
        payload = {
            "models": {
                model_id: {
                    "installed": state.installed,
                    "warm": state.warm,
                    "path": state.path,
                    "repo": MODEL_SPECS[model_id]["repo"],
                    "last_error": state.last_error,
                }
                for model_id, state in self.model_states.items()
            }
        }
        MANIFEST_PATH.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int, channels: int) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm_bytes)

    @staticmethod
    def pcm_bytes_to_audio_array(pcm_bytes: bytes) -> np.ndarray:
        if not pcm_bytes:
            return np.array([], dtype=np.float32)
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    @staticmethod
    def audio_array_to_pcm_bytes(audio: Any) -> bytes:
        array = np.asarray(audio, dtype=np.float32)
        array = np.squeeze(array)
        array = np.clip(array, -1.0, 1.0)
        pcm = (array * 32767).astype(np.int16)
        return pcm.tobytes()

    @staticmethod
    def audio_result_to_wav(audio: Any, sample_rate: int) -> bytes:
        array = np.asarray(audio)
        array = np.clip(array, -1.0, 1.0)
        pcm = (array * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        return buffer.getvalue()


def download_model_snapshot(model_id: str) -> None:
    if model_id not in MODEL_SPECS:
        raise RuntimeError(f"Unknown model ID: {model_id}")

    from huggingface_hub import snapshot_download

    spec = MODEL_SPECS[model_id]
    target_dir = MODELS_ROOT / model_id
    target_dir.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=spec["repo"],
        local_dir=str(target_dir),
        tqdm_class=make_download_progress_tqdm(model_id),
    )
    RuntimeHost.apply_model_compatibility_fixes(model_id, Path(local_dir))


def run_download_progress_adapter_self_test() -> None:
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        progress_class = make_download_progress_tqdm("agent_model")
        progress_bar = progress_class(
            total=100,
            name="huggingface_hub.snapshot_download",
            desc="self-test",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
        progress_bar.update(25)
        progress_bar.close()

    payloads = []
    for line in captured_stdout.getvalue().splitlines():
        line = line.strip()
        if not line:
            continue
        payloads.append(json.loads(line))

    progressed_payloads = [
        payload for payload in payloads if payload.get("type") == "download_progress" and int(payload.get("bytesDownloaded") or 0) > 0
    ]
    if not progressed_payloads:
        raise RuntimeError(f"Download progress self-test failed: {payloads!r}")

    print("download progress adapter self-test passed", flush=True)


async def main() -> None:
    host = RuntimeHost()
    await host.run()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--download-model":
        try:
            download_model_snapshot(sys.argv[2])
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    if len(sys.argv) == 2 and sys.argv[1] == "--self-test-progress-adapter":
        try:
            run_download_progress_adapter_self_test()
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
