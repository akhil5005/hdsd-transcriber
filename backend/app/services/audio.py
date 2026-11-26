"""Audio extraction and windowing (PROMPT.md sections 4.1 and 4.5).

ffmpeg converts any accepted container to the one format Whisper's feature
extractor expects: **16 kHz, mono, signed 16-bit PCM WAV**. Nothing else in
the codebase is allowed to guess at sample rates.

Windowing is deterministic by construction: fixed stride, fixed window length,
no silence detection or VAD. Two runs over the same file must produce the same
windows, or the research protocol's reproducibility claim is void.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: ffprobe/ffmpeg are given a hard ceiling so a malformed upload cannot wedge
#: a worker forever.
PROBE_TIMEOUT_SECONDS = 60
CONVERT_TIMEOUT_SECONDS = 900


class AudioError(Exception):
    """The input cannot be turned into analysable audio."""


class FFmpegNotFound(AudioError):
    pass


class NoAudioStream(AudioError):
    pass


@dataclass(frozen=True, slots=True)
class AudioInfo:
    """Facts about the decoded WAV, used for RTF and chunk planning."""

    path: Path
    duration_sec: float
    sample_rate: int
    channels: int
    frames: int


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """One deterministic analysis window."""

    index: int
    start_sec: float
    end_sec: float
    samples: object  # numpy.ndarray; typed loosely to keep numpy out of imports

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _resolve_binary(configured: str, label: str) -> str:
    """Accept either a bare name on PATH or an absolute path."""
    candidate = Path(configured)
    if candidate.is_file():
        return str(candidate)

    found = shutil.which(configured)
    if found:
        return found

    raise FFmpegNotFound(
        f"{label} not found (configured as {configured!r}).\n"
        f"Install it and set {label.upper()}_BINARY in backend/.env to the "
        "executable path, or put it on PATH.\n"
        "Windows: winget install Gyan.FFmpeg"
    )


def _run(command: list[str], timeout: int, label: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"{label} timed out after {timeout}s") from exc


def probe(path: Path, config: Settings | None = None) -> dict:
    """Return ffprobe's JSON for ``path``.

    Raises :class:`NoAudioStream` when the container has no audio track -- a
    silent video is a user error worth naming, not a zero-length transcript.
    """
    config = config or settings
    ffprobe = _resolve_binary(config.ffprobe_binary, "ffprobe")

    result = _run(
        [
            ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        PROBE_TIMEOUT_SECONDS,
        "ffprobe",
    )
    if result.returncode != 0:
        raise AudioError(
            f"Could not read media file: {path.name}\n"
            f"ffprobe said: {result.stderr.strip() or 'unknown error'}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AudioError(f"ffprobe returned unreadable output for {path.name}") from exc

    audio_streams = [
        stream
        for stream in data.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise NoAudioStream(
            f"{path.name} has no audio stream. "
            "A video with no sound cannot be transcribed."
        )
    return data


def to_wav16k_mono(
    source: Path, destination: Path, config: Settings | None = None
) -> AudioInfo:
    """Decode ``source`` to 16 kHz mono 16-bit PCM WAV at ``destination``.

    Any accepted container (mp4, mov, mkv, webm, wav, mp3, m4a, flac) goes
    through the same path, so the model never sees a format the research
    protocol did not use.
    """
    config = config or settings
    probe(source, config)  # fails fast on a missing audio stream
    ffmpeg = _resolve_binary(config.ffmpeg_binary, "ffmpeg")

    destination.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i", str(source),
            "-vn",                                  # drop video
            "-map", "a:0",                          # first audio stream only
            "-ac", "1",                             # mono
            "-ar", str(config.target_sample_rate),   # 16 kHz
            "-acodec", "pcm_s16le",                  # signed 16-bit PCM
            "-f", "wav",
            str(destination),
        ],
        CONVERT_TIMEOUT_SECONDS,
        "ffmpeg",
    )
    if result.returncode != 0 or not destination.is_file():
        raise AudioError(
            f"Audio extraction failed for {source.name}\n"
            f"ffmpeg said: {(result.stderr or '').strip()[-500:] or 'unknown error'}"
        )

    info = inspect_wav(destination)
    if info.sample_rate != config.target_sample_rate or info.channels != 1:
        raise AudioError(
            f"Converted audio is {info.sample_rate} Hz / {info.channels}ch, "
            f"expected {config.target_sample_rate} Hz mono"
        )
    if info.frames == 0:
        raise NoAudioStream(f"{source.name} decoded to zero audio samples.")
    return info


def inspect_wav(path: Path) -> AudioInfo:
    """Read WAV geometry without loading the samples."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            channels = handle.getnchannels()
    except (OSError, wave.Error) as exc:
        raise AudioError(f"Not a readable WAV file: {path}") from exc

    return AudioInfo(
        path=path,
        duration_sec=(frames / rate) if rate else 0.0,
        sample_rate=rate,
        channels=channels,
        frames=frames,
    )


def load_samples(path: Path):
    """Load a 16 kHz mono WAV as float32 in [-1, 1].

    soundfile is used rather than librosa.load because the file is already at
    the target rate -- there is nothing to resample, and resampling here would
    silently diverge from the research pipeline.
    """
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:  # defensive: ffmpeg already downmixed
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), rate


def plan_windows(
    duration_sec: float, config: Settings | None = None
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` pairs covering ``duration_sec``.

    Training clips were short sentences, so anything longer than
    ``CHUNK_SECONDS`` is split. The stride is ``chunk - overlap``; with the
    default zero overlap the windows simply abut. The final window is clamped
    to the true duration rather than zero-padded, so RTF stays honest.
    """
    config = config or settings
    chunk = float(config.chunk_seconds)
    overlap = float(config.chunk_overlap_seconds)

    if chunk <= 0:
        raise AudioError("CHUNK_SECONDS must be positive")
    if not 0 <= overlap < chunk:
        raise AudioError(
            f"CHUNK_OVERLAP_SECONDS must be in [0, {chunk}); got {overlap}"
        )

    if duration_sec <= chunk:
        return [(0.0, max(duration_sec, 0.0))]

    stride = chunk - overlap
    count = math.ceil((duration_sec - overlap) / stride)
    windows: list[tuple[float, float]] = []
    for index in range(count):
        start = index * stride
        end = min(start + chunk, duration_sec)
        if end - start <= 0:
            break
        windows.append((start, end))
    return windows


def split_windows(samples, sample_rate: int, config: Settings | None = None):
    """Slice loaded samples into :class:`AudioWindow` objects."""
    duration = len(samples) / sample_rate if sample_rate else 0.0
    out: list[AudioWindow] = []
    for index, (start, end) in enumerate(plan_windows(duration, config)):
        begin = int(round(start * sample_rate))
        finish = min(int(round(end * sample_rate)), len(samples))
        out.append(
            AudioWindow(
                index=index,
                start_sec=start,
                end_sec=end,
                samples=samples[begin:finish],
            )
        )
    return out
