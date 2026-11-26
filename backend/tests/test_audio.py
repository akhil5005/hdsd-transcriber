"""Audio pipeline tests (PROMPT.md sections 4.1 and 4.5).

Real media is synthesised with ffmpeg rather than mocked: the thing most
likely to break is the ffmpeg invocation itself, and a mock would never catch
a wrong flag. Tests that need the binary skip cleanly when it is absent.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import pytest

from app.core.config import Settings
from app.services.audio import (
    AudioError,
    FFmpegNotFound,
    NoAudioStream,
    inspect_wav,
    plan_windows,
    probe,
    split_windows,
    to_wav16k_mono,
)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


@pytest.fixture(scope="module")
def ffmpeg_settings() -> Settings:
    """Settings pointing at a real ffmpeg, or skip the test."""
    config = Settings()  # reads .env, where FFMPEG_BINARY is pinned
    try:
        probe_ok = subprocess.run(
            [config.ffmpeg_binary, "-version"], capture_output=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("ffmpeg not available")
    if probe_ok.returncode != 0:
        pytest.skip("ffmpeg not runnable")
    return config


def _make_media(
    config: Settings, destination: Path, *, seconds: float, with_audio=True, video=False
) -> Path:
    """Synthesise a test file with ffmpeg."""
    command = [config.ffmpeg_binary, "-nostdin", "-y"]
    if with_audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if video:
        command += [
            "-f", "lavfi",
            "-i", f"testsrc=size=160x120:rate=10:duration={seconds}",
        ]
    if not with_audio and not video:
        raise ValueError("need at least one stream")
    command += ["-t", str(seconds), str(destination)]

    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-800:]
    return destination


# --- conversion -------------------------------------------------------------
def test_wav_conversion_produces_16k_mono_pcm16(ffmpeg_settings, tmp_path) -> None:
    source = _make_media(ffmpeg_settings, tmp_path / "tone.mp3", seconds=1.0)
    out = tmp_path / "tone.16k.wav"

    info = to_wav16k_mono(source, out, ffmpeg_settings)

    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.duration_sec == pytest.approx(1.0, abs=0.08)
    with wave.open(str(out), "rb") as handle:
        assert handle.getsampwidth() == 2  # signed 16-bit
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16000


def test_mp4_with_video_track_yields_audio_only(ffmpeg_settings, tmp_path) -> None:
    """The real input shape: an MP4 of a speaker."""
    source = _make_media(
        ffmpeg_settings, tmp_path / "clip.mp4", seconds=1.0, video=True
    )
    info = to_wav16k_mono(source, tmp_path / "clip.wav", ffmpeg_settings)

    assert info.channels == 1
    assert info.sample_rate == 16000
    assert info.frames > 0


def test_stereo_input_is_downmixed_to_mono(ffmpeg_settings, tmp_path) -> None:
    source = tmp_path / "stereo.wav"
    result = subprocess.run(
        [
            ffmpeg_settings.ffmpeg_binary, "-nostdin", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
            "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
            "-map", "[a]", "-ar", "44100", str(source),
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert inspect_wav(source).channels == 2

    info = to_wav16k_mono(source, tmp_path / "mono.wav", ffmpeg_settings)
    assert info.channels == 1
    assert info.sample_rate == 16000


def test_video_without_audio_is_rejected_by_name(ffmpeg_settings, tmp_path) -> None:
    source = _make_media(
        ffmpeg_settings,
        tmp_path / "silent.mp4",
        seconds=1.0,
        with_audio=False,
        video=True,
    )

    with pytest.raises(NoAudioStream, match="no audio stream"):
        to_wav16k_mono(source, tmp_path / "out.wav", ffmpeg_settings)


def test_non_media_file_gives_a_clear_error(ffmpeg_settings, tmp_path) -> None:
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not audio", encoding="utf-8")

    with pytest.raises(AudioError):
        to_wav16k_mono(junk, tmp_path / "out.wav", ffmpeg_settings)


def test_probe_reports_streams(ffmpeg_settings, tmp_path) -> None:
    source = _make_media(ffmpeg_settings, tmp_path / "tone.wav", seconds=0.5)
    data = probe(source, ffmpeg_settings)

    assert any(s["codec_type"] == "audio" for s in data["streams"])


def test_missing_ffmpeg_binary_names_the_setting(tmp_path) -> None:
    config = _settings(ffprobe_binary="definitely-not-a-real-binary-xyz")
    media = tmp_path / "x.wav"
    media.write_bytes(b"")

    with pytest.raises(FFmpegNotFound) as excinfo:
        probe(media, config)
    message = str(excinfo.value)
    assert "FFPROBE_BINARY" in message
    assert "winget install Gyan.FFmpeg" in message


# --- sample loading ---------------------------------------------------------
def test_load_samples_returns_float32_in_range(ffmpeg_settings, tmp_path) -> None:
    from app.services.audio import load_samples

    source = _make_media(ffmpeg_settings, tmp_path / "tone.wav", seconds=1.0)
    wav = tmp_path / "tone.16k.wav"
    to_wav16k_mono(source, wav, ffmpeg_settings)

    samples, rate = load_samples(wav)
    assert rate == 16000
    assert samples.dtype.name == "float32"
    assert samples.ndim == 1
    assert float(samples.max()) <= 1.0 and float(samples.min()) >= -1.0


# --- windowing (section 4.5) ------------------------------------------------
def test_short_audio_is_a_single_window() -> None:
    assert plan_windows(12.0, _settings()) == [(0.0, 12.0)]


def test_audio_exactly_at_the_chunk_length_is_one_window() -> None:
    assert plan_windows(30.0, _settings()) == [(0.0, 30.0)]


def test_long_audio_splits_into_abutting_30s_windows() -> None:
    assert plan_windows(70.0, _settings()) == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 70.0),
    ]


def test_windows_cover_the_whole_duration_without_gaps() -> None:
    windows = plan_windows(95.0, _settings())

    assert windows[0][0] == 0.0
    assert windows[-1][1] == pytest.approx(95.0)
    # strict=False: windows[1:] is intentionally one shorter.
    for (_, end), (next_start, _) in zip(windows, windows[1:], strict=False):
        assert next_start <= end  # no gap


def test_final_window_is_clamped_not_padded() -> None:
    """RTF must use real audio length, so the tail is not inflated to 30s."""
    windows = plan_windows(65.0, _settings())
    assert windows[-1] == (60.0, 65.0)


def test_overlap_shortens_the_stride() -> None:
    windows = plan_windows(70.0, _settings(chunk_overlap_seconds=5.0))

    assert windows[0] == (0.0, 30.0)
    assert windows[1] == (25.0, 55.0)
    assert windows[-1][1] == pytest.approx(70.0)


def test_windowing_is_deterministic() -> None:
    config = _settings()
    assert plan_windows(137.4, config) == plan_windows(137.4, config)


@pytest.mark.parametrize("overlap", [30.0, 31.0, -1.0])
def test_invalid_overlap_is_rejected(overlap) -> None:
    with pytest.raises(AudioError, match="CHUNK_OVERLAP_SECONDS"):
        plan_windows(60.0, _settings(chunk_overlap_seconds=overlap))


def test_zero_chunk_seconds_is_rejected() -> None:
    with pytest.raises(AudioError, match="CHUNK_SECONDS"):
        plan_windows(60.0, _settings(chunk_seconds=0))


def test_split_windows_slices_samples_at_window_boundaries() -> None:
    import numpy as np

    rate = 16000
    samples = np.zeros(rate * 70, dtype=np.float32)  # 70 seconds
    windows = split_windows(samples, rate, _settings())

    assert [w.index for w in windows] == [0, 1, 2]
    assert len(windows[0].samples) == 30 * rate
    assert len(windows[1].samples) == 30 * rate
    assert len(windows[2].samples) == 10 * rate
    # Concatenated slices reconstruct the original with zero overlap.
    assert sum(len(w.samples) for w in windows) == len(samples)
