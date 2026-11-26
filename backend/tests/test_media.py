"""Upload validation and storage (section 8).

Type detection is tested with real files built by ffmpeg, because the whole
point of section 8 is that a *renamed* file must still be rejected -- a mock
would happily agree with whatever the extension claimed.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from app.core.config import Settings
from app.services import media as media_service
from app.services.media import (
    ALLOWED_EXTENSIONS,
    MediaTooLarge,
    UnsupportedMediaType,
    sniff_mime,
    store_upload,
    validate_extension,
)


class FakeUpload:
    """The slice of Starlette's UploadFile that store_upload uses."""

    def __init__(self, filename: str, payload: bytes, content_type: str = "") -> None:
        self.filename = filename
        self.content_type = content_type
        self.file = io.BytesIO(payload)


@pytest.fixture
def config(tmp_path) -> Settings:
    return Settings(upload_dir=tmp_path / "uploads")


@pytest.fixture(scope="module")
def ffmpeg() -> str:
    binary = Settings().ffmpeg_binary
    try:
        result = subprocess.run([binary, "-version"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("ffmpeg not available")
    if result.returncode != 0:
        pytest.skip("ffmpeg not runnable")
    return binary


def _make(ffmpeg: str, path: Path, *, seconds: float = 1.0, video: bool = False) -> bytes:
    command = [
        ffmpeg, "-nostdin", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
    ]
    if video:
        command += [
            "-f", "lavfi",
            "-i", f"testsrc=size=160x120:rate=10:duration={seconds}",
        ]
    command += ["-t", str(seconds), str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-600:]
    return path.read_bytes()


# --- extension gate ---------------------------------------------------------
def test_accepted_extensions_match_the_spec() -> None:
    assert ALLOWED_EXTENSIONS == {
        ".mp4", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".m4a", ".flac"
    }


@pytest.mark.parametrize("name", ["a.wav", "A.WAV", "clip.mp4", "x.flac"])
def test_allowed_extensions_pass(name) -> None:
    assert validate_extension(name).startswith(".")


@pytest.mark.parametrize("name", ["notes.txt", "script.exe", "archive.zip", "noext"])
def test_disallowed_extensions_are_refused_by_name(name) -> None:
    with pytest.raises(UnsupportedMediaType):
        validate_extension(name)


def test_missing_filename_is_refused() -> None:
    with pytest.raises(UnsupportedMediaType):
        validate_extension(None)


# --- content sniffing -------------------------------------------------------
def test_sniffs_wav_from_content(ffmpeg, tmp_path) -> None:
    path = tmp_path / "tone.wav"
    _make(ffmpeg, path)
    assert sniff_mime(path) == "audio/wav"


def test_sniffs_mp4_from_content(ffmpeg, tmp_path) -> None:
    path = tmp_path / "clip.mp4"
    _make(ffmpeg, path, video=True)
    assert sniff_mime(path) == "video/mp4"


def test_sniffs_flac_and_mp3(ffmpeg, tmp_path) -> None:
    flac = tmp_path / "a.flac"
    mp3 = tmp_path / "a.mp3"
    _make(ffmpeg, flac)
    _make(ffmpeg, mp3)
    assert sniff_mime(flac) == "audio/flac"
    assert sniff_mime(mp3) == "audio/mpeg"


def test_text_file_renamed_to_wav_is_rejected(config) -> None:
    """Section 8: validate by content, not only by extension."""
    upload = FakeUpload("innocent.wav", b"this is plain text, not audio" * 10)

    with pytest.raises(UnsupportedMediaType, match="do not look like"):
        store_upload(upload, config=config)


def test_a_rejected_upload_leaves_nothing_on_disk(config) -> None:
    upload = FakeUpload("innocent.wav", b"not audio at all" * 10)

    with pytest.raises(UnsupportedMediaType):
        store_upload(upload, config=config)

    assert list(config.resolved_upload_dir().iterdir()) == []


def test_empty_upload_is_rejected(config) -> None:
    with pytest.raises(UnsupportedMediaType, match="empty"):
        store_upload(FakeUpload("x.wav", b""), config=config)


# --- storage ----------------------------------------------------------------
def test_stored_file_gets_a_uuid_name_and_keeps_the_extension(
    ffmpeg, tmp_path, config
) -> None:
    payload = _make(ffmpeg, tmp_path / "tone.wav")
    stored = store_upload(FakeUpload("Patient Recording.wav", payload), config=config)

    assert stored.path.suffix == ".wav"
    assert stored.path.stem != "Patient Recording"
    assert len(stored.path.stem) == 32  # uuid4().hex
    # The display name is kept, but never used as a path.
    assert stored.original_filename == "Patient Recording.wav"


def test_stored_file_is_inside_the_upload_directory(ffmpeg, tmp_path, config) -> None:
    payload = _make(ffmpeg, tmp_path / "tone.wav")
    stored = store_upload(FakeUpload("a.wav", payload), config=config)

    assert stored.path.parent == config.resolved_upload_dir()


def test_path_traversal_in_the_filename_cannot_escape(ffmpeg, tmp_path, config) -> None:
    payload = _make(ffmpeg, tmp_path / "tone.wav")
    stored = store_upload(FakeUpload("../../../evil.wav", payload), config=config)

    assert stored.path.parent == config.resolved_upload_dir()
    assert ".." not in str(stored.path)
    assert stored.original_filename == "evil.wav"


def test_metadata_is_recorded(ffmpeg, tmp_path, config) -> None:
    payload = _make(ffmpeg, tmp_path / "tone.wav", seconds=2.0)
    stored = store_upload(FakeUpload("a.wav", payload), config=config)

    assert stored.size_bytes == len(payload)
    assert len(stored.sha256) == 64
    assert stored.mime_type == "audio/wav"
    assert stored.duration_sec == pytest.approx(2.0, abs=0.1)


def test_sha256_matches_the_bytes(ffmpeg, tmp_path, config) -> None:
    import hashlib

    payload = _make(ffmpeg, tmp_path / "tone.wav")
    stored = store_upload(FakeUpload("a.wav", payload), config=config)
    assert stored.sha256 == hashlib.sha256(payload).hexdigest()


# --- size limit -------------------------------------------------------------
def test_oversized_upload_is_refused(ffmpeg, tmp_path) -> None:
    payload = _make(ffmpeg, tmp_path / "tone.wav", seconds=2.0)
    tiny_limit = Settings(upload_dir=tmp_path / "uploads2", max_upload_mb=0)

    with pytest.raises(MediaTooLarge, match="exceeds"):
        store_upload(FakeUpload("a.wav", payload), config=tiny_limit)


def test_oversized_upload_leaves_no_partial_file(ffmpeg, tmp_path) -> None:
    payload = _make(ffmpeg, tmp_path / "tone.wav", seconds=2.0)
    tiny_limit = Settings(upload_dir=tmp_path / "uploads3", max_upload_mb=0)

    with pytest.raises(MediaTooLarge):
        store_upload(FakeUpload("a.wav", payload), config=tiny_limit)

    assert list(tiny_limit.resolved_upload_dir().iterdir()) == []


# --- deletion guard ---------------------------------------------------------
def test_delete_media_removes_a_stored_file(ffmpeg, tmp_path, monkeypatch) -> None:
    from app.core import config as config_module

    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(config_module.settings, "upload_dir", upload_dir)

    payload = _make(ffmpeg, tmp_path / "tone.wav")
    stored = store_upload(FakeUpload("a.wav", payload), config=config_module.settings)

    assert media_service.delete_media(stored.path) is True
    assert not stored.path.exists()


def test_delete_media_refuses_paths_outside_the_upload_directory(
    tmp_path, monkeypatch
) -> None:
    """stored_path is data; it must never be able to point at the archive."""
    from app.core import config as config_module

    monkeypatch.setattr(config_module.settings, "upload_dir", tmp_path / "uploads")
    outsider = tmp_path / "precious.wav"
    outsider.write_bytes(b"do not delete me")

    assert media_service.delete_media(outsider) is False
    assert outsider.exists()


def test_delete_media_on_a_missing_file_is_not_an_error(
    tmp_path, monkeypatch
) -> None:
    from app.core import config as config_module

    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(config_module.settings, "upload_dir", upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    assert media_service.delete_media(upload_dir / "gone.wav") is False


# --- retention --------------------------------------------------------------
def test_purge_removes_only_expired_files(ffmpeg, tmp_path) -> None:
    import os
    import time

    config = Settings(upload_dir=tmp_path / "uploads", media_retention_days=30)
    target = config.resolved_upload_dir()

    fresh = target / "fresh.wav"
    stale = target / "stale.wav"
    fresh.write_bytes(b"new")
    stale.write_bytes(b"old")
    old = time.time() - 31 * 86400
    os.utime(stale, (old, old))

    assert media_service.purge_expired_media(config) == 1
    assert fresh.exists()
    assert not stale.exists()


def test_purge_is_disabled_when_retention_is_zero(tmp_path) -> None:
    config = Settings(upload_dir=tmp_path / "uploads", media_retention_days=0)
    target = config.resolved_upload_dir()
    (target / "anything.wav").write_bytes(b"x")

    assert media_service.purge_expired_media(config) == 0
    assert (target / "anything.wav").exists()
