"""Upload validation and storage (PROMPT.md section 8).

This is patient audio, so the rules are strict:

* Type is decided by **content**, not by the filename. Three layers: an
  extension allow-list (cheap reject), a magic-byte sniff (catches a mislabeled
  file), and finally ffprobe, which is authoritative -- if ffmpeg cannot find a
  decodable audio stream, the upload is refused.
* Stored under ``UPLOAD_DIR`` with a **UUID filename**, outside any web root,
  so the original name can never influence a path.
* The size limit is enforced **while streaming**, so an oversized upload is
  abandoned rather than buffered in memory first.
* Filenames and ids are logged; transcript text never is.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from app.core.config import Settings, settings
from app.core.logging import get_logger
from app.services.audio import AudioError, probe

logger = get_logger(__name__)

#: Accepted containers (section 8).
ALLOWED_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".m4a", ".flac"}
)

#: Read in 1 MiB blocks: large enough to be fast, small enough to abort early.
CHUNK_BYTES = 1024 * 1024

#: Enough bytes for every signature below.
SNIFF_BYTES = 64


class MediaError(Exception):
    """The upload is not acceptable."""


class UnsupportedMediaType(MediaError):
    pass


class MediaTooLarge(MediaError):
    pass


@dataclass(frozen=True, slots=True)
class StoredMedia:
    """A validated upload on disk."""

    path: Path
    original_filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    duration_sec: float | None


class UploadLike(Protocol):
    """The slice of Starlette's UploadFile this module needs."""

    filename: str | None
    content_type: str | None
    file: BinaryIO


def _sniff(head: bytes) -> str | None:
    """Return a mime type from magic bytes, or None if unrecognised."""
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "audio/wav"
    if head.startswith(b"fLaC"):
        return "audio/flac"
    if head.startswith(b"OggS"):
        return "audio/ogg"
    if head.startswith(b"ID3") or (
        len(head) > 1 and head[0] == 0xFF and head[1] in (0xFB, 0xF3, 0xF2, 0xFA)
    ):
        return "audio/mpeg"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        # EBML: Matroska and WebM share the container.
        return "video/webm" if b"webm" in head[:48] else "video/x-matroska"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4B "):
            return "audio/mp4"
        if brand.startswith(b"qt"):
            return "video/quicktime"
        return "video/mp4"
    return None


def sniff_mime(path: Path) -> str | None:
    with path.open("rb") as handle:
        return _sniff(handle.read(SNIFF_BYTES))


def validate_extension(filename: str | None) -> str:
    """Cheap first gate. Returns the normalised extension."""
    if not filename:
        raise UnsupportedMediaType("The upload has no filename.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedMediaType(
            f"{suffix or 'that file type'} is not supported. "
            f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return suffix


def store_upload(
    upload: UploadLike,
    *,
    config: Settings | None = None,
    destination_dir: Path | None = None,
) -> StoredMedia:
    """Validate and persist one upload. Raises :class:`MediaError` on refusal.

    The partially written file is always removed when validation fails, so a
    rejected upload leaves nothing behind.
    """
    config = config or settings
    suffix = validate_extension(upload.filename)
    target_dir = destination_dir or config.resolved_upload_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    # UUID filename: the client's name never reaches the filesystem.
    stored_path = target_dir / f"{uuid.uuid4().hex}{suffix}"
    digest = hashlib.sha256()
    size = 0
    limit = config.max_upload_bytes

    try:
        source = upload.file
        try:
            source.seek(0)
        except (OSError, AttributeError):  # pragma: no cover - stream already at 0
            pass

        with stored_path.open("wb") as sink:
            while True:
                block = source.read(CHUNK_BYTES)
                if not block:
                    break
                size += len(block)
                if size > limit:
                    raise MediaTooLarge(
                        f"Upload exceeds the {config.max_upload_mb} MB limit."
                    )
                digest.update(block)
                sink.write(block)

        if size == 0:
            raise UnsupportedMediaType("The upload is empty.")

        sniffed = sniff_mime(stored_path)
        if sniffed is None:
            raise UnsupportedMediaType(
                "The file contents do not look like audio or video, whatever "
                "the extension says."
            )

        # Authoritative check: ffprobe must find a decodable audio stream.
        try:
            probed = probe(stored_path, config)
        except AudioError as exc:
            raise UnsupportedMediaType(str(exc)) from exc

        duration = None
        raw_duration = probed.get("format", {}).get("duration")
        if raw_duration is not None:
            try:
                duration = float(raw_duration)
            except (TypeError, ValueError):
                duration = None

        logger.info(
            "Stored upload %s (%d bytes, %s, %.2fs)",
            stored_path.name,
            size,
            sniffed,
            duration or 0.0,
        )
        return StoredMedia(
            path=stored_path,
            original_filename=Path(upload.filename or "upload").name,
            mime_type=sniffed,
            size_bytes=size,
            sha256=digest.hexdigest(),
            duration_sec=duration,
        )
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise


def delete_media(path: str | Path) -> bool:
    """Remove a stored file. Returns True if something was deleted.

    Refuses to touch anything outside ``UPLOAD_DIR`` -- a stored_path column is
    data, and data should never be able to point the deleter at the research
    archive.
    """
    candidate = Path(path)
    root = settings.resolved_upload_dir()
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        logger.error(
            "Refusing to delete %s: outside the upload directory %s", candidate, root
        )
        return False

    if not candidate.is_file():
        return False
    candidate.unlink()
    return True


def purge_expired_media(config: Settings | None = None) -> int:
    """Delete stored files older than ``MEDIA_RETENTION_DAYS``.

    Operates on the filesystem only; the database rows are cleaned up by the
    caller so the two cannot drift apart silently.
    """
    import time

    config = config or settings
    if config.media_retention_days <= 0:
        return 0

    cutoff = time.time() - config.media_retention_days * 86400
    removed = 0
    for path in config.resolved_upload_dir().iterdir():
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        logger.info("Purged %d expired media file(s)", removed)
    return removed


def copy_for_processing(source: Path, work_dir: Path) -> Path:
    """Copy a stored upload into a scratch directory."""
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / source.name
    shutil.copy2(source, target)
    return target
