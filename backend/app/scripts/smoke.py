"""CLI smoke test for the inference stack (PROMPT.md section 10.2).

    python -m app.scripts.smoke --model all --file sample.mp4

Prints the transcript from each requested model with its timings, and
**asserts that the resolved generation-config hash is identical across
models** -- the check that makes a three-way comparison meaningful.

Nothing here writes to the research archive, and the converted WAV goes to a
temporary directory that is removed on exit unless --keep-wav is passed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.research_assets import ResearchAssetError, verify_checkpoints
from app.models.enums import ModelKey
from app.services import generation
from app.services.audio import AudioError
from app.services.metrics import score
from app.services.normalization import normalize_text
from app.services.transcription import (
    InferenceDisabled,
    TranscriptionService,
    resolve_device,
)

ALL_KEYS = [ModelKey.PRETRAINED.value, ModelKey.FULL_FT.value, ModelKey.LORA.value]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.scripts.smoke",
        description="Transcribe a file with one or all three Whisper models.",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=[*ALL_KEYS, "all"],
        help="Model to run (default: all).",
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Media file to transcribe (mp4, wav, mp3, ...).",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional Devanagari reference transcript, to print WER/CER.",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep the converted 16 kHz WAV instead of deleting it.",
    )
    return parser.parse_args(argv)


def _print_header(media: Path, keys: list[str]) -> None:
    print("=" * 72)
    print("HDSD transcription smoke test")
    print("=" * 72)
    print(f"file         : {media}")
    print(f"models       : {', '.join(keys)}")
    print(f"device       : {resolve_device()} (DEVICE={settings.device})")
    print(f"base model   : {settings.base_model_id}@{settings.base_model_revision}")
    print(f"max_new_tokens={settings.max_new_tokens}  chunk={settings.chunk_seconds}s")
    print()


def _print_result(result, reference: str | None) -> None:
    print("-" * 72)
    print(f"model          : {result.model_key}")
    print(f"checkpoint     : {result.checkpoint}")
    print(f"device         : {result.device}")
    print(f"gen_config     : {result.gen_config_hash}")
    print(f"audio duration : {result.audio_duration_sec:.2f}s")
    print(f"inference      : {result.inference_ms} ms  (RTF {result.rtf:.2f})")
    print(f"windows        : {len(result.chunks)}")
    print()
    print("transcript (raw):")
    print(f"  {result.text_raw or '<empty>'}")
    print()
    print("transcript (normalized):")
    print(f"  {result.text_normalized or '<empty>'}")

    if len(result.chunks) > 1:
        print()
        print("per-window:")
        for chunk in result.chunks:
            window = f"[{chunk.start_sec:7.2f}-{chunk.end_sec:7.2f}]"
            print(f"  {window} {chunk.text or '<empty>'}")

    if reference:
        scored = score(reference, result.text_normalized)
        wer = "n/a" if scored.wer is None else f"{scored.wer:.2f}%"
        cer = "n/a" if scored.cer is None else f"{scored.cer:.2f}%"
        print()
        print(
            f"vs reference   : WER {wer}  CER {cer}  "
            f"S={scored.substitutions} D={scored.deletions} I={scored.insertions}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()

    media: Path = args.file.expanduser()
    if not media.is_file():
        print(f"error: file not found: {media}", file=sys.stderr)
        return 2

    keys = ALL_KEYS if args.model == "all" else [args.model]
    reference = normalize_text(args.reference) if args.reference else None

    try:
        verify_checkpoints()
    except ResearchAssetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_header(media, keys)

    service = TranscriptionService()
    work_dir = Path(tempfile.mkdtemp(prefix="hdsd-smoke-"))
    try:
        try:
            wav_path, info = service.prepare_audio(media, work_dir)
        except AudioError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        print(
            f"decoded to {info.sample_rate} Hz mono, "
            f"{info.duration_sec:.2f}s, {info.frames} frames"
        )
        print()

        results = {}
        fingerprints: dict[str, str] = {}
        for key in keys:
            try:
                result = service.transcribe_wav(key, wav_path)
            except InferenceDisabled as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            results[key] = result
            fingerprints[key] = result.gen_config_hash
            _print_result(result, reference)

        # The assertion section 10.2 asks for.
        print("=" * 72)
        if len(keys) > 1:
            try:
                shared = generation.assert_identical(fingerprints)
            except generation.GenerationConfigError as exc:
                print(f"FAIL: {exc}", file=sys.stderr)
                return 1
            print(f"PASS generation config identical across {len(keys)} models")
            print(f"     SHA-256 {shared}")
        else:
            print(f"single model; gen_config SHA-256 {fingerprints[keys[0]]}")

        # Agreement between models (no ground truth involved).
        if len(keys) > 1:
            print()
            print("pairwise agreement (agreement, not accuracy):")
            for left in keys:
                for right in keys:
                    if left >= right:
                        continue
                    value = score(
                        results[left].text_normalized,
                        results[right].text_normalized,
                    ).wer
                    shown = "n/a" if value is None else f"{value:.2f}%"
                    print(f"  {left:>10} vs {right:<10} {shown}")
        print("=" * 72)
        return 0
    finally:
        if args.keep_wav:
            print(f"converted WAV kept at {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
