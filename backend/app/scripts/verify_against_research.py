"""Verify the app's pipeline reproduces the saved final_v3 predictions.

    python -m app.scripts.verify_against_research
    python -m app.scripts.verify_against_research --sample-size 10 --mp4-check

READ-ONLY against both the research archive and the HDSD dataset: this script
opens files for reading and never writes, renames or deletes anything inside
either. All paths come from backend/.env (PREPARED_DIR, FINAL_V3_DIR,
HDSD_DATASET_DIR).

PRIVACY: the manifest and predictions.csv contain patient transcripts and
reference text. Nothing from those columns is ever printed, logged or written.
Output is restricted to identifiers, counts and scores; manifest access goes
through :func:`safe`, which refuses any field not on ``SAFE_FIELDS``.

Matching chain -- each link verified, none assumed:

    predictions.csv.utterance_id
      -> manifest.jsonl record, same utterance_id, split == SPLIT
      -> record.m2_path          (the M2 close-talk channel the research used)
      -> file under HDSD_DATASET_DIR
      -> sha256(file) == record.m2_sha256 == predictions.csv.audio_sha256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.research_assets import ResearchAssetError, verify_checkpoints
from app.models.enums import ModelKey
from app.services.metrics import score
from app.services.normalization import normalize_text
from app.services.transcription import TranscriptionService

#: The ONLY manifest fields this script may surface. Everything else -- in
#: particular draft_hindi, reference_hindi, transcript_roman and prompt -- is
#: transcript or reference text and must never leave the process.
SAFE_FIELDS = frozenset(
    {
        "utterance_id",
        "speaker",
        "session",
        "split",
        "speaker_partition",
        "prompt_partition",
        "duration_seconds",
        "m1_path",
        "m2_path",
        "m1_sha256",
        "m2_sha256",
        "review_status",
    }
)

#: app model key -> research run directory under FINAL_V3_DIR
RUNS = {
    ModelKey.PRETRAINED.value: "baseline_dev",
    ModelKey.FULL_FT.value: "full_lr5e-06_seed123",
    ModelKey.LORA.value: "lora_lr5e-05_seed2026",
}

DEFAULT_SPLIT = "test"


def safe(record: dict, field: str):
    """Read one allow-listed manifest field.

    Raises on anything text-bearing, so a future edit cannot accidentally
    print a transcript.
    """
    if field not in SAFE_FIELDS:
        raise KeyError(
            f"{field!r} is not on the safe allow-list; it may contain "
            "transcript or reference text and must not be displayed."
        )
    return record[field]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(run: str, split: str) -> dict[str, dict[str, str]]:
    path = settings.final_v3_dir / run / split / "predictions.csv"
    if not path.is_file():
        raise ResearchAssetError(f"Saved predictions not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["utterance_id"]: row for row in csv.DictReader(handle)}


def load_manifest(split: str) -> dict[str, dict]:
    path = settings.prepared_dir / "manifest.jsonl"
    if not path.is_file():
        raise ResearchAssetError(f"Prepared manifest not found: {path}")
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("split") == split:
                records[record["utterance_id"]] = record
    return records


def research_gen_config_hash(split: str) -> str | None:
    run = RUNS[ModelKey.FULL_FT.value]
    path = settings.final_v3_dir / run / split / "metadata.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("resolved_generation_config_sha256")


def _fmt(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:6.2f}"


def resolve_chain(manifest: dict, predictions: dict):
    """Verify every ID -> file link. Returns (candidates, failure counts)."""
    candidates: list[tuple[str, Path]] = []
    counts = {"audio_sha256_disagreed": 0, "file_absent": 0, "content_mismatch": 0}

    for utt_id in sorted(manifest):
        record = manifest[utt_id]
        expected = safe(record, "m2_sha256")

        pred_hashes = {
            predictions[key][utt_id]["audio_sha256"]
            for key in RUNS
            if utt_id in predictions[key]
        }
        if pred_hashes != {expected}:
            counts["audio_sha256_disagreed"] += 1
            continue

        wav = settings.hdsd_dataset_dir / safe(record, "m2_path")
        if not wav.is_file():
            counts["file_absent"] += 1
            continue
        if sha256_file(wav) != expected:
            counts["content_mismatch"] += 1
            continue

        candidates.append((utt_id, wav))
    return candidates, counts


def run_mp4_check(
    service: TranscriptionService, utt_id: str, wav: Path, predictions: dict
) -> None:
    """Encode one lossy AAC MP4 in the project folder and compare scores."""
    project_tmp = settings.resolved_upload_dir() / "_mp4_check"
    project_tmp.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="hdsd-mp4-"))
    mp4 = project_tmp / f"{utt_id}_aac.mp4"

    print()
    print("-" * 78)
    print(f"MP4 path check (AAC 128k) on {utt_id}")
    try:
        encode = subprocess.run(
            [
                settings.ffmpeg_binary, "-nostdin", "-y",
                "-i", str(wav),
                "-c:a", "aac", "-b:a", "128k", "-ac", "1",
                "-ar", str(settings.target_sample_rate),
                str(mp4),
            ],
            capture_output=True, text=True, timeout=300,
        )
        if encode.returncode != 0:
            print("  ERROR: ffmpeg encode failed")
            return

        print(f"  mp4: {mp4.stat().st_size} B (lossy)   wav: {wav.stat().st_size} B")
        print()
        print(
            f"  {'model':<11} {'WERwav':>7} {'WERmp4':>7} {'dWER':>7} "
            f"{'CERwav':>7} {'CERmp4':>7} {'dCER':>7} {'same text':>10}"
        )

        from_wav_path, _ = service.prepare_audio(wav, work / "fromwav")
        from_mp4_path, _ = service.prepare_audio(mp4, work / "frommp4")

        for key in RUNS:
            reference = normalize_text(predictions[key][utt_id]["reference"])
            from_wav = service.transcribe_wav(key, from_wav_path)
            from_mp4 = service.transcribe_wav(key, from_mp4_path)
            wav_score = score(reference, from_wav.text_normalized)
            mp4_score = score(reference, from_mp4.text_normalized)
            identical = from_wav.text_normalized == from_mp4.text_normalized
            print(
                f"  {key:<11} {_fmt(wav_score.wer)} {_fmt(mp4_score.wer)} "
                f"{(mp4_score.wer or 0.0) - (wav_score.wer or 0.0):+7.2f} "
                f"{_fmt(wav_score.cer)} {_fmt(mp4_score.cer)} "
                f"{(mp4_score.cer or 0.0) - (wav_score.cer or 0.0):+7.2f} "
                f"{str(identical):>10}"
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(project_tmp, ignore_errors=True)
        print()
        print(f"  temp MP4 removed: {not mp4.exists()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.scripts.verify_against_research",
        description="Compare the app pipeline against saved final_v3 predictions.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument(
        "--mp4-check",
        action="store_true",
        help="Also run one recording through a lossy AAC MP4.",
    )
    args = parser.parse_args(argv)
    configure_logging()

    try:
        verify_checkpoints()
    except ResearchAssetError as exc:
        print(f"error: {exc}")
        return 2

    manifest = load_manifest(args.split)
    predictions = {key: load_predictions(run, args.split) for key, run in RUNS.items()}

    print("=" * 78)
    print("Verification: app pipeline vs saved final_v3 predictions")
    print("=" * 78)
    print(f"dataset   : {settings.hdsd_dataset_dir}  (read-only)")
    print(f"prepared  : {settings.prepared_dir}  (read-only)")
    print(f"final_v3  : {settings.final_v3_dir}  (read-only)")
    print(f"split     : {args.split!r} -> {len(manifest)} manifest records")
    for key, run in RUNS.items():
        print(f"  {key:<11} {run:<22} {len(predictions[key])} saved rows")

    candidates, counts = resolve_chain(manifest, predictions)
    print()
    print(f"ID->file chain verified : {len(candidates)}/{len(manifest)}")
    for name, value in counts.items():
        print(f"  {name:<24}: {value}")

    if len(candidates) < args.sample_size:
        print("error: not enough verified recordings")
        return 1

    chosen = candidates[: args.sample_size]
    print()
    print(f"Selected (first {args.sample_size} by sorted utterance_id):")
    for utt_id, wav in chosen:
        record = manifest[utt_id]
        speaker = safe(record, "speaker")
        duration = safe(record, "duration_seconds")
        print(f"  {utt_id:<14} speaker={speaker:<5} {duration:>6.2f}s  {wav.name}")

    service = TranscriptionService()
    service.reset_peak_vram()
    shared = service.preload_all()
    after_load = service.peak_vram()

    print()
    print(f"app generation-config SHA-256 : {shared}")
    print(f"research resolved hash        : {research_gen_config_hash(args.split)}")
    print("  (different field sets; the invariant is that each is identical")
    print("   across its own three models)")
    print(
        f"VRAM after loading all 3      : "
        f"{after_load['allocated_mib']:.0f} MiB allocated, "
        f"{after_load['reserved_mib']:.0f} MiB reserved"
    )
    print(
        f"SHARE_BASE_MODEL={settings.share_base_model}  "
        f"MAX_RESIDENT_MODELS={settings.max_resident_models}"
    )

    work = Path(tempfile.mkdtemp(prefix="hdsd-verify-"))
    results: dict[str, list[dict]] = {key: [] for key in RUNS}
    compare_peak = after_load
    try:
        for utt_id, wav in chosen:
            # ffmpeg writes into `work`, never the dataset folder.
            wav16k, info = service.prepare_audio(wav, work)
            for key in RUNS:
                row = predictions[key][utt_id]
                reference = normalize_text(row["reference"])
                saved_pred = normalize_text(row["prediction"])

                result = service.transcribe_wav(key, wav16k)
                app_score = score(reference, result.text_normalized)
                saved_score = score(reference, saved_pred)

                results[key].append(
                    {
                        "utt": utt_id,
                        "exact": result.text_normalized == saved_pred,
                        "app_wer": app_score.wer,
                        "app_cer": app_score.cer,
                        "saved_wer": saved_score.wer,
                        "saved_cer": saved_score.cer,
                        "ref_words": app_score.reference_words,
                        "device": result.device,
                        "ms": result.inference_ms,
                        "rtf": result.rtf,
                        "dur": info.duration_sec,
                    }
                )
        compare_peak = service.peak_vram()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    total_exact = 0
    for key in RUNS:
        rows = results[key]
        exact = sum(1 for row in rows if row["exact"])
        total_exact += exact
        print()
        print("-" * 78)
        print(f"model {key}  ({RUNS[key]})")
        print(f"  EXACT normalized matches vs saved prediction: {exact}/{len(rows)}")
        print()
        print(
            f"  {'utterance':<14} {'refW':>5} {'appWER':>7} {'savWER':>7} "
            f"{'appCER':>7} {'savCER':>7} {'exact':>6} {'ms':>6} {'rtf':>5}"
        )
        for row in rows:
            print(
                f"  {row['utt']:<14} {row['ref_words']:>5} "
                f"{_fmt(row['app_wer'])} {_fmt(row['saved_wer'])} "
                f"{_fmt(row['app_cer'])} {_fmt(row['saved_cer'])} "
                f"{str(row['exact']):>6} {row['ms']:>6} {row['rtf']:>5.2f}"
            )

        def mean(name: str, rows: list[dict] = rows) -> float | None:
            values = [r[name] for r in rows if r[name] is not None]
            return sum(values) / len(values) if values else None

        print(
            f"  {'MEAN':<14} {'':>5} {_fmt(mean('app_wer'))} "
            f"{_fmt(mean('saved_wer'))} {_fmt(mean('app_cer'))} "
            f"{_fmt(mean('saved_cer'))}"
        )
        print(f"  device: {rows[0]['device']}")

    print()
    print("=" * 78)
    total = sum(len(results[key]) for key in RUNS)
    print(f"TOTAL exact matches : {total_exact}/{total}")
    print(
        f"PEAK VRAM (compare) : {compare_peak['allocated_mib']:.0f} MiB allocated, "
        f"{compare_peak['reserved_mib']:.0f} MiB reserved "
        f"of {compare_peak['total_mib']:.0f} MiB"
    )
    print("=" * 78)

    if args.mp4_check:
        run_mp4_check(service, chosen[0][0], chosen[0][1], predictions)

    return 0 if total_exact == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
