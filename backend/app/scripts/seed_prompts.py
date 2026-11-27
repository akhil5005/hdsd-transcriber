"""Seed the 30 HDSD sentence prompts used as ground truth.

    python -m app.scripts.seed_prompts
    python -m app.scripts.seed_prompts --dry-run

Idempotent: re-running updates the text of existing rows and inserts any that
are missing, so correcting data/hdsd_prompts.csv and re-running is the normal
way to fix a sentence.

Source is data/hdsd_prompts.csv in this repo, NOT the research archive. The
Devanagari there was transcribed from the corpus's own romanised word labels;
the `verified` column records whether a human has checked it.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

from sqlalchemy import select

from app.core.logging import configure_logging
from app.db.session import SessionLocal, engine
from app.models.prompt import DatasetPrompt

PROMPTS_CSV = Path(__file__).resolve().parents[2] / "data" / "hdsd_prompts.csv"

REQUIRED_COLUMNS = {"prompt_id", "romanized", "devanagari", "verified"}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        example = path.with_name("hdsd_prompts.example.csv").name
        raise FileNotFoundError(
            f"Prompt table not found: {path}. It holds corpus sentences, so "
            f"it is git-ignored rather than shipped. Copy {example} next to "
            f"it and fill it in from the corpus you hold."
        )
    with open(path, encoding="utf-8", newline="") as fh:
        # Comment lines let the example file document its own columns.
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        reader = csv.DictReader(lines)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
        rows = [r for r in reader if (r.get("prompt_id") or "").strip()]
    blank = [r["prompt_id"] for r in rows if not (r.get("devanagari") or "").strip()]
    if blank:
        raise ValueError(
            f"No Devanagari for {', '.join(blank)}. Fill those in before seeding: "
            "a blank ground truth would silently disable scoring for that sentence."
        )
    return rows


async def _seed(dry_run: bool) -> int:
    rows = load_rows(PROMPTS_CSV)
    unverified = sum(1 for r in rows if (r["verified"] or "").strip().lower() != "yes")

    if dry_run:
        print(f"{PROMPTS_CSV}: {len(rows)} prompts")
        print(f"  unverified: {unverified}")
        for r in rows[:5]:
            print(f"  {r['prompt_id']}  {len(r['devanagari'].split())} words")
        print("  ... (dry run, nothing written)")
        return 0

    inserted = updated = 0
    async with SessionLocal() as db:
        existing = {
            p.prompt_id: p for p in (await db.execute(select(DatasetPrompt))).scalars()
        }
        for r in rows:
            pid = r["prompt_id"].strip()
            verified = (r["verified"] or "").strip().lower() == "yes"
            current = existing.get(pid)
            if current is None:
                db.add(
                    DatasetPrompt(
                        prompt_id=pid,
                        romanized=r["romanized"].strip(),
                        devanagari=r["devanagari"].strip(),
                        verified=verified,
                    )
                )
                inserted += 1
            else:
                current.romanized = r["romanized"].strip()
                current.devanagari = r["devanagari"].strip()
                current.verified = verified
                updated += 1
        await db.commit()

    print(f"prompts seeded: {inserted} inserted, {updated} updated")
    if unverified:
        print(
            f"  NOTE: {unverified} prompt(s) are not marked verified. The app will "
            "label their scores as provisional until the CSV says verified=yes."
        )
    return 0


async def _run(dry_run: bool) -> int:
    try:
        return await _seed(dry_run)
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be seeded."
    )
    args = parser.parse_args()
    configure_logging()
    return asyncio.run(_run(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
