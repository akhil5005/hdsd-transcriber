"""Seed the research benchmark tables into the database.

    python -m app.scripts.seed_benchmark

Idempotent: safe to re-run after a migration or a checkpoint-path change.
Reads only the three allow-listed CSVs under BENCHMARK_DIR; writes nothing
back to the research archive.
"""

from __future__ import annotations

import argparse
import asyncio

from app.core.logging import configure_logging
from app.core.research_assets import ResearchAssetError
from app.db.session import SessionLocal, engine
from app.services import benchmark


async def _run(dry_run: bool) -> int:
    try:
        return await _seed(dry_run)
    finally:
        # Must happen on the same event loop that opened the connections.
        await engine.dispose()


async def _seed(dry_run: bool) -> int:
    if dry_run:
        per_run = benchmark.read_per_run()
        summary = benchmark.read_model_summary()
        efficiency = benchmark.read_efficiency()
        print(f"per_run.csv       : {len(per_run)} rows")
        print(f"model_summary.csv : {len(summary)} rows")
        print(f"efficiency.csv    : {len(efficiency)} rows")
        print()
        print("selected runs:")
        for key, run in benchmark.selected_runs().items():
            print(f"  {key:<11} -> {run}")
        print()
        print("evaluation cases:")
        for case in benchmark.EVALUATION_CASES:
            flag = " (primary)" if case.is_primary else ""
            print(f"  {case.split:<28} n={case.utterances:<4} {case.label}{flag}")
        return 0

    async with SessionLocal() as session:
        result = await benchmark.seed_all(session)

    for table, counts in result.items():
        print(
            f"{table:<18} inserted={counts['inserted']:<4} "
            f"updated={counts['updated']:<4} total={counts['total']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.scripts.seed_benchmark")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the CSVs and report what would be seeded, without writing.",
    )
    args = parser.parse_args(argv)
    configure_logging()

    try:
        return asyncio.run(_run(args.dry_run))
    except ResearchAssetError as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
