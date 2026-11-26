"""Read the frozen research tables and seed them into the database.

Section 5a. The three CSVs under ``BENCHMARK_DIR`` are read through
:func:`app.core.research_assets.benchmark_csv`, which allow-lists exactly those
three filenames -- the rest of that directory holds private patient transcripts.

Seeding is idempotent: every row is keyed on
``(model_key, run, split)`` / ``key``, so re-running updates in place rather
than duplicating. Nothing is ever written back to the research archive.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, settings
from app.core.logging import get_logger
from app.core.research_assets import benchmark_csv
from app.models.benchmark import BenchmarkMetric
from app.models.enums import ModelKey, ModelKind
from app.models.registry import ModelRegistry

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One of the three evaluation cases. They must never be pooled."""

    split: str
    label: str
    speaker_familiarity: str
    sentence_familiarity: str
    utterances: int
    is_primary: bool


#: Section 5a. Ordered primary-first, which is the order the UI charts use.
EVALUATION_CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase(
        split="test",
        label="Unfamiliar speaker + unfamiliar sentence",
        speaker_familiarity="unfamiliar",
        sentence_familiarity="unfamiliar",
        utterances=45,
        is_primary=True,
    ),
    EvaluationCase(
        split="test_speaker_train_prompt",
        label="Unfamiliar speaker + familiar sentence",
        speaker_familiarity="unfamiliar",
        sentence_familiarity="familiar",
        utterances=189,
        is_primary=False,
    ),
    EvaluationCase(
        split="train_speaker_test_prompt",
        label="Familiar speaker + unfamiliar sentence",
        speaker_familiarity="familiar",
        sentence_familiarity="unfamiliar",
        utterances=213,
        is_primary=False,
    ),
)

CASES_BY_SPLIT = {case.split: case for case in EVALUATION_CASES}

#: The research tables' `mode` column -> the API's model key.
MODE_TO_KEY = {
    "pretrained": ModelKey.PRETRAINED.value,
    "full": ModelKey.FULL_FT.value,
    "lora": ModelKey.LORA.value,
}

MODE_TO_KIND = {
    "pretrained": ModelKind.PRETRAINED,
    "full": ModelKind.FULL,
    "lora": ModelKind.LORA,
}

#: Must not be pooled -- shown under every chart (section 5a).
BENCHMARK_DISCLAIMER = (
    "Exploratory research results on the HDSD dataset; the three evaluation "
    "cases differ in speakers and sentences and must not be pooled. "
    "Not a clinical assessment."
)


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _read_csv(filename: str, config: Settings) -> list[dict[str, str]]:
    path: Path = benchmark_csv(filename, config)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_runs(config: Settings | None = None) -> dict[str, str]:
    """The run directory name backing each model key.

    Derived from the configured checkpoint paths rather than rebuilt from the
    seed numbers, so switching ``FULL_FT_DIR`` automatically re-points which
    run is flagged selected.
    """
    config = config or settings
    return {
        ModelKey.PRETRAINED.value: "baseline_dev",
        ModelKey.FULL_FT.value: config.full_ft_dir.parent.name,
        ModelKey.LORA.value: config.lora_dir.parent.name,
    }


# --- readers ----------------------------------------------------------------
def read_per_run(config: Settings | None = None) -> list[dict[str, Any]]:
    """One row per (run, split) from ``per_run.csv``."""
    config = config or settings
    chosen = selected_runs(config)

    rows: list[dict[str, Any]] = []
    for raw in _read_csv("per_run.csv", config):
        mode = raw["mode"]
        key = MODE_TO_KEY.get(mode)
        if key is None:
            logger.warning("Skipping unknown mode %r in per_run.csv", mode)
            continue
        rows.append(
            {
                "model_key": key,
                "run": raw["run"],
                "seed": _int(raw.get("seed")),
                "is_selected": raw["run"] == chosen[key],
                "split": raw["split"],
                "utterances": _int(raw.get("utterances")),
                "wer": _float(raw.get("wer_percent")),
                "cer": _float(raw.get("cer_percent")),
                "macro_speaker_wer": _float(raw.get("macro_speaker_wer_percent")),
                "substitutions": _int(raw.get("substitutions")),
                "deletions": _int(raw.get("deletions")),
                "insertions": _int(raw.get("insertions")),
                "reference_units": _int(raw.get("reference_units")),
            }
        )
    return rows


def read_model_summary(config: Settings | None = None) -> list[dict[str, Any]]:
    """Seed mean +/- SD per (mode, split) from ``model_summary.csv``."""
    config = config or settings
    rows: list[dict[str, Any]] = []
    for raw in _read_csv("model_summary.csv", config):
        key = MODE_TO_KEY.get(raw["mode"])
        if key is None:
            continue
        rows.append(
            {
                "model_key": key,
                "split": raw["split"],
                "seed_count": _int(raw.get("seed_count")),
                "mean_wer": _float(raw.get("mean_wer_percent")),
                "sd_wer": _float(raw.get("sample_sd_wer_pp")),
                "mean_cer": _float(raw.get("mean_cer_percent")),
                "sd_cer": _float(raw.get("sample_sd_cer_pp")),
                "mean_substitutions": _float(raw.get("mean_substitutions")),
                "mean_deletions": _float(raw.get("mean_deletions")),
                "mean_insertions": _float(raw.get("mean_insertions")),
                "reference_words": _int(raw.get("reference_words")),
            }
        )
    return rows


def read_efficiency(config: Settings | None = None) -> list[dict[str, Any]]:
    """Trainable parameters, checkpoint size and training peak memory."""
    config = config or settings
    chosen = selected_runs(config)

    rows: list[dict[str, Any]] = []
    for raw in _read_csv("efficiency.csv", config):
        mode = raw["mode"]
        key = MODE_TO_KEY.get(mode)
        if key is None:
            continue
        rows.append(
            {
                "model_key": key,
                "run": raw["run"],
                "seed": _int(raw.get("seed")),
                "is_selected": raw["run"] == chosen[key],
                "trainable_parameters": _int(raw.get("trainable_parameters")) or 0,
                "checkpoint_bytes": _int(raw.get("selected_checkpoint_bytes")),
                "adaptation_minutes": _float(
                    raw.get("recorded_adaptation_end_to_end_minutes")
                ),
                "adaptation_peak_gib": _float(
                    raw.get("recorded_adaptation_peak_allocated_gib")
                ),
                "inference_seconds_447": _float(
                    raw.get("inference_seconds_447_recordings")
                ),
                "checkpoint_scope": raw.get("checkpoint_scope") or "",
            }
        )
    return rows


# --- seeding ----------------------------------------------------------------
async def seed_benchmark_metrics(
    db: AsyncSession, config: Settings | None = None
) -> dict[str, int]:
    """Upsert every (model_key, run, split) row. Idempotent."""
    config = config or settings
    rows = read_per_run(config)

    existing = {
        (metric.model_key, metric.run, metric.split): metric
        for metric in (await db.execute(select(BenchmarkMetric))).scalars()
    }

    inserted = updated = 0
    for row in rows:
        identity = (row["model_key"], row["run"], row["split"])
        metric = existing.get(identity)
        if metric is None:
            db.add(BenchmarkMetric(**row))
            inserted += 1
            continue
        for field, value in row.items():
            setattr(metric, field, value)
        updated += 1

    await db.commit()
    logger.info(
        "Seeded benchmark_metrics: %d inserted, %d updated", inserted, updated
    )
    return {"inserted": inserted, "updated": updated, "total": len(rows)}


async def seed_models_registry(
    db: AsyncSession, config: Settings | None = None
) -> dict[str, int]:
    """Upsert the three-model registry from config + efficiency.csv."""
    config = config or settings
    efficiency = {
        (row["model_key"], row["run"]): row for row in read_efficiency(config)
    }
    chosen = selected_runs(config)

    # Imported here to avoid a module-level dependency on the ML stack.
    from app.services.transcription import model_specs

    specs = model_specs(config)
    kinds = {
        ModelKey.PRETRAINED.value: ModelKind.PRETRAINED,
        ModelKey.FULL_FT.value: ModelKind.FULL,
        ModelKey.LORA.value: ModelKind.LORA,
    }

    existing = {
        entry.key: entry
        for entry in (await db.execute(select(ModelRegistry))).scalars()
    }

    inserted = updated = 0
    for key, spec in specs.items():
        eff = efficiency.get((key, chosen[key]), {})
        payload = {
            "display_name": spec.display_name,
            "kind": kinds[key],
            "learning_rate": spec.learning_rate,
            "seed": spec.seed,
            "path_or_revision": spec.source,
            "trainable_params": eff.get("trainable_parameters", 0),
            "checkpoint_bytes": eff.get("checkpoint_bytes"),
            "is_active": True,
        }
        entry = existing.get(key)
        if entry is None:
            db.add(ModelRegistry(key=key, **payload))
            inserted += 1
            continue
        for field, value in payload.items():
            setattr(entry, field, value)
        updated += 1

    await db.commit()
    logger.info("Seeded models_registry: %d inserted, %d updated", inserted, updated)
    return {"inserted": inserted, "updated": updated, "total": len(specs)}


async def seed_all(
    db: AsyncSession, config: Settings | None = None
) -> dict[str, dict[str, int]]:
    return {
        "models_registry": await seed_models_registry(db, config),
        "benchmark_metrics": await seed_benchmark_metrics(db, config),
    }
