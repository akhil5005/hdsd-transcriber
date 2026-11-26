"""Benchmark ingestion and endpoints (section 5a).

The CSV readers run against the real frozen tables -- that is the point, since
a schema drift in the research archive should fail here and not silently
produce empty charts. They skip if the archive is absent.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.core.research_assets import ResearchAssetError
from app.models import BenchmarkMetric, ModelRegistry
from app.services import benchmark as bench

#: The published figures from PROMPT.md section 5a, WER/CER in percent.
SPEC_TABLE = {
    ("pretrained", "test"): (71.51, 43.69),
    ("full_ft", "test"): (57.82, 32.05),
    ("lora", "test"): (65.92, 34.34),
    ("pretrained", "test_speaker_train_prompt"): (93.03, 57.61),
    ("full_ft", "test_speaker_train_prompt"): (16.72, 9.77),
    ("lora", "test_speaker_train_prompt"): (35.88, 19.33),
    ("pretrained", "train_speaker_test_prompt"): (75.17, 43.05),
    ("full_ft", "train_speaker_test_prompt"): (61.75, 34.17),
    ("lora", "train_speaker_test_prompt"): (61.93, 32.97),
}


def _archive_available() -> bool:
    try:
        bench.read_per_run(Settings())
    except (ResearchAssetError, OSError):
        return False
    return True


needs_archive = pytest.mark.skipif(
    not _archive_available(), reason="research benchmark CSVs not available"
)


# --- evaluation cases -------------------------------------------------------
def test_three_evaluation_cases_in_spec_order() -> None:
    splits = [case.split for case in bench.EVALUATION_CASES]
    assert splits == ["test", "test_speaker_train_prompt", "train_speaker_test_prompt"]


def test_primary_case_is_unfamiliar_speaker_and_sentence() -> None:
    primary = [case for case in bench.EVALUATION_CASES if case.is_primary]
    assert len(primary) == 1
    assert primary[0].split == "test"
    assert primary[0].utterances == 45


def test_case_sizes_match_the_spec() -> None:
    sizes = {case.split: case.utterances for case in bench.EVALUATION_CASES}
    assert sizes == {
        "test": 45,
        "test_speaker_train_prompt": 189,
        "train_speaker_test_prompt": 213,
    }


def test_disclaimer_forbids_pooling_and_disclaims_clinical_use() -> None:
    text = bench.BENCHMARK_DISCLAIMER.lower()
    assert "must not be pooled" in text
    assert "not a clinical assessment" in text


# --- selected runs ----------------------------------------------------------
def test_selected_runs_follow_the_configured_checkpoint_paths() -> None:
    runs = bench.selected_runs(Settings(_env_file=None))
    assert runs["pretrained"] == "baseline_dev"
    assert runs["full_ft"] == "full_lr5e-06_seed123"
    assert runs["lora"] == "lora_lr5e-05_seed2026"


def test_changing_the_checkpoint_path_moves_the_selected_run(tmp_path) -> None:
    config = Settings(_env_file=None, full_ft_dir=tmp_path / "full_lr5e-06_seed42" / "best")
    assert bench.selected_runs(config)["full_ft"] == "full_lr5e-06_seed42"


# --- CSV readers ------------------------------------------------------------
@needs_archive
def test_per_run_covers_every_run_and_case() -> None:
    rows = bench.read_per_run(Settings())
    # 7 runs x 3 evaluation cases.
    assert len(rows) == 21
    assert {row["split"] for row in rows} == {
        case.split for case in bench.EVALUATION_CASES
    }
    assert {row["model_key"] for row in rows} == {"pretrained", "full_ft", "lora"}


@needs_archive
def test_per_run_values_match_the_published_table() -> None:
    rows = [row for row in bench.read_per_run(Settings()) if row["is_selected"]]
    actual = {(row["model_key"], row["split"]): (row["wer"], row["cer"]) for row in rows}

    assert len(actual) == 9
    for identity, (wer, cer) in SPEC_TABLE.items():
        assert actual[identity][0] == pytest.approx(wer, abs=0.01), identity
        assert actual[identity][1] == pytest.approx(cer, abs=0.01), identity


@needs_archive
def test_exactly_one_selected_run_per_model_and_case() -> None:
    rows = bench.read_per_run(Settings())
    selected = [row for row in rows if row["is_selected"]]
    seen = [(row["model_key"], row["split"]) for row in selected]
    assert len(seen) == len(set(seen)) == 9


@needs_archive
def test_model_summary_carries_seed_mean_and_sd() -> None:
    rows = {
        (row["model_key"], row["split"]): row
        for row in bench.read_model_summary(Settings())
    }
    full = rows[("full_ft", "test")]
    lora = rows[("lora", "test")]
    pretrained = rows[("pretrained", "test")]

    assert full["seed_count"] == 3
    assert full["mean_wer"] == pytest.approx(63.04, abs=0.01)
    assert full["sd_wer"] == pytest.approx(6.97, abs=0.01)
    assert lora["mean_wer"] == pytest.approx(63.31, abs=0.01)
    assert lora["sd_wer"] == pytest.approx(3.18, abs=0.01)
    # A single run has no sample SD.
    assert pretrained["seed_count"] == 1
    assert pretrained["sd_wer"] is None


@needs_archive
def test_efficiency_matches_the_published_figures() -> None:
    rows = {
        (row["model_key"], row["is_selected"]): row
        for row in bench.read_efficiency(Settings())
    }
    full = rows[("full_ft", True)]
    lora = rows[("lora", True)]
    pretrained = rows[("pretrained", True)]

    assert full["trainable_parameters"] == 240_582_912
    assert full["checkpoint_bytes"] == 972_633_406
    assert full["adaptation_peak_gib"] == pytest.approx(4.6, abs=0.05)
    assert lora["trainable_parameters"] == 1_769_472
    assert lora["checkpoint_bytes"] == 12_737_654
    assert lora["adaptation_peak_gib"] == pytest.approx(1.89, abs=0.05)
    assert pretrained["trainable_parameters"] == 0
    assert "requires pinned base" in lora["checkpoint_scope"]


@needs_archive
def test_readers_refuse_files_outside_the_allow_list(tmp_path) -> None:
    """The benchmark directory also holds private transcripts."""
    config = Settings(_env_file=None, benchmark_dir=tmp_path)
    with pytest.raises(ResearchAssetError):
        bench.read_per_run(config)


# --- seeding ----------------------------------------------------------------
@needs_archive
async def test_seeding_populates_both_tables(db) -> None:
    result = await bench.seed_all(db, Settings())

    assert result["benchmark_metrics"]["inserted"] == 21
    assert result["models_registry"]["inserted"] == 3

    metrics = (await db.execute(select(BenchmarkMetric))).scalars().all()
    assert len(metrics) == 21
    registry = (await db.execute(select(ModelRegistry))).scalars().all()
    assert {entry.key for entry in registry} == {"pretrained", "full_ft", "lora"}


@needs_archive
async def test_seeding_is_idempotent(db) -> None:
    await bench.seed_all(db, Settings())
    second = await bench.seed_all(db, Settings())

    assert second["benchmark_metrics"]["inserted"] == 0
    assert second["benchmark_metrics"]["updated"] == 21
    assert second["models_registry"]["inserted"] == 0

    total = (await db.execute(select(BenchmarkMetric))).scalars().all()
    assert len(total) == 21, "re-seeding must not duplicate rows"


@needs_archive
async def test_seeding_flags_the_selected_seed_only(db) -> None:
    await bench.seed_benchmark_metrics(db, Settings())
    rows = (
        await db.execute(
            select(BenchmarkMetric).where(BenchmarkMetric.is_selected.is_(True))
        )
    ).scalars().all()

    assert len(rows) == 9  # 3 models x 3 cases
    by_model = {}
    for row in rows:
        by_model.setdefault(row.model_key, set()).add(row.run)
    assert by_model["full_ft"] == {"full_lr5e-06_seed123"}
    assert by_model["lora"] == {"lora_lr5e-05_seed2026"}


@needs_archive
async def test_registry_records_efficiency_figures(db) -> None:
    await bench.seed_models_registry(db, Settings())
    rows = {
        entry.key: entry
        for entry in (await db.execute(select(ModelRegistry))).scalars()
    }

    assert rows["full_ft"].trainable_params == 240_582_912
    assert rows["full_ft"].seed == 123
    assert rows["lora"].trainable_params == 1_769_472
    assert rows["lora"].seed == 2026
    assert rows["pretrained"].trainable_params == 0
    assert "973afd24" in rows["pretrained"].path_or_revision


# --- endpoints --------------------------------------------------------------
async def test_benchmark_requires_authentication(client) -> None:
    assert (await client.get("/api/v1/benchmark")).status_code == 401
    assert (await client.get("/api/v1/benchmark/summary")).status_code == 401


@needs_archive
async def test_benchmark_endpoint_returns_rows_and_disclaimer(
    client, db, tokens
) -> None:
    await bench.seed_all(db, Settings())
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    body = (await client.get("/api/v1/benchmark", headers=headers)).json()

    assert len(body["rows"]) == 21
    assert "must not be pooled" in body["disclaimer"]
    assert len(body["evaluation_cases"]) == 3
    assert body["evaluation_cases"][0]["is_primary"] is True
    assert body["selected_runs"]["full_ft"] == "full_lr5e-06_seed123"


@needs_archive
async def test_benchmark_selected_only_filter(client, db, tokens) -> None:
    await bench.seed_all(db, Settings())
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    body = (
        await client.get(
            "/api/v1/benchmark", params={"selected_only": True}, headers=headers
        )
    ).json()

    assert len(body["rows"]) == 9
    assert all(row["is_selected"] for row in body["rows"])


@needs_archive
async def test_summary_is_shaped_for_grouped_bar_charts(client, db, tokens) -> None:
    await bench.seed_all(db, Settings())
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    body = (await client.get("/api/v1/benchmark/summary", headers=headers)).json()

    assert body["case_labels"][0].startswith("Unfamiliar speaker + unfamiliar")
    assert [s["model_key"] for s in body["wer_by_case"]] == [
        "pretrained",
        "full_ft",
        "lora",
    ]
    # One value per evaluation case, in case order.
    for series in body["wer_by_case"]:
        assert len(series["values"]) == 3

    primary_wer = {s["model_key"]: s["values"][0] for s in body["wer_by_case"]}
    assert primary_wer["pretrained"] == pytest.approx(71.51, abs=0.01)
    assert primary_wer["full_ft"] == pytest.approx(57.82, abs=0.01)
    assert primary_wer["lora"] == pytest.approx(65.92, abs=0.01)


@needs_archive
async def test_summary_includes_seed_spread_and_efficiency(client, db, tokens) -> None:
    await bench.seed_all(db, Settings())
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    body = (await client.get("/api/v1/benchmark/summary", headers=headers)).json()

    spread = {
        (row["model_key"], row["split"]): row for row in body["seed_spread"]
    }
    assert spread[("full_ft", "test")]["sd_wer"] == pytest.approx(6.97, abs=0.01)

    efficiency = {
        row["run"]: row for row in body["efficiency"] if row["is_selected"]
    }
    assert efficiency["lora_lr5e-05_seed2026"]["checkpoint_bytes"] == 12_737_654

    breakdown = body["error_breakdown"]
    assert len(breakdown) == 9
    assert all(
        {"substitutions", "deletions", "insertions"} <= set(row) for row in breakdown
    )
