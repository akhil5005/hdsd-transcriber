"""Research benchmark endpoints (section 5a).

Served from the seeded tables -- ``benchmark_metrics``,
``benchmark_seed_spread`` and ``benchmark_efficiency`` -- not from the CSVs,
so a request never touches the research archive and the endpoints work on a
host that has no archive at all. ``/summary`` reshapes the same rows for the
charts; the three evaluation cases stay separate everywhere because pooling
them would be scientifically wrong.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.models.benchmark import (
    BenchmarkEfficiency,
    BenchmarkMetric,
    BenchmarkSeedSpread,
)
from app.models.registry import ModelRegistry
from app.schemas.benchmark import (
    BenchmarkResponse,
    BenchmarkRow,
    BenchmarkSummary,
    ChartSeries,
    EfficiencyRow,
    ErrorBreakdown,
    EvaluationCaseOut,
    SeedSpread,
)
from app.services import benchmark as bench

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

#: Chart series order: pretrained first, then the two adapted models.
MODEL_ORDER = ("pretrained", "full_ft", "lora")

SUMMARY_NOTES = [
    "Lower is better for WER and CER; both are percentages.",
    "The LoRA adapter also requires the pinned base model at inference time.",
    "Training seed was selected on development-set WER, never on the test set.",
]


def _cases() -> list[EvaluationCaseOut]:
    return [
        EvaluationCaseOut(
            split=case.split,
            label=case.label,
            speaker_familiarity=case.speaker_familiarity,
            sentence_familiarity=case.sentence_familiarity,
            utterances=case.utterances,
            is_primary=case.is_primary,
        )
        for case in bench.EVALUATION_CASES
    ]


@router.get("", response_model=BenchmarkResponse)
async def get_benchmark(
    _: CurrentUser,
    db: DbSession,
    selected_only: bool = Query(
        False, description="Only the development-selected seed per model."
    ),
) -> BenchmarkResponse:
    statement = select(BenchmarkMetric)
    if selected_only:
        statement = statement.where(BenchmarkMetric.is_selected.is_(True))
    rows = (await db.execute(statement)).scalars().all()

    order = {case.split: index for index, case in enumerate(bench.EVALUATION_CASES)}
    rows = sorted(
        rows,
        key=lambda row: (
            order.get(row.split, 99),
            MODEL_ORDER.index(row.model_key) if row.model_key in MODEL_ORDER else 99,
            row.run,
        ),
    )

    return BenchmarkResponse(
        disclaimer=bench.BENCHMARK_DISCLAIMER,
        evaluation_cases=_cases(),
        rows=[BenchmarkRow.model_validate(row) for row in rows],
        selected_runs=bench.selected_runs(),
    )


@router.get("/summary", response_model=BenchmarkSummary)
async def get_benchmark_summary(_: CurrentUser, db: DbSession) -> BenchmarkSummary:
    selected = (
        await db.execute(
            select(BenchmarkMetric).where(BenchmarkMetric.is_selected.is_(True))
        )
    ).scalars().all()
    registry = {
        entry.key: entry
        for entry in (await db.execute(select(ModelRegistry))).scalars()
    }

    # Both of these used to be read from the research archive's CSVs on every
    # request, which 500s on any host without that archive -- and quietly made
    # a read of patient-derived research files part of serving a page. They are
    # seeded into their own tables now, like every other benchmark figure.
    case_order = {case.split: index for index, case in enumerate(bench.EVALUATION_CASES)}

    def model_rank(model_key: str) -> int:
        return MODEL_ORDER.index(model_key) if model_key in MODEL_ORDER else 99

    # Ordered in Python, not SQL: these two lists are read in order by the
    # charts, and the meaningful order is the research one -- evaluation case,
    # then pretrained before the adapted models, then ascending seed. Sorting
    # by column would give alphabetical nonsense (full_ft first, seed 123
    # before 42).
    spread_rows = sorted(
        (await db.execute(select(BenchmarkSeedSpread))).scalars().all(),
        key=lambda row: (
            case_order.get(row.split, 99),
            model_rank(row.model_key),
        ),
    )
    efficiency_rows = sorted(
        (await db.execute(select(BenchmarkEfficiency))).scalars().all(),
        key=lambda row: (
            model_rank(row.model_key),
            row.seed if row.seed is not None else -1,
            row.run,
        ),
    )

    by_key_split = {(row.model_key, row.split): row for row in selected}
    cases = bench.EVALUATION_CASES

    def series(attribute: str) -> list[ChartSeries]:
        out: list[ChartSeries] = []
        for key in MODEL_ORDER:
            entry = registry.get(key)
            out.append(
                ChartSeries(
                    model_key=key,
                    display_name=entry.display_name if entry else key,
                    values=[
                        getattr(by_key_split.get((key, case.split)), attribute, None)
                        for case in cases
                    ],
                )
            )
        return out

    return BenchmarkSummary(
        disclaimer=bench.BENCHMARK_DISCLAIMER,
        evaluation_cases=_cases(),
        case_labels=[case.label for case in cases],
        wer_by_case=series("wer"),
        cer_by_case=series("cer"),
        macro_speaker_wer_by_case=series("macro_speaker_wer"),
        seed_spread=[
            SeedSpread.model_validate(row) for row in spread_rows
        ],
        error_breakdown=[
            ErrorBreakdown(
                model_key=key,
                split=case.split,
                substitutions=getattr(
                    by_key_split.get((key, case.split)), "substitutions", None
                ),
                deletions=getattr(
                    by_key_split.get((key, case.split)), "deletions", None
                ),
                insertions=getattr(
                    by_key_split.get((key, case.split)), "insertions", None
                ),
                reference_units=getattr(
                    by_key_split.get((key, case.split)), "reference_units", None
                ),
            )
            for case in cases
            for key in MODEL_ORDER
        ],
        efficiency=[EfficiencyRow.model_validate(row) for row in efficiency_rows],
        selected_runs=bench.selected_runs(),
        notes=SUMMARY_NOTES,
    )
