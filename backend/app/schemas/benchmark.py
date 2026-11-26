"""Response models for the research benchmark endpoints (section 5a)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCaseOut(BaseModel):
    split: str
    label: str
    speaker_familiarity: str
    sentence_familiarity: str
    utterances: int
    is_primary: bool


class BenchmarkRow(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    model_key: str
    run: str
    seed: int | None = None
    is_selected: bool
    split: str
    utterances: int | None = None
    wer: float | None = None
    cer: float | None = None
    macro_speaker_wer: float | None = None
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None
    reference_units: int | None = None


class SeedSpread(BaseModel):
    """Mean +/- sample SD across training seeds, for the error-bar view."""

    model_config = ConfigDict(protected_namespaces=())

    model_key: str
    split: str
    seed_count: int | None = None
    mean_wer: float | None = None
    sd_wer: float | None = None
    mean_cer: float | None = None
    sd_cer: float | None = None
    mean_substitutions: float | None = None
    mean_deletions: float | None = None
    mean_insertions: float | None = None
    reference_words: int | None = None


class EfficiencyRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_key: str
    run: str
    seed: int | None = None
    is_selected: bool
    trainable_parameters: int
    checkpoint_bytes: int | None = None
    adaptation_minutes: float | None = None
    adaptation_peak_gib: float | None = None
    inference_seconds_447: float | None = None
    checkpoint_scope: str


class ErrorBreakdown(BaseModel):
    """Stacked S/D/I for one (model, case)."""

    model_config = ConfigDict(protected_namespaces=())

    model_key: str
    split: str
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None
    reference_units: int | None = None


class BenchmarkResponse(BaseModel):
    """Everything in `GET /benchmark`."""

    disclaimer: str
    evaluation_cases: list[EvaluationCaseOut]
    rows: list[BenchmarkRow]
    selected_runs: dict[str, str]


class ChartSeries(BaseModel):
    """One grouped-bar series: a model across the three evaluation cases."""

    model_config = ConfigDict(protected_namespaces=())

    model_key: str
    display_name: str
    values: list[float | None]


class BenchmarkSummary(BaseModel):
    """Chart-shaped payload for `GET /benchmark/summary` (section 5a)."""

    disclaimer: str
    evaluation_cases: list[EvaluationCaseOut]
    #: Axis labels, primary case first.
    case_labels: list[str]
    wer_by_case: list[ChartSeries]
    cer_by_case: list[ChartSeries]
    macro_speaker_wer_by_case: list[ChartSeries]
    seed_spread: list[SeedSpread]
    error_breakdown: list[ErrorBreakdown]
    efficiency: list[EfficiencyRow]
    selected_runs: dict[str, str]
    notes: list[str] = Field(default_factory=list)
