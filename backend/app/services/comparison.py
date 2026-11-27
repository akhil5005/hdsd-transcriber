"""Live comparison of model outputs on a user's own upload (section 5b).

Two modes, decided by whether a reference transcript was supplied:

* **With a reference** -- WER, CER and S/D/I per model, plus a word-level diff
  against the reference for highlighting.
* **Without** -- transcripts and timings only. Models are never scored
  against each other: without ground truth there is no accuracy to report,
  and a models-agree-with-each-other number was too easily read as one.

Everything is computed on text already put through
:func:`app.services.normalization.normalize_text`, so a score always refers to
the same text shown in the normalized view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.metrics import AlignmentOp, OpKind, ScoreResult, score
from app.services.normalization import normalize_text


@dataclass(frozen=True, slots=True)
class DiffToken:
    """One token of the word-level diff shown against the reference."""

    kind: str
    reference: str | None
    hypothesis: str | None

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
        }


def build_diff(ops: tuple[AlignmentOp, ...]) -> list[DiffToken]:
    """Turn an alignment into tokens the UI can colour directly."""
    return [
        DiffToken(kind=str(op.kind), reference=op.ref, hypothesis=op.hyp)
        for op in ops
    ]


def diff_counts(ops: tuple[AlignmentOp, ...]) -> dict[str, int]:
    counts = {str(kind): 0 for kind in OpKind}
    for op in ops:
        counts[str(op.kind)] += 1
    return counts


@dataclass(frozen=True, slots=True)
class ScoredModel:
    """One model's scored result."""

    model_key: str
    scores: ScoreResult
    diff: list[DiffToken]


def score_against_reference(
    reference_text: str | None, transcripts: dict[str, str]
) -> dict[str, ScoredModel]:
    """Score each model's normalized transcript against the reference.

    ``transcripts`` maps model key -> already-normalized text. Returns an empty
    mapping when no usable reference was given: a whitespace- or
    punctuation-only reference normalizes to "" and is treated as absent rather
    than scored against, which would otherwise produce meaningless counts.
    """
    reference = normalize_text(reference_text)
    if not reference:
        return {}

    scored: dict[str, ScoredModel] = {}
    for model_key, hypothesis in transcripts.items():
        result = score(reference, hypothesis)
        scored[model_key] = ScoredModel(
            model_key=model_key,
            scores=result,
            diff=build_diff(result.word_ops),
        )
    return scored


def unanimous(transcripts: dict[str, str]) -> bool:
    """True when every model produced the same normalized text."""
    return len(set(transcripts.values())) <= 1


def summarize(
    transcripts: dict[str, str], reference_text: str | None = None
) -> dict[str, Any]:
    """Everything the compare view needs, in one JSON-ready mapping."""
    scored = score_against_reference(reference_text, transcripts)
    has_reference = bool(scored)

    best_model = None
    if has_reference:
        ranked = [
            (entry.scores.wer, key)
            for key, entry in scored.items()
            if entry.scores.wer is not None
        ]
        if ranked:
            best_model = min(ranked)[1]

    return {
        "has_reference": has_reference,
        "models": sorted(transcripts),
        "unanimous": unanimous(transcripts),
        "lowest_wer_model": best_model,
        "scores": {
            key: {
                "wer": entry.scores.wer,
                "cer": entry.scores.cer,
                "substitutions": entry.scores.substitutions,
                "deletions": entry.scores.deletions,
                "insertions": entry.scores.insertions,
                "reference_words": entry.scores.reference_words,
                "diff": [token.as_json() for token in entry.diff],
                "diff_counts": diff_counts(entry.scores.word_ops),
            }
            for key, entry in scored.items()
        },
    }
