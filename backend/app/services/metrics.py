"""WER / CER with substitution, deletion and insertion counts.

Written out rather than delegated to `jiwer` because section 5b fixes details
a library will not promise: the tie-break order during backtrace, and a
word-level alignment the UI needs for diff highlighting. The test suite
cross-checks the error counts against `jiwer` so this stays honest.

Definitions used throughout (section 5b):

* **WER** counts whitespace-delimited words.
* **CER** counts Unicode code points **including spaces**.
* A **deletion** is a reference unit absent from the hypothesis; an
  **insertion** is an extra hypothesis unit.
* Rates are reported as **percentages**, matching the research benchmark
  tables, so a live score and a benchmark score are directly comparable.

Both inputs must already be normalized (see
:func:`app.services.normalization.normalize_text`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.services.normalization import characters, words


class OpKind(StrEnum):
    EQUAL = "equal"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


@dataclass(frozen=True, slots=True)
class AlignmentOp:
    """One aligned pair. ``ref``/``hyp`` is ``None`` where the side is absent."""

    kind: OpKind
    ref: str | None
    hyp: str | None


@dataclass(frozen=True, slots=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_units: int

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float | None:
        """Errors per reference unit, as a **percentage**.

        ``None`` when the reference is empty: there is no denominator, and
        inventing one (0 or 100) would be a silent lie about an unscoreable
        pair. Callers store ``NULL`` and the UI shows a dash.
        """
        if self.reference_units == 0:
            return None
        return 100.0 * self.total_errors / self.reference_units


@dataclass(frozen=True, slots=True)
class Alignment:
    counts: ErrorCounts
    ops: tuple[AlignmentOp, ...]


# Backtrace preference, fixed by section 5b: substitution, then deletion,
# then insertion. Only matters when several paths share the minimal cost, but
# it must be deterministic or the diff highlight would flicker between runs.
def _align(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    n, m = len(reference), len(hypothesis)

    # distance[i][j] = edit distance between reference[:i] and hypothesis[:j].
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        distance[i][0] = i
    for j in range(1, m + 1):
        distance[0][j] = j

    for i in range(1, n + 1):
        row, prev = distance[i], distance[i - 1]
        ref_token = reference[i - 1]
        for j in range(1, m + 1):
            diagonal = prev[j - 1] + (0 if ref_token == hypothesis[j - 1] else 1)
            row[j] = min(diagonal, prev[j] + 1, row[j - 1] + 1)

    ops: list[AlignmentOp] = []
    substitutions = deletions = insertions = hits = 0
    i, j = n, m

    while i > 0 or j > 0:
        # Diagonal first (equal or substitution), then deletion, then insertion.
        if i > 0 and j > 0:
            ref_token, hyp_token = reference[i - 1], hypothesis[j - 1]
            cost = 0 if ref_token == hyp_token else 1
            if distance[i][j] == distance[i - 1][j - 1] + cost:
                if cost == 0:
                    hits += 1
                    ops.append(AlignmentOp(OpKind.EQUAL, ref_token, hyp_token))
                else:
                    substitutions += 1
                    ops.append(AlignmentOp(OpKind.SUBSTITUTION, ref_token, hyp_token))
                i, j = i - 1, j - 1
                continue

        if i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1
            ops.append(AlignmentOp(OpKind.DELETION, reference[i - 1], None))
            i -= 1
            continue

        # Only insertion remains reachable here.
        insertions += 1
        ops.append(AlignmentOp(OpKind.INSERTION, None, hypothesis[j - 1]))
        j -= 1

    ops.reverse()
    return Alignment(
        counts=ErrorCounts(
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            hits=hits,
            reference_units=n,
        ),
        ops=tuple(ops),
    )


def word_alignment(reference: str, hypothesis: str) -> Alignment:
    """Word-level alignment. Feeds both WER and the UI diff highlight."""
    return _align(words(reference), hypothesis=words(hypothesis))


def character_alignment(reference: str, hypothesis: str) -> Alignment:
    """Code-point-level alignment, spaces included."""
    return _align(characters(reference), hypothesis=characters(hypothesis))


def wer(reference: str, hypothesis: str) -> float | None:
    """Word error rate as a percentage, or ``None`` for an empty reference."""
    return word_alignment(reference, hypothesis).counts.error_rate


def cer(reference: str, hypothesis: str) -> float | None:
    """Character error rate as a percentage, or ``None`` for an empty reference."""
    return character_alignment(reference, hypothesis).counts.error_rate


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Everything the API stores for one (reference, hypothesis) pair.

    ``substitutions``/``deletions``/``insertions`` are the **word**-level
    counts, which is what section 6's columns and the research tables report.
    """

    wer: float | None
    cer: float | None
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    word_ops: tuple[AlignmentOp, ...]


def score(reference: str, hypothesis: str) -> ScoreResult:
    """Score a normalized hypothesis against a normalized reference."""
    word = word_alignment(reference, hypothesis)
    char = character_alignment(reference, hypothesis)

    return ScoreResult(
        wer=word.counts.error_rate,
        cer=char.counts.error_rate,
        substitutions=word.counts.substitutions,
        deletions=word.counts.deletions,
        insertions=word.counts.insertions,
        reference_words=word.counts.reference_units,
        word_ops=word.ops,
    )


def pairwise_agreement(left: str, right: str) -> float | None:
    """WER of one model's output against another's.

    Section 5b is explicit that this is **agreement, not accuracy** -- there is
    no ground truth involved, so the result is not symmetric and must never be
    labelled an error rate in the UI.
    """
    return wer(left, right)
