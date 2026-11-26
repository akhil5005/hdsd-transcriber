"""WER/CER tests.

Every expected number below is hand-computed in the comment above it, as
PROMPT.md section 5b requires, and the counts are additionally cross-checked
against `jiwer` so a refactor cannot quietly change the arithmetic.
"""

from __future__ import annotations

import pytest

from app.services.metrics import (
    OpKind,
    cer,
    character_alignment,
    pairwise_agreement,
    score,
    wer,
    word_alignment,
)
from app.services.normalization import normalize_text


# --- hand-computed word-level cases -----------------------------------------
def test_identical_text_scores_zero() -> None:
    assert wer("यह घर है", "यह घर है") == 0.0
    assert cer("यह घर है", "यह घर है") == 0.0


def test_one_substitution_in_three_words_is_33_percent() -> None:
    # ref: यह घर है | hyp: यह गाँव है -> 1 substitution, N=3 -> 1/3 = 33.33%
    counts = word_alignment("यह घर है", "यह गाँव है").counts
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 0)
    assert counts.hits == 2
    assert wer("यह घर है", "यह गाँव है") == pytest.approx(100 / 3)


def test_one_deletion_in_three_words_is_33_percent() -> None:
    # ref: यह घर है | hyp: यह है -> 1 deletion, N=3
    counts = word_alignment("यह घर है", "यह है").counts
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 1, 0)
    assert wer("यह घर है", "यह है") == pytest.approx(100 / 3)


def test_one_insertion_against_three_words_is_33_percent() -> None:
    # ref: यह घर है | hyp: यह बड़ा घर है -> 1 insertion, N=3
    counts = word_alignment("यह घर है", "यह बड़ा घर है").counts
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 1)
    assert wer("यह घर है", "यह बड़ा घर है") == pytest.approx(100 / 3)


def test_wer_can_exceed_one_hundred_percent() -> None:
    # ref: घर (N=1) | hyp: यह बड़ा घर है -> 3 insertions -> 300%
    counts = word_alignment("घर", "यह बड़ा घर है").counts
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 3)
    assert wer("घर", "यह बड़ा घर है") == pytest.approx(300.0)


def test_empty_hypothesis_deletes_every_reference_word() -> None:
    # ref: यह घर है | hyp: "" -> 3 deletions, N=3 -> 100%
    counts = word_alignment("यह घर है", "").counts
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 3, 0)
    assert wer("यह घर है", "") == pytest.approx(100.0)


def test_empty_reference_has_no_denominator() -> None:
    """Rather than invent a 0% or 100%, report None and store NULL."""
    assert wer("", "यह घर") is None
    assert cer("", "यह घर") is None
    assert word_alignment("", "यह घर").counts.insertions == 2


def test_both_empty_is_unscoreable_not_perfect() -> None:
    assert wer("", "") is None


def test_mixed_errors_are_counted_separately() -> None:
    # ref: यह बड़ा घर है        (N=4)
    # hyp: यह छोटा घर था बहुत
    #   यह=यह hit | बड़ा->छोटा sub | घर=घर hit | है->था sub | +बहुत ins
    #   -> S=2 D=0 I=1, total 3 errors / 4 = 75%
    counts = word_alignment("यह बड़ा घर है", "यह छोटा घर था बहुत").counts
    assert (counts.substitutions, counts.deletions, counts.insertions) == (2, 0, 1)
    assert counts.hits == 2
    assert counts.error_rate == pytest.approx(75.0)


# --- hand-computed character-level cases ------------------------------------
def test_cer_counts_code_points_including_spaces() -> None:
    # ref "यह घर" is 5 code points: य ह ␠ घ र
    # hyp "यह गर" substitutes घ->ग -> 1 error / 5 = 20%
    assert cer("यह घर", "यह गर") == pytest.approx(20.0)


def test_a_missing_space_is_a_character_error() -> None:
    # ref "यह घर" (5 cp) vs hyp "यहघर" (4 cp): the space is deleted -> 1/5 = 20%
    counts = character_alignment("यह घर", "यहघर").counts
    assert counts.deletions == 1
    assert counts.error_rate == pytest.approx(20.0)


def test_cer_is_lower_than_wer_for_a_one_character_slip() -> None:
    """The point of reporting both: CER is the kinder, finer-grained view."""
    reference, hypothesis = "यह घर है", "यह गर है"
    assert cer(reference, hypothesis) < wer(reference, hypothesis)


# --- alignment / tie-breaking ------------------------------------------------
def test_alignment_ops_reconstruct_both_sides_in_order() -> None:
    alignment = word_alignment("यह बड़ा घर है", "यह छोटा घर था बहुत")

    assert [op.ref for op in alignment.ops if op.ref] == ["यह", "बड़ा", "घर", "है"]
    assert [op.hyp for op in alignment.ops if op.hyp] == [
        "यह",
        "छोटा",
        "घर",
        "था",
        "बहुत",
    ]


def test_substitution_wins_ties_over_deletion_and_insertion() -> None:
    """One sub and one del+ins both cost 1 here; section 5b picks the sub."""
    alignment = word_alignment("a", "b")
    kinds = [op.kind for op in alignment.ops]

    assert kinds == [OpKind.SUBSTITUTION]
    assert alignment.counts.substitutions == 1
    assert alignment.counts.deletions == 0
    assert alignment.counts.insertions == 0


def test_tie_breaking_is_deterministic_across_calls() -> None:
    first = word_alignment("a b c d", "b a d c").ops
    for _ in range(5):
        assert word_alignment("a b c d", "b a d c").ops == first


def test_total_errors_plus_hits_covers_the_reference() -> None:
    counts = word_alignment("यह बड़ा घर है", "यह छोटा घर था बहुत").counts
    assert counts.hits + counts.substitutions + counts.deletions == counts.reference_units


# --- normalization is applied by the caller, not silently here --------------
def test_scoring_uses_the_shared_normalizer() -> None:
    """Punctuation and spacing differences must not count as errors."""
    reference = normalize_text("यह  घर है।")
    hypothesis = normalize_text("यह घर है")
    assert wer(reference, hypothesis) == 0.0


# --- score() bundle ---------------------------------------------------------
def test_score_bundles_word_counts_and_both_rates() -> None:
    result = score("यह बड़ा घर है", "यह छोटा घर था बहुत")

    assert (result.substitutions, result.deletions, result.insertions) == (2, 0, 1)
    assert result.reference_words == 4
    assert result.wer == pytest.approx(75.0)
    assert result.cer is not None
    assert result.word_ops


def test_score_on_empty_reference_keeps_counts_but_drops_rates() -> None:
    result = score("", "यह घर")

    assert result.wer is None and result.cer is None
    assert result.insertions == 2
    assert result.reference_words == 0


# --- agreement mode ---------------------------------------------------------
def test_pairwise_agreement_is_wer_between_two_hypotheses() -> None:
    assert pairwise_agreement("यह घर है", "यह गाँव है") == pytest.approx(100 / 3)


def test_pairwise_agreement_is_not_symmetric() -> None:
    """Why the UI must label it 'agreement, not accuracy'."""
    long_text, short_text = "यह बड़ा घर है", "घर"
    assert pairwise_agreement(long_text, short_text) != pairwise_agreement(
        short_text, long_text
    )


# --- cross-check against jiwer ----------------------------------------------
@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("यह घर है", "यह गाँव है"),
        ("यह घर है", "यह है"),
        ("यह घर है", "यह बड़ा घर है"),
        ("यह बड़ा घर है", "यह छोटा घर था बहुत"),
        ("घर", "यह बड़ा घर है"),
        ("यह घर है", ""),
        ("a b c d e", "a c b d x e"),
        ("one two three", "one two three"),
    ],
)
def test_counts_and_wer_match_jiwer(reference, hypothesis) -> None:
    jiwer = pytest.importorskip("jiwer")

    output = jiwer.process_words([reference], [hypothesis])
    counts = word_alignment(reference, hypothesis).counts

    assert counts.substitutions == output.substitutions
    assert counts.deletions == output.deletions
    assert counts.insertions == output.insertions
    assert counts.hits == output.hits
    assert counts.error_rate == pytest.approx(100.0 * output.wer)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("यह घर", "यह गर"),
        ("यह घर", "यहघर"),
        ("यह बड़ा घर है", "यह छोटा घर था बहुत"),
    ],
)
def test_cer_matches_jiwer(reference, hypothesis) -> None:
    jiwer = pytest.importorskip("jiwer")
    assert cer(reference, hypothesis) == pytest.approx(
        100.0 * jiwer.cer(reference, hypothesis)
    )
