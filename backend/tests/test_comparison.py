"""Live comparison logic (section 5b)."""

from __future__ import annotations

import pytest

from app.services import comparison
from app.services.normalization import normalize_text

THREE = {
    "pretrained": normalize_text("यह गाँव है"),
    "full_ft": normalize_text("यह घर है"),
    "lora": normalize_text("यह घर था"),
}
REFERENCE = "यह घर है"


# --- with a reference -------------------------------------------------------
def test_scores_every_model_against_the_reference() -> None:
    scored = comparison.score_against_reference(REFERENCE, THREE)

    assert set(scored) == {"pretrained", "full_ft", "lora"}
    assert scored["full_ft"].scores.wer == 0.0
    assert scored["pretrained"].scores.wer == pytest.approx(100 / 3)
    assert scored["lora"].scores.wer == pytest.approx(100 / 3)


def test_reference_punctuation_is_not_counted_as_an_error() -> None:
    """Scoring uses the shared normalizer, so a danda cannot cost a point."""
    scored = comparison.score_against_reference("यह घर है।", {"full_ft": "यह घर है"})
    assert scored["full_ft"].scores.wer == 0.0


@pytest.mark.parametrize("reference", [None, "", "   ", "।", ".,!"])
def test_a_blank_reference_counts_as_no_reference(reference) -> None:
    """A punctuation-only reference normalizes to '' -- scoring it would
    produce counts with no meaning, so it is treated as absent."""
    assert comparison.score_against_reference(reference, THREE) == {}


def test_diff_marks_each_word_with_its_operation() -> None:
    scored = comparison.score_against_reference(REFERENCE, THREE)
    diff = scored["lora"].diff

    kinds = [token.kind for token in diff]
    assert kinds.count("equal") == 2
    assert kinds.count("substitution") == 1
    substitution = next(t for t in diff if t.kind == "substitution")
    assert substitution.reference == "है"
    assert substitution.hypothesis == "था"


def test_diff_tokens_serialize_for_the_frontend() -> None:
    scored = comparison.score_against_reference(REFERENCE, {"full_ft": "यह घर"})
    payload = [token.as_json() for token in scored["full_ft"].diff]

    assert all(set(token) == {"kind", "reference", "hypothesis"} for token in payload)
    assert any(token["kind"] == "deletion" for token in payload)


def test_diff_counts_sum_to_the_alignment_length() -> None:
    scored = comparison.score_against_reference(REFERENCE, THREE)
    result = scored["pretrained"].scores
    counts = comparison.diff_counts(result.word_ops)
    assert sum(counts.values()) == len(result.word_ops)


# --- without a reference ----------------------------------------------------
def test_agreement_covers_every_ordered_pair() -> None:
    rows = comparison.pairwise_agreement(THREE)
    pairs = {(row["reference_model"], row["hypothesis_model"]) for row in rows}

    # 3 models -> 6 ordered pairs, self-comparisons excluded.
    assert len(rows) == 6
    assert ("full_ft", "lora") in pairs
    assert ("lora", "full_ft") in pairs
    assert not any(a == b for a, b in pairs)


def test_agreement_rows_are_labelled_not_accuracy() -> None:
    for row in comparison.pairwise_agreement(THREE):
        assert row["note"] == "agreement, not accuracy"


def test_agreement_is_asymmetric_and_both_directions_are_reported() -> None:
    transcripts = {"a": "यह बड़ा घर है", "b": "घर"}
    rows = {
        (row["reference_model"], row["hypothesis_model"]): row["wer"]
        for row in comparison.pairwise_agreement(transcripts)
    }
    assert rows[("a", "b")] != rows[("b", "a")]


def test_identical_outputs_agree_perfectly() -> None:
    same = {"a": "यह घर है", "b": "यह घर है"}
    assert all(row["wer"] == 0.0 for row in comparison.pairwise_agreement(same))


# --- summarize bundle -------------------------------------------------------
def test_summary_with_a_reference_scores_and_skips_agreement() -> None:
    body = comparison.summarize(THREE, REFERENCE)

    assert body["has_reference"] is True
    assert set(body["scores"]) == {"pretrained", "full_ft", "lora"}
    assert body["agreement"] == []
    assert body["lowest_wer_model"] == "full_ft"
    assert body["unanimous"] is False


def test_summary_without_a_reference_reports_agreement_only() -> None:
    body = comparison.summarize(THREE, None)

    assert body["has_reference"] is False
    assert body["scores"] == {}
    assert len(body["agreement"]) == 6
    assert body["lowest_wer_model"] is None


def test_summary_detects_unanimity() -> None:
    same = {"a": "यह घर है", "b": "यह घर है", "c": "यह घर है"}
    assert comparison.summarize(same, None)["unanimous"] is True
    assert comparison.summarize(THREE, None)["unanimous"] is False


def test_summary_includes_the_agreement_caveat_even_when_scored() -> None:
    body = comparison.summarize(THREE, REFERENCE)
    assert body["agreement_note"] == "agreement, not accuracy"


def test_summary_scores_carry_sdi_and_diff() -> None:
    entry = comparison.summarize(THREE, REFERENCE)["scores"]["pretrained"]

    assert entry["substitutions"] == 1
    assert entry["deletions"] == 0
    assert entry["insertions"] == 0
    assert entry["reference_words"] == 3
    assert entry["diff"]
    assert entry["diff_counts"]["substitution"] == 1


def test_summary_handles_a_single_model() -> None:
    body = comparison.summarize({"full_ft": "यह घर है"}, REFERENCE)

    assert body["models"] == ["full_ft"]
    assert body["scores"]["full_ft"]["wer"] == 0.0
    assert body["agreement"] == []


def test_lowest_wer_model_ignores_unscoreable_entries() -> None:
    """An empty reference makes every WER None; there is no winner to pick."""
    body = comparison.summarize(THREE, "")
    assert body["lowest_wer_model"] is None
