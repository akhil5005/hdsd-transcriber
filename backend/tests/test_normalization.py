"""Normalization is the contract behind every score (PROMPT.md section 4.6)."""

from __future__ import annotations

import unicodedata

import pytest

from app.services.normalization import characters, normalize_text, words

# "क़" has two valid spellings: the precomposed U+0958, and KA + combining
# nukta (U+0915 U+093C). Devanagari nukta letters are Unicode *composition
# exclusions*, so NFC unifies them by DEcomposing U+0958 -- the direction is
# surprising, but the property that matters holds either way: two transcripts
# that look identical must score identical.
PRECOMPOSED_QA = "क़"
KA_PLUS_NUKTA = "क़"


def test_nfc_unifies_the_two_spellings_of_a_nukta_letter() -> None:
    assert PRECOMPOSED_QA != KA_PLUS_NUKTA  # guard: the fixture really differs
    assert normalize_text(PRECOMPOSED_QA) == normalize_text(KA_PLUS_NUKTA)


def test_nfc_decomposes_composition_excluded_nukta_forms() -> None:
    """Pins the direction so a normalizer swap cannot flip it silently."""
    assert normalize_text(PRECOMPOSED_QA) == KA_PLUS_NUKTA


def test_nfc_composes_where_unicode_allows_it() -> None:
    """Confirms NFC is really running and is not a no-op."""
    decomposed = unicodedata.normalize("NFD", "café")
    assert len(decomposed) == 5
    assert normalize_text(decomposed) == "café"


def test_devanagari_danda_becomes_a_space() -> None:
    assert normalize_text("यह घर है। वह नहीं॥") == "यह घर है वह नहीं"


def test_ascii_punctuation_becomes_a_space() -> None:
    assert normalize_text("हाँ, ठीक है!") == "हाँ ठीक है"


def test_punctuation_separates_rather_than_joins_tokens() -> None:
    """A dash must not weld two words into one -- that would hide a WER error."""
    assert words(normalize_text("राम-श्याम")) == ["राम", "श्याम"]


def test_whitespace_is_collapsed_and_stripped() -> None:
    assert normalize_text("  यह   \t घर \n है  ") == "यह घर है"


def test_devanagari_characters_are_preserved() -> None:
    """No transliteration, no stripping of matras or virama."""
    text = "क्ष त्र ज्ञ श्री"
    assert normalize_text(text) == text


def test_digits_are_not_remapped() -> None:
    assert normalize_text("४२ और 42") == "४२ और 42"


def test_normalization_is_idempotent() -> None:
    once = normalize_text("  यह, घर   है।  ")
    assert normalize_text(once) == once


@pytest.mark.parametrize("value", [None, "", "   ", "।", ".,!?"])
def test_empty_and_punctuation_only_input_yields_empty_string(value) -> None:
    assert normalize_text(value) == ""


def test_words_splits_on_whitespace() -> None:
    assert words(normalize_text("यह घर है")) == ["यह", "घर", "है"]


def test_characters_includes_spaces_per_section_5b() -> None:
    chars = characters(normalize_text("यह घर"))
    assert " " in chars
    assert len(chars) == len("यह घर")


def test_characters_counts_code_points_not_graphemes() -> None:
    """'कि' is one grapheme cluster but two code points; CER counts two."""
    assert len(characters("कि")) == 2
