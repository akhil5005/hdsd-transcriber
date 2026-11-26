"""Devanagari text normalization (PROMPT.md section 4.6).

One normalizer, used for both the stored `text_normalized` column and every
WER/CER computation, so a score always refers to the same text the user sees
in the normalized view.

The three steps are exactly:

1. Unicode **NFC** composition.
2. Replace Unicode **punctuation** with spaces.
3. Collapse whitespace and strip.

Deliberately NOT done: no transliteration, no case folding (Devanagari is
unicase), no digit mapping, no removal of nukta/virama/matras. Dysarthric
speech research needs the graphemes left alone -- anything that merges
distinct characters would flatter the scores.
"""

from __future__ import annotations

import re
import unicodedata

#: Devanagari danda and double danda are sentence punctuation but belong to
#: category Po, so the generic rule below already covers them. Listed here so
#: it is clear they are intentionally treated as punctuation, not as letters.
DEVANAGARI_PUNCTUATION = ("।", "॥")  # । ॥

_WHITESPACE = re.compile(r"\s+")


def _is_punctuation(char: str) -> bool:
    """True for any Unicode punctuation or symbol category.

    ``P*`` covers ASCII and Devanagari punctuation; ``S*`` catches currency
    and maths symbols that ffmpeg-era transcripts occasionally pick up.
    """
    return unicodedata.category(char).startswith(("P", "S"))


def normalize_text(text: str | None) -> str:
    """Return the comparison form of ``text``.

    Idempotent: ``normalize_text(normalize_text(x)) == normalize_text(x)``.
    """
    if not text:
        return ""

    composed = unicodedata.normalize("NFC", text)
    despunctuated = "".join(
        " " if _is_punctuation(char) else char for char in composed
    )
    return _WHITESPACE.sub(" ", despunctuated).strip()


def words(text: str) -> list[str]:
    """Whitespace-delimited word tokens of already-normalized text.

    WER counts words, so tokenization is just the split -- normalization has
    already removed the punctuation that would otherwise cling to a token.
    """
    return text.split()


def characters(text: str) -> list[str]:
    """Unicode code points of already-normalized text, **including spaces**.

    Section 5b fixes this definition. Spaces count, so word-boundary errors are
    visible in CER; and code points (not grapheme clusters) means a base
    consonant and its matra count separately, which is what the research
    protocol scored.
    """
    return list(text)
