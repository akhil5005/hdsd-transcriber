"""Ground truth recognised from an HDSD filename.

Every speaker reads the same 30 sentences and the corpus encodes which one in
the filename, so a dataset upload can be scored without anyone transcribing
it. These cover the parsing, which is the part that silently mis-scores a job
if it matches the wrong sentence.
"""

from __future__ import annotations

import pytest

from app.services.prompts import prompt_id_from_filename


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("M22_S1_H15_M1.wav", "H15"),
        ("CF00_S1_H01_M2.wav", "H01"),
        ("F07_S1_H30_M1.wav", "H30"),
        # Mic suffix is optional; the label files have none.
        ("M22_S1_H15.txt", "H15"),
        # Case from a renamed download.
        ("m22_s1_h15_m1.wav", "H15"),
    ],
)
def test_sentence_id_is_read_from_corpus_filenames(filename, expected) -> None:
    assert prompt_id_from_filename(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "recording.webm",
        "my notes H15.wav",  # space-delimited, not an id segment
        "clinic-visit-2026.wav",
        "H1.wav",  # ids are two digits
        "patient_H155_x.wav",  # three digits is not a sentence id
        "",
        None,
    ],
)
def test_non_corpus_filenames_match_nothing(filename) -> None:
    assert prompt_id_from_filename(filename) is None


def test_a_browser_recording_never_matches_a_prompt() -> None:
    """Recordings are new speech, so they must fall back to a typed reference."""
    assert prompt_id_from_filename("recording-2026-11-27T10-15-00.webm") is None
