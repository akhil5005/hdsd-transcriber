"""Generation-config fingerprinting and the cross-model equality check.

These exercise the pure logic in app/services/generation.py with a stand-in
object, so they run without torch/transformers. The part that genuinely needs
the Hub (loading the pinned base config and asserting the task tokens are in
`suppress_tokens`) lives in test_models_real.py and is skipped when the ML
stack is absent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.generation import (
    DECODER_PREFIX_TOKENS,
    FINGERPRINT_FIELDS,
    TASK_TOKEN_IDS,
    GenerationConfigError,
    assert_identical,
    config_fingerprint,
    describe,
    resolve_decoder_prefix,
)

#: Stand-in for the ids the real tokenizer resolves for DECODER_PREFIX_TOKENS.
PREFIX = (50258, 50276, 50359, 50363)


def fake_config(**overrides):
    """A GenerationConfig-shaped object with frozen-protocol values."""
    base = {
        "max_new_tokens": 96,
        "num_beams": 1,
        "do_sample": False,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "return_timestamps": False,
        "forced_decoder_ids": None,
        "decoder_start_token_id": 50258,
        "eos_token_id": 50257,
        "pad_token_id": 50257,
        "bos_token_id": 50257,
        "begin_suppress_tokens": [220, 50257],
        "suppress_tokens": [1, 2, *TASK_TOKEN_IDS],
        "is_multilingual": True,
        "language": "hi",
        "task": "transcribe",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# --- describe ---------------------------------------------------------------
def test_describe_covers_every_fingerprint_field() -> None:
    described = describe(fake_config(), PREFIX)
    for field in FINGERPRINT_FIELDS:
        assert field in described
    assert described["decoder_prefix_ids"] == list(PREFIX)


def test_prefix_tokens_are_the_hindi_transcribe_no_timestamps_set() -> None:
    assert DECODER_PREFIX_TOKENS == (
        "<|startoftranscript|>",
        "<|hi|>",
        "<|transcribe|>",
        "<|notimestamps|>",
    )


def test_describe_normalizes_tuples_to_lists() -> None:
    """So a tuple vs list difference does not read as a config change."""
    described = describe(fake_config(suppress_tokens=(1, 2, *TASK_TOKEN_IDS)), PREFIX)
    assert described["suppress_tokens"] == [1, 2, *TASK_TOKEN_IDS]


def test_describe_tolerates_missing_attributes() -> None:
    minimal = SimpleNamespace(max_new_tokens=96)
    described = describe(minimal, PREFIX)
    assert described["max_new_tokens"] == 96
    assert described["num_beams"] is None


# --- fingerprint ------------------------------------------------------------
def test_fingerprint_is_a_sha256_hex_digest() -> None:
    digest = config_fingerprint(fake_config(), PREFIX)
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_fingerprint_is_stable_across_calls() -> None:
    assert config_fingerprint(fake_config(), PREFIX) == config_fingerprint(
        fake_config(), PREFIX
    )


def test_fingerprint_ignores_attribute_definition_order() -> None:
    """Keys are sorted, so two equal configs hash alike regardless of order."""
    first = fake_config()
    reordered = SimpleNamespace(**dict(reversed(list(vars(first).items()))))
    assert config_fingerprint(first, PREFIX) == config_fingerprint(reordered, PREFIX)


@pytest.mark.parametrize(
    "override",
    [
        {"max_new_tokens": 128},
        {"num_beams": 5},
        {"do_sample": True},
        {"return_timestamps": True},
        {"language": "en"},
        {"task": "translate"},
        {"temperature": 0.7},
        {"suppress_tokens": [1, 2]},
        {"forced_decoder_ids": [[1, 50259]]},
    ],
)
def test_any_decoding_change_changes_the_fingerprint(override) -> None:
    assert config_fingerprint(fake_config(**override), PREFIX) != config_fingerprint(
        fake_config(), PREFIX
    )


def test_dropping_the_task_tokens_changes_the_fingerprint() -> None:
    """The exact full-FT checkpoint defect this whole mechanism guards against."""
    defective = fake_config(suppress_tokens=[1, 2])
    assert config_fingerprint(defective, PREFIX) != config_fingerprint(fake_config(), PREFIX)


# --- cross-model equality (section 4.3) -------------------------------------
def test_assert_identical_returns_the_shared_hash() -> None:
    digest = config_fingerprint(fake_config(), PREFIX)
    shared = assert_identical(
        {"pretrained": digest, "full_ft": digest, "lora": digest}
    )
    assert shared == digest


def test_assert_identical_rejects_a_divergent_model() -> None:
    good = config_fingerprint(fake_config(), PREFIX)
    bad = config_fingerprint(fake_config(suppress_tokens=[1, 2]), PREFIX)

    with pytest.raises(GenerationConfigError) as excinfo:
        assert_identical({"pretrained": good, "full_ft": bad, "lora": good})

    message = str(excinfo.value)
    assert "full_ft" in message
    assert "comparison would be" in message


def test_assert_identical_accepts_a_single_model() -> None:
    digest = config_fingerprint(fake_config(), PREFIX)
    assert assert_identical({"pretrained": digest}) == digest


# --- decoder prefix resolution ---------------------------------------------
class _StubTokenizer:
    def __init__(self, mapping, unk_token_id=None):
        self._mapping = mapping
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, tokens):
        return [self._mapping.get(token, self.unk_token_id) for token in tokens]


class _StubProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def test_resolve_decoder_prefix_returns_ids_in_prefix_order() -> None:
    mapping = dict(zip(DECODER_PREFIX_TOKENS, PREFIX, strict=True))
    processor = _StubProcessor(_StubTokenizer(mapping))

    assert resolve_decoder_prefix(processor) == PREFIX


def test_resolve_decoder_prefix_rejects_an_unknown_prefix_token() -> None:
    """A checkpoint with a different vocabulary must not pass silently."""
    mapping = dict(zip(DECODER_PREFIX_TOKENS, PREFIX, strict=True))
    del mapping["<|hi|>"]
    processor = _StubProcessor(_StubTokenizer(mapping, unk_token_id=50257))

    with pytest.raises(GenerationConfigError, match=r"<\|hi\|>"):
        resolve_decoder_prefix(processor)


def test_a_different_decoder_prefix_changes_the_fingerprint() -> None:
    """Identical language/task but a divergent tokenizer is still caught."""
    other = (50258, 50259, 50359, 50363)  # <|en|> instead of <|hi|>
    assert config_fingerprint(fake_config(), PREFIX) != config_fingerprint(
        fake_config(), other
    )
