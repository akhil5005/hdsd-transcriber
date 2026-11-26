"""Frozen decoding configuration (PROMPT.md sections 4.2 and 4.3).

The single most important rule in this file: the ``GenerationConfig`` is loaded
from the **pinned base-model snapshot**, never from a fine-tuned checkpoint
folder. The saved full-FT checkpoint's ``generation_config.json`` has an
incomplete ``suppress_tokens`` list -- it is missing the task-token ids 50358
and 50359 -- so using it would let the model emit task tokens as text and the
three models would no longer be decoding under identical rules.

Everything here is deliberately deterministic: greedy, no sampling, no
timestamps, fixed ``max_new_tokens``. :func:`config_fingerprint` hashes the
fields that actually steer decoding so the service can prove at startup that
all three models agree.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from app.core.config import Settings, settings
from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import GenerationConfig, WhisperProcessor

logger = get_logger(__name__)

LANGUAGE = "hi"
TASK = "transcribe"

#: Whisper task tokens. The full-FT checkpoint's saved suppress_tokens omits
#: these two, which is the whole reason we reload the config from the base
#: snapshot. Asserted present in :func:`build_generation_config`.
TASK_TOKEN_IDS = (50358, 50359)

#: The Hindi / transcribe / no-timestamps decoder prefix, as token *strings*.
#: Resolved through each checkpoint's own tokenizer so that a checkpoint
#: shipping a different vocab is caught -- `language`/`task` would still look
#: identical in that case, but the ids fed to the decoder would not be.
DECODER_PREFIX_TOKENS = (
    "<|startoftranscript|>",
    "<|hi|>",
    "<|transcribe|>",
    "<|notimestamps|>",
)

#: Fields that change what the model produces. Hashed for the cross-model
#: equality check; anything not listed here is presentation or bookkeeping.
FINGERPRINT_FIELDS = (
    "max_new_tokens",
    "num_beams",
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "return_timestamps",
    "forced_decoder_ids",
    "decoder_start_token_id",
    "eos_token_id",
    "pad_token_id",
    "bos_token_id",
    "begin_suppress_tokens",
    "suppress_tokens",
    "is_multilingual",
    "language",
    "task",
)


class GenerationConfigError(RuntimeError):
    """The decoding configuration is not what the frozen protocol requires."""


def build_generation_config(
    processor: WhisperProcessor,
    config: Settings | None = None,
) -> GenerationConfig:
    """Load the base-model generation config and apply the frozen settings.

    ``processor`` supplies the decoder prompt ids for Hindi/transcribe without
    timestamps, which is the supported way to pin the decoder prefix.
    """
    from transformers import GenerationConfig

    config = config or settings

    generation_config = GenerationConfig.from_pretrained(
        config.base_model_id,
        revision=config.base_model_revision,
    )

    # The reason this function exists: refuse to run if the suppress list that
    # reaches the model is missing the task tokens.
    suppress = list(generation_config.suppress_tokens or [])
    missing = [token for token in TASK_TOKEN_IDS if token not in suppress]
    if missing:
        raise GenerationConfigError(
            "The base-model generation config is missing task tokens "
            f"{missing} from suppress_tokens. Expected them to be present at "
            f"{config.base_model_id}@{config.base_model_revision}."
        )

    # --- frozen decoding settings (section 4.2) ---------------------------
    generation_config.language = LANGUAGE
    generation_config.task = TASK
    generation_config.return_timestamps = False
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.max_new_tokens = config.max_new_tokens

    # Greedy means these must not be left at sampling defaults.
    generation_config.temperature = 1.0
    generation_config.top_k = 50
    generation_config.top_p = 1.0

    # Section 4.2: clear forced_decoder_ids. It is the deprecated mechanism;
    # current transformers derives the decoder prefix from language/task/
    # return_timestamps above, and a non-None value here would override that.
    generation_config.forced_decoder_ids = None

    return generation_config, resolve_decoder_prefix(processor)


def resolve_decoder_prefix(processor: WhisperProcessor) -> tuple[int, ...]:
    """Resolve the Hindi/transcribe/no-timestamps prefix to token ids.

    This is a *verification* value, not a config field: transformers builds the
    prefix itself from ``language``/``task``/``return_timestamps``. Resolving it
    per checkpoint and folding it into the fingerprint is what proves the three
    tokenizers agree on these ids.
    """
    tokenizer = processor.tokenizer
    ids = tokenizer.convert_tokens_to_ids(list(DECODER_PREFIX_TOKENS))
    unknown = getattr(tokenizer, "unk_token_id", None)

    resolved: list[int] = []
    for token, token_id in zip(DECODER_PREFIX_TOKENS, ids, strict=True):
        if token_id is None or (unknown is not None and token_id == unknown):
            raise GenerationConfigError(
                f"Tokenizer does not know the decoder-prefix token {token!r}; "
                "this checkpoint's vocabulary is not the expected Whisper one."
            )
        resolved.append(int(token_id))
    return tuple(resolved)


def describe(
    generation_config: GenerationConfig,
    decoder_prefix: Sequence[int] | None = None,
) -> dict[str, Any]:
    """The decoding-relevant subset, in a stable order, JSON-ready."""
    described: dict[str, Any] = {}
    for field in FINGERPRINT_FIELDS:
        value = getattr(generation_config, field, None)
        if isinstance(value, tuple):
            value = list(value)
        described[field] = value
    described["decoder_prefix_ids"] = list(decoder_prefix or [])
    return described


def config_fingerprint(
    generation_config: GenerationConfig,
    decoder_prefix: Sequence[int] | None = None,
) -> str:
    """SHA-256 over the decoding-relevant fields plus the decoder prefix.

    Stored on every transcription row and logged at startup, so a change in
    decoding behaviour is visible in the data rather than inferred.
    """
    payload = json.dumps(
        describe(generation_config, decoder_prefix),
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_identical(fingerprints: dict[str, str]) -> str:
    """Require that every model resolved the same decoding config.

    Section 4.3: PEFT forwards ``generate()`` to the base model, so setting the
    config on the wrapper alone is not enough. This is the check that catches
    it. Returns the single shared fingerprint.
    """
    unique = set(fingerprints.values())
    if len(unique) != 1:
        detail = "\n".join(
            f"  {key}: {value}" for key, value in sorted(fingerprints.items())
        )
        raise GenerationConfigError(
            "Models resolved different generation configs; comparison would be "
            f"invalid:\n{detail}"
        )
    return unique.pop()
