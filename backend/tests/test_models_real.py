"""Integration tests against the real checkpoints and the pinned base model.

These are the tests that can actually catch the defect PROMPT.md section 4.2
warns about, so they load real weights. They are slow and are skipped unless
the ML stack is installed, the research checkpoints are present, and
HDSD_RUN_MODEL_TESTS=1 is set:

    HDSD_RUN_MODEL_TESTS=1 python -m pytest tests/test_models_real.py -v

Nothing here writes to the research archive.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.research_assets import ResearchAssetError, verify_checkpoints
from app.models.enums import ModelKey
from app.services import generation

pytestmark = pytest.mark.slow

ENV_FLAG = "HDSD_RUN_MODEL_TESTS"


def _requirements_met() -> tuple[bool, str]:
    if os.environ.get(ENV_FLAG) != "1":
        return False, f"set {ENV_FLAG}=1 to run model integration tests"
    for module in ("torch", "transformers", "peft"):
        try:
            __import__(module)
        except ImportError:
            return False, f"{module} is not installed"
    try:
        verify_checkpoints(Settings())
    except ResearchAssetError as exc:
        return False, f"checkpoints unavailable: {exc}"
    return True, ""


_ok, _reason = _requirements_met()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not _ok, reason=_reason)]


@pytest.fixture(scope="module")
def config() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def service(config):
    from app.services.transcription import TranscriptionService

    return TranscriptionService(config)


# --- the section 4.2 defect -------------------------------------------------
def test_saved_full_ft_config_really_is_missing_the_task_tokens(config) -> None:
    """Documents the defect, so the workaround is never 'cleaned up'.

    If this ever fails, the checkpoint on disk changed -- investigate before
    touching app/services/generation.py.
    """
    import json

    saved = json.loads(
        (config.full_ft_dir / "generation_config.json").read_text(encoding="utf-8")
    )
    suppress = saved.get("suppress_tokens") or []
    missing = [t for t in generation.TASK_TOKEN_IDS if t not in suppress]

    assert missing == list(generation.TASK_TOKEN_IDS), (
        "The saved full-FT config now contains the task tokens; "
        "the base-snapshot reload may no longer be necessary."
    )


def test_base_snapshot_config_contains_the_task_tokens(config) -> None:
    from transformers import GenerationConfig

    base = GenerationConfig.from_pretrained(
        config.base_model_id, revision=config.base_model_revision
    )
    suppress = list(base.suppress_tokens or [])
    for token in generation.TASK_TOKEN_IDS:
        assert token in suppress


def test_every_checkpoint_resolves_the_same_decoder_prefix(service) -> None:
    """Catches a checkpoint shipping a different tokenizer vocabulary."""
    prefixes = {
        key: service.get(key).decoder_prefix
        for key in ("pretrained", "full_ft", "lora")
    }
    assert len(set(prefixes.values())) == 1, prefixes
    assert len(next(iter(prefixes.values()))) == len(
        generation.DECODER_PREFIX_TOKENS
    )


def test_build_generation_config_applies_the_frozen_protocol(config, service) -> None:
    loaded = service.get(ModelKey.PRETRAINED.value)
    gc = loaded.generation_config

    assert gc.language == "hi"
    assert gc.task == "transcribe"
    assert gc.return_timestamps is False
    assert gc.do_sample is False
    assert gc.num_beams == 1
    assert gc.max_new_tokens == config.max_new_tokens == 96
    assert gc.forced_decoder_ids is None
    for token in generation.TASK_TOKEN_IDS:
        assert token in list(gc.suppress_tokens)


# --- loading and the cross-model guarantee ----------------------------------
def test_all_three_models_load_and_share_one_generation_config(service) -> None:
    shared = service.preload_all()

    assert len(shared) == 64
    assert service.loaded_keys() == ["full_ft", "lora", "pretrained"]
    for key in ("pretrained", "full_ft", "lora"):
        assert service.get(key).gen_config_hash == shared


def test_lora_wrapper_and_its_base_resolve_the_same_config(service) -> None:
    """Section 4.3: PEFT forwards generate() to the base model."""
    loaded = service.get(ModelKey.LORA.value)
    prefix = loaded.decoder_prefix
    wrapper_hash = generation.config_fingerprint(
        loaded.model.generation_config, prefix
    )

    inner = loaded.model.base_model
    inner = getattr(inner, "model", inner)
    inner_hash = generation.config_fingerprint(inner.generation_config, prefix)

    assert wrapper_hash == inner_hash == loaded.gen_config_hash


def test_lora_is_loaded_as_an_adapter_not_a_standalone_model(service) -> None:
    from peft import PeftModel

    assert isinstance(service.get(ModelKey.LORA.value).model, PeftModel)


def test_models_are_fp32_and_in_eval_mode(service) -> None:
    import torch

    for key in ("pretrained", "full_ft", "lora"):
        model = service.get(key).model
        assert not model.training, f"{key} is not in eval() mode"
        assert next(model.parameters()).dtype is torch.float32, f"{key} is not FP32"


def test_tf32_flags_match_section_4_4(service) -> None:
    import torch

    service.get(ModelKey.PRETRAINED.value)  # loading applies the flags
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is True


def test_models_are_cached_not_reloaded(service) -> None:
    first = service.get(ModelKey.FULL_FT.value)
    second = service.get(ModelKey.FULL_FT.value)
    assert first is second
    assert first.model is second.model


def test_full_ft_and_pretrained_have_different_weights(service) -> None:
    """Proves the fine-tuned checkpoint actually loaded, not the base twice."""
    import torch

    pretrained = service.get(ModelKey.PRETRAINED.value).model
    full_ft = service.get(ModelKey.FULL_FT.value).model

    a = pretrained.get_decoder().layers[0].self_attn.q_proj.weight
    b = full_ft.get_decoder().layers[0].self_attn.q_proj.weight
    assert not torch.allclose(a.cpu(), b.cpu())


# --- end-to-end transcription ----------------------------------------------
@pytest.fixture(scope="module")
def tone_wav(config, tmp_path_factory) -> Path:
    """A synthetic 2 s tone.

    The research archive deliberately excludes the original patient audio, so
    a real dysarthric sample is not available to the test suite. A tone cannot
    validate transcription *quality* -- only that the pipeline runs, the
    plumbing is wired, and the timings are recorded.
    """
    import subprocess

    out = tmp_path_factory.mktemp("audio") / "tone.wav"
    result = subprocess.run(
        [
            config.ffmpeg_binary, "-nostdin", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg unavailable: {result.stderr[-300:]}")
    return out


def test_transcribe_records_everything_section_4_7_requires(service, tone_wav) -> None:
    result = service.transcribe_wav(ModelKey.PRETRAINED.value, tone_wav)

    assert result.model_key == "pretrained"
    assert result.checkpoint  # revision or path
    assert len(result.gen_config_hash) == 64
    assert result.device in ("cpu", "cuda")
    assert result.audio_duration_sec == pytest.approx(2.0, abs=0.1)
    assert result.inference_ms > 0
    assert result.rtf > 0
    assert result.text_normalized == result.text_normalized.strip()
    assert isinstance(result.chunks_as_json(), list)


def test_short_audio_produces_exactly_one_window(service, tone_wav) -> None:
    result = service.transcribe_wav(ModelKey.PRETRAINED.value, tone_wav)
    assert len(result.chunks) == 1
    assert result.chunks[0].index == 0


def test_greedy_decoding_is_reproducible(service, tone_wav) -> None:
    """num_beams=1 and do_sample=False must make runs byte-identical."""
    first = service.transcribe_wav(ModelKey.PRETRAINED.value, tone_wav)
    second = service.transcribe_wav(ModelKey.PRETRAINED.value, tone_wav)
    assert first.text_raw == second.text_raw


def test_all_three_models_transcribe_the_same_file(service, tone_wav) -> None:
    hashes = set()
    for key in ("pretrained", "full_ft", "lora"):
        result = service.transcribe_wav(key, tone_wav)
        hashes.add(result.gen_config_hash)
        assert isinstance(result.text_raw, str)

    assert len(hashes) == 1, "models decoded under different configs"


# --- base sharing must not change outputs (VRAM optimisation) --------------
def test_shared_base_pretrained_matches_an_independent_base(config, tone_wav) -> None:
    """`pretrained` via disable_adapter() must equal a standalone base model.

    LoRA adds BA.x at forward time and never mutates base weights, so this
    should hold exactly. It is the correctness condition for the ~973 MB VRAM
    saving, so it is asserted on real weights rather than argued.
    """
    from app.services.transcription import TranscriptionService

    shared = TranscriptionService(config.model_copy(update={"share_base_model": True}))
    separate = TranscriptionService(
        config.model_copy(update={"share_base_model": False})
    )

    shared_loaded = shared.get(ModelKey.PRETRAINED.value)
    assert shared_loaded.disable_adapter is True

    separate_loaded = separate.get(ModelKey.PRETRAINED.value)
    assert separate_loaded.disable_adapter is False

    from_shared = shared.transcribe_wav(ModelKey.PRETRAINED.value, tone_wav)
    from_separate = separate.transcribe_wav(ModelKey.PRETRAINED.value, tone_wav)

    assert from_shared.text_raw == from_separate.text_raw
    assert from_shared.gen_config_hash == from_separate.gen_config_hash


def test_sharing_reuses_one_object_for_pretrained_and_lora(config) -> None:
    from app.services.transcription import TranscriptionService

    service = TranscriptionService(
        config.model_copy(update={"share_base_model": True})
    )
    pretrained = service.get(ModelKey.PRETRAINED.value)
    lora = service.get(ModelKey.LORA.value)

    assert pretrained.model is lora.model
    # Same weights, opposite adapter state.
    assert pretrained.disable_adapter is True
    assert lora.disable_adapter is False


def test_lora_adapter_actually_changes_the_output(config, tone_wav) -> None:
    """Guards the sharing trick: if disable_adapter() silently did nothing,
    `pretrained` and `lora` would be identical and the comparison worthless."""
    from app.services.transcription import TranscriptionService

    service = TranscriptionService(
        config.model_copy(update={"share_base_model": True})
    )
    import torch

    loaded_lora = service.get(ModelKey.LORA.value)
    features = loaded_lora.processor.feature_extractor(
        [0.0] * 16000, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(loaded_lora.device, dtype=torch.float32)

    with torch.inference_mode():
        with_adapter = loaded_lora.model(
            input_features=features,
            decoder_input_ids=torch.tensor([[50258]], device=loaded_lora.device),
        ).logits
        with loaded_lora.model.disable_adapter():
            without_adapter = loaded_lora.model(
                input_features=features,
                decoder_input_ids=torch.tensor([[50258]], device=loaded_lora.device),
            ).logits

    assert not torch.allclose(with_adapter, without_adapter), (
        "disable_adapter() had no effect; LoRA weights are not being applied"
    )


# --- attention mask (suppresses the spurious transformers warning) ---------
def test_passing_an_attention_mask_does_not_change_token_ids(service, config) -> None:
    """Whisper's encoder documents that it ignores this mask; prove it."""
    import torch

    loaded = service.get(ModelKey.PRETRAINED.value)
    extracted = loaded.processor.feature_extractor(
        [0.0] * 32000,
        sampling_rate=config.target_sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    features = extracted.input_features.to(loaded.device, dtype=torch.float32)
    mask = extracted.attention_mask.to(loaded.device)

    with torch.inference_mode():
        without = loaded.model.generate(
            input_features=features, generation_config=loaded.generation_config
        )
        with_mask = loaded.model.generate(
            input_features=features,
            attention_mask=mask,
            generation_config=loaded.generation_config,
        )

    assert torch.equal(without.cpu(), with_mask.cpu())


def test_generate_emits_no_attention_mask_warning(service, tone_wav, caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="transformers.generation.utils"):
        service.transcribe_wav(ModelKey.PRETRAINED.value, tone_wav)

    offenders = [
        record.getMessage()
        for record in caplog.records
        if "attention mask is not set" in record.getMessage()
    ]
    assert offenders == []
