"""TranscriptionService behaviour that does not require model weights.

The weight-loading paths live in test_models_real.py. Everything here is the
surrounding contract: the model registry, the guards, and the shape of the
result object.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from app.core.config import Settings
from app.services.transcription import (
    ChunkResult,
    InferenceDisabled,
    ModelLoadError,
    TranscriptionResult,
    TranscriptionService,
    get_transcription_service,
    model_specs,
    reset_transcription_service,
)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_transcription_service()
    yield
    reset_transcription_service()


# --- the three-model registry ----------------------------------------------
def test_exactly_three_models_are_registered() -> None:
    specs = model_specs(_settings())
    assert sorted(specs) == ["full_ft", "lora", "pretrained"]


def test_pretrained_source_is_the_pinned_revision() -> None:
    spec = model_specs(_settings())["pretrained"]
    assert spec.source == (
        "openai/whisper-small@973afd24965f72e36ca33b3055d56a652f456b4d"
    )
    assert spec.kind == "pretrained"
    assert spec.seed is None


def test_fine_tuned_specs_carry_the_selected_seeds_and_rates() -> None:
    specs = model_specs(_settings())

    assert specs["full_ft"].seed == 123
    assert specs["full_ft"].learning_rate == pytest.approx(5e-6)
    assert specs["lora"].seed == 2026
    assert specs["lora"].learning_rate == pytest.approx(5e-5)


def test_display_names_state_the_selected_seed() -> None:
    """The UI shows these, and the seed choice is a research claim."""
    specs = model_specs(_settings())
    assert "seed 123" in specs["full_ft"].display_name
    assert "seed 2026" in specs["lora"].display_name
    assert "r=16" in specs["lora"].display_name


def test_changing_the_seed_setting_moves_the_checkpoint_path(tmp_path) -> None:
    """Section 3 requires the selected seed to be a config value."""
    alternative = tmp_path / "full_lr5e-06_seed42" / "best"
    specs = model_specs(_settings(full_ft_dir=alternative, full_ft_seed=42))

    assert specs["full_ft"].seed == 42
    assert "seed42" in specs["full_ft"].source


# --- guards -----------------------------------------------------------------
def test_inference_disabled_is_refused_before_any_weight_is_touched() -> None:
    service = TranscriptionService(_settings(inference_enabled=False))

    with pytest.raises(InferenceDisabled, match="INFERENCE_ENABLED=false"):
        service.get("pretrained")


def test_unknown_model_key_lists_the_valid_keys() -> None:
    service = TranscriptionService(_settings())

    with pytest.raises(ModelLoadError) as excinfo:
        service.get("whisper-large")
    message = str(excinfo.value)
    assert "whisper-large" in message
    assert "full_ft" in message


def test_nothing_is_loaded_before_first_use() -> None:
    """Lazy loading: constructing the service must not read ~3 GB from disk."""
    assert TranscriptionService(_settings()).loaded_keys() == []


def test_shared_hash_is_unset_until_preload() -> None:
    assert TranscriptionService(_settings()).shared_gen_config_hash is None


# --- process-wide singleton -------------------------------------------------
def test_service_singleton_is_reused() -> None:
    assert get_transcription_service() is get_transcription_service()


def test_reset_clears_the_singleton() -> None:
    first = get_transcription_service()
    reset_transcription_service()
    assert get_transcription_service() is not first


# --- result shape (section 4.7) ---------------------------------------------
def _result(**overrides) -> TranscriptionResult:
    base = {
        "model_key": "full_ft",
        "text_raw": "यह घर है।",
        "text_normalized": "यह घर है",
        "chunks": (
            ChunkResult(0, 0.0, 30.0, "यह घर"),
            ChunkResult(1, 30.0, 41.2345, "है"),
        ),
        "checkpoint": "E:/.../full_lr5e-06_seed123/best",
        "gen_config_hash": "a" * 64,
        "device": "cuda",
        "audio_duration_sec": 41.2345,
        "inference_ms": 8200,
        "rtf": 0.199,
    }
    base.update(overrides)
    return TranscriptionResult(**base)


def test_chunks_serialize_with_window_bounds() -> None:
    chunks = _result().chunks_as_json()

    assert [c["index"] for c in chunks] == [0, 1]
    assert chunks[0]["start_sec"] == 0.0
    assert chunks[0]["end_sec"] == 30.0
    assert chunks[1]["text"] == "है"


def test_chunk_bounds_are_rounded_for_stable_jsonb() -> None:
    assert _result().chunks_as_json()[1]["end_sec"] == 41.234


def test_result_keeps_raw_and_normalized_separately() -> None:
    result = _result()
    assert "।" in result.text_raw
    assert "।" not in result.text_normalized


# --- GPU serialization (section 4.8) ---------------------------------------
def _stub_loaded(service):
    """A LoadedModel with the weights replaced by None."""
    from app.services.transcription import LoadedModel

    return LoadedModel(
        key="pretrained",
        spec=service.specs["pretrained"],
        model=None,
        processor=None,
        generation_config=None,
        decoder_prefix=(50258, 50276, 50359, 50363),
        gen_config_hash="b" * 64,
        device="cpu",
    )


def test_transcribe_wav_serializes_inference_across_threads(monkeypatch) -> None:
    """Two threads must never be inside generate() at once.

    A 4 GB card cannot hold two concurrent FP32 decodes, so this is a
    correctness guard rather than a throughput choice. Only the weights and
    audio I/O are stubbed -- the lock under test is the real one inside
    ``transcribe_wav``.
    """
    service = TranscriptionService(_settings())
    loaded = _stub_loaded(service)
    monkeypatch.setattr(service, "get", lambda key: loaded)

    # 40 s of silence -> 2 windows, so each call takes the lock twice.
    monkeypatch.setattr(
        "app.services.transcription.load_samples",
        lambda path: ([0.0] * (16000 * 40), 16000),
    )

    concurrent = 0
    max_concurrent = 0
    counter_lock = threading.Lock()

    def fake_generate(model, samples, sample_rate):
        nonlocal concurrent, max_concurrent
        with counter_lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.02)
        with counter_lock:
            concurrent -= 1
        return "कुछ"

    monkeypatch.setattr(service, "_generate_window", fake_generate)

    results: list[object] = []
    errors: list[BaseException] = []

    def run():
        try:
            results.append(service.transcribe_wav("pretrained", Path("ignored.wav")))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert len(results) == 4
    assert max_concurrent == 1, f"{max_concurrent} concurrent decodes"


def test_transcribe_wav_records_timings_and_joins_windows(monkeypatch) -> None:
    service = TranscriptionService(_settings())
    loaded = _stub_loaded(service)
    monkeypatch.setattr(service, "get", lambda key: loaded)
    monkeypatch.setattr(
        "app.services.transcription.load_samples",
        lambda path: ([0.0] * (16000 * 40), 16000),
    )
    monkeypatch.setattr(
        service,
        "_generate_window",
        lambda model, samples, rate: "यह घर है।",
    )

    result = service.transcribe_wav("pretrained", Path("ignored.wav"))

    # 40 s -> two windows, joined with a space.
    assert len(result.chunks) == 2
    assert result.text_raw == "यह घर है। यह घर है।"
    assert result.text_normalized == "यह घर है यह घर है"
    assert result.audio_duration_sec == pytest.approx(40.0)
    assert result.inference_ms >= 0
    assert result.rtf == pytest.approx(result.inference_ms / 1000.0 / 40.0)
    assert result.gen_config_hash == "b" * 64
    assert result.device == "cpu"


def test_rtf_is_zero_for_zero_length_audio(monkeypatch) -> None:
    """Guards the division; an empty decode must not raise."""
    service = TranscriptionService(_settings())
    loaded = _stub_loaded(service)
    monkeypatch.setattr(service, "get", lambda key: loaded)
    monkeypatch.setattr(
        "app.services.transcription.load_samples", lambda path: ([], 16000)
    )
    monkeypatch.setattr(service, "_generate_window", lambda m, s, r: "")

    result = service.transcribe_wav("pretrained", Path("ignored.wav"))
    assert result.rtf == 0.0
    assert result.text_raw == ""


# --- VRAM control (eviction / sharing bookkeeping) -------------------------
def test_eviction_is_a_no_op_when_unlimited() -> None:
    service = TranscriptionService(_settings(max_resident_models=0))
    for key in ("pretrained", "full_ft", "lora"):
        service._loaded[key] = _stub_loaded(service)
        service._lru.append(key)

    with service._cache_lock:
        service._evict_if_needed(keeping="lora")

    assert sorted(service._loaded) == ["full_ft", "lora", "pretrained"]


def test_eviction_drops_least_recently_used_down_to_the_limit() -> None:
    service = TranscriptionService(_settings(max_resident_models=1))
    for key in ("pretrained", "full_ft", "lora"):
        loaded = _stub_loaded(service)
        loaded.model = object()  # distinct objects -> independent memory
        service._loaded[key] = loaded
        service._lru.append(key)

    with service._cache_lock:
        service._evict_if_needed(keeping="lora")

    assert list(service._loaded) == ["lora"]


def test_eviction_drops_keys_sharing_one_model_object_together() -> None:
    """Freeing only one of two keys on the same object frees no memory."""
    service = TranscriptionService(_settings(max_resident_models=1))
    shared = object()
    for key in ("pretrained", "lora"):
        loaded = _stub_loaded(service)
        loaded.model = shared
        service._loaded[key] = loaded
        service._lru.append(key)

    separate = _stub_loaded(service)
    separate.model = object()
    service._loaded["full_ft"] = separate
    service._lru.append("full_ft")
    service._shared_peft = shared

    with service._cache_lock:
        service._evict_if_needed(keeping="full_ft")

    assert list(service._loaded) == ["full_ft"]
    assert service._shared_peft is None, "shared handle must be released too"


def test_eviction_never_drops_the_key_being_loaded() -> None:
    service = TranscriptionService(_settings(max_resident_models=1))
    for key in ("pretrained", "full_ft"):
        loaded = _stub_loaded(service)
        loaded.model = object()
        service._loaded[key] = loaded
        service._lru.append(key)

    with service._cache_lock:
        service._evict_if_needed(keeping="pretrained")

    assert "pretrained" in service._loaded


def test_peak_vram_reports_the_expected_fields() -> None:
    stats = TranscriptionService.peak_vram()
    assert set(stats) == {"allocated_mib", "reserved_mib", "total_mib"}
    assert all(value >= 0 for value in stats.values())


def test_touch_moves_a_key_to_most_recently_used() -> None:
    service = TranscriptionService(_settings())
    with service._cache_lock:
        for key in ("pretrained", "full_ft", "lora"):
            service._touch(key)
        service._touch("pretrained")

    assert service._lru == ["full_ft", "lora", "pretrained"]


# --- a CUDA context that dies after the models are already cached -----------
#
# A sleep/resume cycle, a driver reset or a Windows TDR timeout leaves the
# process holding a context whose every kernel launch fails with a generic
# "RuntimeError: CUDA error: unknown error". The models still look resident and
# torch.cuda.is_available() still says yes, so nothing in the load path
# notices: every job then failed instantly, forever, until the server was
# restarted. These cover the detect-evict-retry-on-CPU path.

CUDA_FAULT = (
    "CUDA error: unknown error\n"
    "CUDA kernel errors might be asynchronously reported at some other API "
    "call, so the stacktrace below might be incorrect."
)


def _cuda_loaded(service, key="pretrained"):
    """A LoadedModel that claims to be resident on a GPU."""
    from app.services.transcription import LoadedModel

    return LoadedModel(
        key=key,
        spec=service.specs[key],
        model=None,
        processor=None,
        generation_config=None,
        decoder_prefix=(50258, 50276, 50359, 50363),
        gen_config_hash="c" * 64,
        device="cuda",
    )


def test_cuda_fault_is_told_apart_from_an_ordinary_runtime_error() -> None:
    from app.services.transcription import _is_cuda_fault

    assert _is_cuda_fault(RuntimeError(CUDA_FAULT))
    assert _is_cuda_fault(RuntimeError("CUDA error: out of memory"))
    assert not _is_cuda_fault(RuntimeError("ffmpeg exited with code 1"))
    assert not _is_cuda_fault(ValueError("bad sample rate"))


def test_a_dead_cuda_context_retries_on_cpu_instead_of_failing(monkeypatch) -> None:
    """The job must still finish -- the machine can do it, just slower."""
    service = TranscriptionService(_settings())
    monkeypatch.setattr(
        "app.services.transcription.load_samples",
        lambda path: ([0.0] * 16000, 16000),
    )

    # get() hands back a GPU model first; after the eviction it rebuilds on CPU.
    devices_served = []

    def fake_get(key):
        device = "cpu" if service._gpu_disabled else "cuda"
        devices_served.append(device)
        loaded = _cuda_loaded(service) if device == "cuda" else _stub_loaded(service)
        return loaded

    monkeypatch.setattr(service, "get", fake_get)

    def fake_generate(loaded, samples, sample_rate):
        if loaded.device == "cuda":
            from app.services.transcription import CudaContextLost

            raise CudaContextLost(CUDA_FAULT)
        return "ठीक"

    monkeypatch.setattr(service, "_generate_window", fake_generate)

    result = service.transcribe_wav("pretrained", Path("ignored.wav"))

    assert result.text_raw == "ठीक"
    assert result.device == "cpu", "the retry must report the device it really used"
    assert devices_served == ["cuda", "cpu"], devices_served
    assert service._gpu_disabled is True


def test_losing_the_context_drops_every_gpu_model_and_pins_to_cpu() -> None:
    """The cached objects hold CUDA storage that can no longer be touched."""
    service = TranscriptionService(_settings())
    service._loaded = {
        "pretrained": _cuda_loaded(service, "pretrained"),
        "full_ft": _cuda_loaded(service, "full_ft"),
    }
    service._lru = ["pretrained", "full_ft"]
    service._shared_peft = object()

    service._disable_gpu_and_evict()

    assert service._loaded == {}
    assert service._lru == []
    assert service._shared_peft is None
    assert service._gpu_disabled is True


def test_a_cpu_model_that_fails_is_not_mistaken_for_a_lost_gpu(monkeypatch) -> None:
    """A genuine error on CPU must surface, not trigger an endless CPU retry."""
    service = TranscriptionService(_settings())
    monkeypatch.setattr(
        "app.services.transcription.load_samples",
        lambda path: ([0.0] * 16000, 16000),
    )
    monkeypatch.setattr(service, "get", lambda key: _stub_loaded(service))

    def boom(loaded, samples, sample_rate):
        raise RuntimeError(CUDA_FAULT)

    monkeypatch.setattr(service, "_generate_window", boom)

    with pytest.raises(RuntimeError, match="CUDA error"):
        service.transcribe_wav("pretrained", Path("ignored.wav"))
    assert service._gpu_disabled is False, "a CPU failure must not disable the GPU"


def test_once_the_gpu_is_disabled_new_loads_go_straight_to_cpu(monkeypatch) -> None:
    """Even while torch still insists a GPU is there."""
    service = TranscriptionService(_settings())
    monkeypatch.setattr(
        "app.services.transcription.resolve_device", lambda config=None: "cuda"
    )

    assert service._device_for_build() == "cuda"

    service._disable_gpu_and_evict()

    assert service._device_for_build() == "cpu"
