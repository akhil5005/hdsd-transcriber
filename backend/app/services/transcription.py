"""TranscriptionService -- loads the three models once and transcribes with them.

Design points fixed by PROMPT.md section 4:

* Each model is loaded **lazily on first use and then cached** (4.0).
* The generation config comes from the pinned base snapshot and is passed
  **explicitly to every `generate()` call**, because PEFT forwards `generate()`
  to the wrapped base model and would otherwise use the base model's own
  config (4.3).
* FP32, no autocast, `eval()`, `inference_mode()`, TF32 off for matmul and on
  for cuDNN (4.4).
* Audio longer than `CHUNK_SECONDS` is split deterministically; per-window
  text is retained (4.5).
* **One inference at a time**, behind a lock, so a 4 GB GPU is never asked to
  run two models concurrently (4.8).

The service is sync and blocking by design. Callers run it off the event loop
(`asyncio.to_thread` in Phase 3); keeping torch calls out of async code avoids
pinning the loop during a multi-second decode.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import Settings, settings
from app.core.logging import get_logger
from app.core.research_assets import verify_checkpoints
from app.models.enums import ModelKey
from app.services import generation
from app.services.audio import AudioInfo, load_samples, split_windows, to_wav16k_mono
from app.services.normalization import normalize_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import WhisperProcessor

logger = get_logger(__name__)


class ModelLoadError(RuntimeError):
    """A model could not be loaded."""


class InferenceDisabled(RuntimeError):
    """Transcription was requested on a host configured without weights."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static description of one of the three models."""

    key: str
    display_name: str
    kind: str
    source: str
    learning_rate: float | None = None
    seed: int | None = None


@dataclass(slots=True)
class LoadedModel:
    key: str
    spec: ModelSpec
    model: Any
    processor: WhisperProcessor
    generation_config: Any
    #: Resolved Hindi/transcribe/no-timestamps prefix ids, folded into the hash.
    decoder_prefix: tuple[int, ...]
    gen_config_hash: str
    device: str
    #: True when ``model`` is a shared PeftModel that must have its adapter
    #: switched OFF for this key (how `pretrained` reuses the LoRA base).
    disable_adapter: bool = False


@dataclass(frozen=True, slots=True)
class ChunkResult:
    index: int
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Everything section 4.7 requires us to record for one run."""

    model_key: str
    text_raw: str
    text_normalized: str
    chunks: tuple[ChunkResult, ...]
    checkpoint: str
    gen_config_hash: str
    device: str
    audio_duration_sec: float
    inference_ms: int
    rtf: float

    def chunks_as_json(self) -> list[dict[str, Any]]:
        return [
            {
                "index": chunk.index,
                "start_sec": round(chunk.start_sec, 3),
                "end_sec": round(chunk.end_sec, 3),
                "text": chunk.text,
            }
            for chunk in self.chunks
        ]


def model_specs(config: Settings | None = None) -> dict[str, ModelSpec]:
    """The three models, with the selected seeds from the research."""
    config = config or settings
    return {
        ModelKey.PRETRAINED.value: ModelSpec(
            key=ModelKey.PRETRAINED.value,
            display_name="Whisper-small (pretrained, OpenAI)",
            kind="pretrained",
            source=f"{config.base_model_id}@{config.base_model_revision}",
        ),
        ModelKey.FULL_FT.value: ModelSpec(
            key=ModelKey.FULL_FT.value,
            display_name=f"Full fine-tune (lr {config.full_ft_learning_rate:g}, "
            f"seed {config.full_ft_seed})",
            kind="full",
            source=str(config.full_ft_dir),
            learning_rate=config.full_ft_learning_rate,
            seed=config.full_ft_seed,
        ),
        ModelKey.LORA.value: ModelSpec(
            key=ModelKey.LORA.value,
            display_name=f"LoRA adapter (lr {config.lora_learning_rate:g}, "
            f"seed {config.lora_seed}, r=16, alpha=32, q/v)",
            kind="lora",
            source=str(config.lora_dir),
            learning_rate=config.lora_learning_rate,
            seed=config.lora_seed,
        ),
    }


def resolve_device(config: Settings | None = None) -> str:
    config = config or settings
    if config.device != "auto":
        return config.device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _apply_torch_determinism() -> None:
    """Section 4.4: FP32 everywhere, TF32 off for matmul, on for cuDNN."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


class TranscriptionService:
    """Lazy, cached, serialized access to the three Whisper variants."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or settings
        self._specs = model_specs(self._config)
        self._loaded: dict[str, LoadedModel] = {}
        #: One PeftModel instance serving both `lora` (adapter on) and
        #: `pretrained` (adapter off) when SHARE_BASE_MODEL is set.
        self._shared_peft: Any | None = None
        #: Cache keys in least-recently-used order, for MAX_RESIDENT_MODELS.
        self._lru: list[str] = []
        # Guards the cache dict; cheap and only held around bookkeeping.
        self._cache_lock = threading.Lock()
        # Guards the accelerator: exactly one generate() at a time.
        self._inference_lock = threading.Lock()
        self._shared_hash: str | None = None

    # --- introspection ----------------------------------------------------
    @property
    def specs(self) -> dict[str, ModelSpec]:
        return self._specs

    def loaded_keys(self) -> list[str]:
        with self._cache_lock:
            return sorted(self._loaded)

    @property
    def shared_gen_config_hash(self) -> str | None:
        """Set once all three models have been loaded and compared."""
        return self._shared_hash

    # --- loading ----------------------------------------------------------
    def _require_enabled(self) -> None:
        if not self._config.inference_enabled:
            raise InferenceDisabled(
                "Transcription is disabled on this host (INFERENCE_ENABLED=false). "
                "Run the backend on a machine that has the model weights."
            )

    def _load_processor(self, source: Path | str) -> WhisperProcessor:
        """Load the processor, preferring the checkpoint's own tokenizer files.

        The fine-tuned folders ship the full tokenizer/feature-extractor set,
        so they are self-describing; the pretrained key falls back to the
        pinned Hub revision.
        """
        from transformers import WhisperProcessor

        if isinstance(source, Path):
            return WhisperProcessor.from_pretrained(str(source))
        return WhisperProcessor.from_pretrained(
            self._config.base_model_id, revision=self._config.base_model_revision
        )

    def _load_base_model(self):
        from transformers import WhisperForConditionalGeneration

        return WhisperForConditionalGeneration.from_pretrained(
            self._config.base_model_id,
            revision=self._config.base_model_revision,
        )

    def _shared_peft_model(self):
        """Load the pinned base wrapped in the LoRA adapter, once.

        Serves both `lora` (adapter active) and `pretrained` (adapter disabled
        at generate time). Called with the cache lock NOT held; the caller
        resolves races by keeping whichever instance lands first.
        """
        if self._shared_peft is not None:
            return self._shared_peft

        from peft import PeftModel

        wrapped = PeftModel.from_pretrained(
            self._load_base_model(), str(self._config.lora_dir)
        )
        self._shared_peft = wrapped
        logger.info(
            "Loaded one shared base+adapter instance for pretrained and lora "
            "(SHARE_BASE_MODEL=true)"
        )
        return wrapped

    def _evict_if_needed(self, keeping: str) -> None:
        """Honour MAX_RESIDENT_MODELS by dropping least-recently-used models.

        Called with the cache lock held. Keys whose model object is shared are
        dropped together, since freeing one would not free any memory.
        """
        limit = self._config.max_resident_models
        if limit <= 0:
            return

        while len(self._loaded) > limit:
            victim = next((k for k in self._lru if k != keeping), None)
            if victim is None:
                return

            model = self._loaded[victim].model
            # Drop every key pointing at the same object, or the memory stays.
            siblings = [k for k, v in self._loaded.items() if v.model is model]
            for key in siblings:
                self._loaded.pop(key, None)
                if key in self._lru:
                    self._lru.remove(key)
            if model is self._shared_peft:
                self._shared_peft = None
            logger.info("Evicted model(s) %s to respect MAX_RESIDENT_MODELS", siblings)

        self._release_accelerator_memory()

    @staticmethod
    def _release_accelerator_memory() -> None:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- VRAM instrumentation --------------------------------------------
    @staticmethod
    def reset_peak_vram() -> None:
        """Zero the CUDA peak-memory counters before a measured run."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def peak_vram() -> dict[str, float]:
        """Peak CUDA memory since the last reset, in MiB."""
        import torch

        if not torch.cuda.is_available():
            return {"allocated_mib": 0.0, "reserved_mib": 0.0, "total_mib": 0.0}
        mib = 1024 * 1024
        return {
            "allocated_mib": torch.cuda.max_memory_allocated() / mib,
            "reserved_mib": torch.cuda.max_memory_reserved() / mib,
            "total_mib": torch.cuda.get_device_properties(0).total_memory / mib,
        }

    def _build(self, key: str) -> LoadedModel:
        import torch
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration

        spec = self._specs[key]
        device = resolve_device(self._config)
        _apply_torch_determinism()

        logger.info("Loading model %s from %s", key, spec.source)
        started = time.perf_counter()

        share = self._config.share_base_model
        disable_adapter = False

        if key == ModelKey.PRETRAINED.value:
            processor = self._load_processor("hub")
            if share:
                # Reuse the LoRA wrapper's base weights with the adapter
                # switched off. LoRA never mutates base weights -- it adds
                # BA.x at forward time -- so disable_adapter() is exactly the
                # pretrained model, and we save ~973 MB of VRAM.
                model = self._shared_peft_model()
                disable_adapter = True
            else:
                model = self._load_base_model()
        elif key == ModelKey.FULL_FT.value:
            directory = self._config.full_ft_dir
            processor = self._load_processor(directory)
            model = WhisperForConditionalGeneration.from_pretrained(str(directory))
        elif key == ModelKey.LORA.value:
            directory = self._config.lora_dir
            processor = self._load_processor(directory)
            model = self._shared_peft_model() if share else PeftModel.from_pretrained(
                # The adapter is not a standalone model: the pinned base must
                # be loaded first and then wrapped.
                self._load_base_model(),
                str(directory),
            )
        else:
            raise ModelLoadError(f"Unknown model key {key!r}")

        # Section 4.2/4.3: the config always comes from the base snapshot, and
        # is set on BOTH the wrapper and the wrapped base model so that
        # whichever object resolves generate() sees the same rules.
        generation_config, decoder_prefix = generation.build_generation_config(
            processor, self._config
        )
        model.generation_config = generation_config
        if isinstance(model, PeftModel):
            # get_base_model() is the documented way to reach the wrapped
            # model. Every HF model also exposes a `base_model` property, but
            # for Whisper that resolves to the inner WhisperModel, which does
            # not take part in generate() -- setting it there would do nothing.
            model.get_base_model().generation_config = generation_config

        model = model.to(dtype=torch.float32)
        try:
            model = model.to(device)
        except RuntimeError as exc:
            # torch.cuda.OutOfMemoryError subclasses RuntimeError, so this one
            # clause covers OOM as well as "no CUDA device".
            # A 4 GB card cannot always hold three FP32 copies of whisper-small
            # (~973 MB each). Degrade to CPU for this model rather than failing
            # the whole comparison.
            logger.warning(
                "Could not place %s on %s (%s); falling back to CPU", key, device, exc
            )
            device = "cpu"
            model = model.to(device)

        model.eval()

        gen_hash = generation.config_fingerprint(generation_config, decoder_prefix)
        logger.info(
            "Loaded %s on %s in %.1fs (gen_config %s)",
            key,
            device,
            time.perf_counter() - started,
            gen_hash[:12],
        )

        return LoadedModel(
            key=key,
            spec=spec,
            model=model,
            processor=processor,
            generation_config=generation_config,
            decoder_prefix=decoder_prefix,
            gen_config_hash=gen_hash,
            device=device,
            disable_adapter=disable_adapter,
        )

    def get(self, key: str) -> LoadedModel:
        """Return the cached model for ``key``, loading it on first use."""
        self._require_enabled()
        if key not in self._specs:
            raise ModelLoadError(
                f"Unknown model key {key!r}; expected one of {sorted(self._specs)}"
            )

        with self._cache_lock:
            cached = self._loaded.get(key)
            if cached is not None:
                self._touch(key)
                return cached

        built = self._build(key)
        with self._cache_lock:
            # Another thread may have won the race; keep whichever landed first
            # so there is never more than one copy of a model in memory.
            existing = self._loaded.get(key)
            if existing is not None:
                self._touch(key)
                return existing
            self._loaded[key] = built
            self._touch(key)
            self._evict_if_needed(keeping=key)
        return built

    def _touch(self, key: str) -> None:
        """Mark ``key`` most-recently-used. Cache lock must be held."""
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)

    def preload_all(self) -> str:
        """Load all three models and assert their configs match (section 4.3).

        Returns the shared generation-config hash and logs it.
        """
        self._require_enabled()
        verify_checkpoints(self._config)

        fingerprints = {key: self.get(key).gen_config_hash for key in self._specs}
        shared = generation.assert_identical(fingerprints)
        self._shared_hash = shared
        logger.info("Generation config identical across models; SHA-256 %s", shared)
        return shared

    # --- inference --------------------------------------------------------
    def _generate_window(self, loaded: LoadedModel, samples, sample_rate: int) -> str:
        import torch

        # return_attention_mask: Whisper's own docs say the encoder ignores
        # this mask ("Whisper does not support masking of the input_features,
        # this argument is preserved for compatibility, but it is not used"),
        # but passing it suppresses the spurious "attention mask is not set ...
        # pad token is same as eos token" warning that transformers emits for
        # every call. Verified byte-identical token ids with and without it.
        extracted = loaded.processor.feature_extractor(
            samples,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = extracted.input_features.to(
            device=loaded.device, dtype=torch.float32
        )
        attention_mask = extracted.attention_mask.to(device=loaded.device)

        def _decode():
            with torch.inference_mode():
                return loaded.model.generate(
                    input_features=features,
                    attention_mask=attention_mask,
                    # Explicit on every call -- PEFT forwards generate() to the
                    # base model, so relying on the wrapper's attribute is
                    # unsafe.
                    generation_config=loaded.generation_config,
                )

        if loaded.disable_adapter:
            # `pretrained` sharing the LoRA wrapper's base weights: switch the
            # adapter off so the forward pass is the untouched base model.
            with loaded.model.disable_adapter():
                token_ids = _decode()
        else:
            token_ids = _decode()

        return loaded.processor.batch_decode(token_ids, skip_special_tokens=True)[0]

    def transcribe_wav(self, model_key: str, wav_path: Path) -> TranscriptionResult:
        """Transcribe an already-converted 16 kHz mono WAV."""
        loaded = self.get(model_key)
        samples, sample_rate = load_samples(wav_path)
        duration = len(samples) / sample_rate if sample_rate else 0.0
        windows = split_windows(samples, sample_rate, self._config)

        chunks: list[ChunkResult] = []
        started = time.perf_counter()
        # One model, one accelerator, one decode at a time.
        with self._inference_lock:
            for window in windows:
                text = self._generate_window(loaded, window.samples, sample_rate)
                chunks.append(
                    ChunkResult(
                        index=window.index,
                        start_sec=window.start_sec,
                        end_sec=window.end_sec,
                        text=text.strip(),
                    )
                )
        inference_ms = int(round((time.perf_counter() - started) * 1000))

        raw = " ".join(chunk.text for chunk in chunks if chunk.text).strip()
        return TranscriptionResult(
            model_key=model_key,
            text_raw=raw,
            text_normalized=normalize_text(raw),
            chunks=tuple(chunks),
            checkpoint=loaded.spec.source,
            gen_config_hash=loaded.gen_config_hash,
            device=loaded.device,
            audio_duration_sec=duration,
            inference_ms=inference_ms,
            rtf=(inference_ms / 1000.0 / duration) if duration > 0 else 0.0,
        )

    def transcribe(
        self, model_key: str, media_path: Path, work_dir: Path | None = None
    ) -> TranscriptionResult:
        """Convert ``media_path`` to WAV if needed, then transcribe it."""
        wav_path, info = self.prepare_audio(media_path, work_dir)
        del info
        return self.transcribe_wav(model_key, wav_path)

    def prepare_audio(
        self, media_path: Path, work_dir: Path | None = None
    ) -> tuple[Path, AudioInfo]:
        """Decode any accepted container to the canonical WAV form."""
        target_dir = work_dir or media_path.parent
        wav_path = target_dir / f"{media_path.stem}.16k.wav"
        info = to_wav16k_mono(media_path, wav_path, self._config)
        return wav_path, info

    def transcribe_all(
        self, media_path: Path, work_dir: Path | None = None
    ) -> dict[str, TranscriptionResult]:
        """Run one file through all three models (the 'compare all 3' mode)."""
        wav_path, _ = self.prepare_audio(media_path, work_dir)
        return {
            key: self.transcribe_wav(key, wav_path) for key in sorted(self._specs)
        }


#: Process-wide instance. Phase 3's job worker uses this so models are loaded
#: once per process, not once per request.
_service: TranscriptionService | None = None
_service_lock = threading.Lock()


def get_transcription_service(config: Settings | None = None) -> TranscriptionService:
    global _service
    with _service_lock:
        if _service is None:
            _service = TranscriptionService(config)
        return _service


def reset_transcription_service() -> None:
    """Drop the cached service (tests only)."""
    global _service
    with _service_lock:
        _service = None
