"""Application settings, loaded from the environment / .env file.

Every tunable the research protocol depends on (checkpoint paths, pinned base
revision, decoding limits, selected seeds) lives here so it can be changed
without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---------------------------------------------------------------
    app_name: str = "HDSD Hindi Transcriber"
    environment: Literal["development", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    # --- Database ----------------------------------------------------------
    # A managed Postgres connection string. `postgres://` and libpq query
    # parameters (`?sslmode=require`) are normalized for asyncpg at engine
    # construction -- paste a provider URL verbatim. See app/db/url.py.
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/hdsd_transcriber"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_recycle_seconds: int = 280

    # --- Auth --------------------------------------------------------------
    secret_key: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    login_rate_limit_attempts: int = 5
    login_rate_limit_window_seconds: int = 300

    # --- CORS --------------------------------------------------------------
    # NoDecode: read the raw .env string and split it in the validator below,
    # instead of pydantic-settings trying to JSON-decode it.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    #: Matches Vercel preview deployments, whose hostname changes per commit.
    #: Leave empty to allow only the exact origins above.
    cors_origin_regex: str | None = None

    # --- Research assets (READ-ONLY) ---------------------------------------
    hdsd_root: Path = Path(
        "<RESEARCH_ARCHIVE>/hdsd_simple/runs"
    )
    full_ft_dir: Path = Path(
        "<RESEARCH_ARCHIVE>/hdsd_simple/runs/full_lr5e-06_seed123/best"
    )
    lora_dir: Path = Path(
        "<RESEARCH_ARCHIVE>/hdsd_simple/runs/lora_lr5e-05_seed2026/best"
    )
    benchmark_dir: Path = Path(
        "<RESEARCH_ARCHIVE>/hdsd_simple/runs/final_v3/tables"
    )
    #: Frozen prepared manifest (read-only). Used only by the verification
    #: script to map utterance ids to audio files.
    prepared_dir: Path = Path(
        "<RESEARCH_ARCHIVE>/hdsd_simple/prepared/v1"
    )
    #: Saved per-model predictions for every split (read-only), also
    #: verification-only.
    final_v3_dir: Path = Path(
        "<RESEARCH_ARCHIVE>/hdsd_simple/runs/final_v3"
    )
    #: Original HDSD audio corpus (READ-ONLY). Not needed to run the app --
    #: only app.scripts.verify_against_research reads it.
    hdsd_dataset_dir: Path = Path("<HDSD_DATASET>")

    base_model_id: str = "openai/whisper-small"
    base_model_revision: str = "973afd24965f72e36ca33b3055d56a652f456b4d"

    full_ft_seed: int = 123
    lora_seed: int = 2026
    full_ft_learning_rate: float = 5e-06
    lora_learning_rate: float = 5e-05

    # --- Inference ---------------------------------------------------------
    #: False on hosts without the checkpoints (e.g. a free cloud instance):
    #: auth, benchmark and history still work, transcription returns 503.
    inference_enabled: bool = True
    device: Literal["auto", "cuda", "cpu"] = "auto"

    #: Share one base-model instance between `pretrained` and `lora`.
    #: The LoRA adapter leaves base weights untouched and is switched off with
    #: PEFT's disable_adapter(), so `pretrained` is mathematically unchanged --
    #: and it saves ~973 MB of VRAM, which matters on a 4 GB card.
    share_base_model: bool = True
    #: Maximum models kept resident on the accelerator at once. 0 = unlimited.
    #: With sharing on, all three fit in <2 GB; set to 1 to force load-one,
    #: unload-after-use on very small cards.
    max_resident_models: int = 0
    max_new_tokens: int = 96
    chunk_seconds: int = 30
    chunk_overlap_seconds: float = 0.0
    target_sample_rate: int = 16000

    # --- Uploads -----------------------------------------------------------
    upload_dir: Path = BACKEND_DIR / "uploads"
    max_upload_mb: int = 200
    media_retention_days: int = 30
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"

    # ----------------------------------------------------------------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string from .env as well as a real list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("cors_origin_regex", mode="before")
    @classmethod
    def _empty_regex_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def resolved_upload_dir(self) -> Path:
        """Absolute upload directory, created on demand."""
        path = self.upload_dir
        if not path.is_absolute():
            path = (BACKEND_DIR / path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
