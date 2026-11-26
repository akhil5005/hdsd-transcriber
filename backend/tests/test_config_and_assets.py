"""Tests for settings parsing and the read-only research-asset guard."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.research_assets import (
    ALLOWED_BENCHMARK_FILES,
    ResearchAssetError,
    benchmark_csv,
    describe_assets,
    verify_checkpoints,
)


def defaults(**overrides) -> Settings:
    """Settings built from code defaults only.

    ``_env_file=None`` skips backend/.env, so these assertions describe what a
    fresh checkout does rather than what the current developer's .env happens
    to say.
    """
    return Settings(_env_file=None, **overrides)


def test_cors_origins_accepts_a_comma_separated_string() -> None:
    config = defaults(cors_origins="http://a.test, http://b.test ,")
    assert config.cors_origins == ["http://a.test", "http://b.test"]


def test_cors_origins_accepts_a_list() -> None:
    config = defaults(cors_origins=["http://a.test"])
    assert config.cors_origins == ["http://a.test"]


def test_log_level_is_upper_cased() -> None:
    assert defaults(log_level="debug").log_level == "DEBUG"


def test_max_upload_bytes_matches_the_megabyte_setting() -> None:
    assert defaults(max_upload_mb=200).max_upload_bytes == 200 * 1024 * 1024


def test_pinned_base_revision_is_the_researched_commit() -> None:
    config = defaults()
    assert config.base_model_id == "openai/whisper-small"
    assert config.base_model_revision == "973afd24965f72e36ca33b3055d56a652f456b4d"


def test_selected_seeds_match_the_development_set_choice() -> None:
    config = defaults()
    assert config.full_ft_seed == 123
    assert config.lora_seed == 2026


def test_decoding_limits_match_the_frozen_protocol() -> None:
    config = defaults()
    assert config.max_new_tokens == 96
    assert config.chunk_seconds == 30
    assert config.target_sample_rate == 16000


# --- research-asset guard ---------------------------------------------------
def test_missing_checkpoint_directory_names_the_path(tmp_path) -> None:
    config = defaults(
        full_ft_dir=tmp_path / "no-such-full",
        lora_dir=tmp_path / "no-such-lora",
    )
    with pytest.raises(ResearchAssetError) as excinfo:
        verify_checkpoints(config)
    assert "no-such-full" in str(excinfo.value)


def test_missing_weights_file_names_the_expected_file(tmp_path) -> None:
    full_dir = tmp_path / "full" / "best"
    full_dir.mkdir(parents=True)
    config = defaults(full_ft_dir=full_dir, lora_dir=full_dir)

    with pytest.raises(ResearchAssetError) as excinfo:
        verify_checkpoints(config)
    assert "model.safetensors" in str(excinfo.value)


def test_verify_checkpoints_accepts_present_weights(tmp_path) -> None:
    full_dir = tmp_path / "full"
    lora_dir = tmp_path / "lora"
    full_dir.mkdir()
    lora_dir.mkdir()
    (full_dir / "model.safetensors").write_bytes(b"")
    (lora_dir / "adapter_model.safetensors").write_bytes(b"")

    resolved = verify_checkpoints(defaults(full_ft_dir=full_dir, lora_dir=lora_dir))
    assert resolved["full_ft"].name == "model.safetensors"
    assert resolved["lora"].name == "adapter_model.safetensors"


@pytest.mark.parametrize(
    "filename",
    ["speakers.csv", "utterance_errors.csv", "prompts.csv", "../sources.json"],
)
def test_benchmark_csv_refuses_files_outside_the_allow_list(tmp_path, filename) -> None:
    """The research folder holds private patient transcripts next to the tables."""
    config = defaults(benchmark_dir=tmp_path)
    with pytest.raises(ResearchAssetError):
        benchmark_csv(filename, config)


def test_benchmark_csv_resolves_allow_listed_files(tmp_path) -> None:
    for name in ALLOWED_BENCHMARK_FILES:
        (tmp_path / name).write_text("model_key\n", encoding="utf-8")

    config = defaults(benchmark_dir=tmp_path)
    for name in ALLOWED_BENCHMARK_FILES:
        assert benchmark_csv(name, config).name == name


def test_describe_assets_reports_absence_without_raising(tmp_path) -> None:
    config = defaults(
        full_ft_dir=tmp_path / "nope",
        lora_dir=tmp_path / "nope",
        benchmark_dir=tmp_path / "nope",
    )
    summary = describe_assets(config)
    assert summary["full_ft_weights_present"] is False
    assert summary["lora_weights_present"] is False
    assert set(summary["benchmark_files_present"]) == set(ALLOWED_BENCHMARK_FILES)


# --- deployment settings ----------------------------------------------------
def test_cors_origin_regex_defaults_to_none() -> None:
    assert defaults().cors_origin_regex is None


def test_blank_cors_origin_regex_becomes_none() -> None:
    """An unset Render env var arrives as an empty string, not as absent."""
    assert defaults(cors_origin_regex="   ").cors_origin_regex is None


def test_cors_origin_regex_matches_vercel_preview_hosts() -> None:
    import re

    pattern = defaults(cors_origin_regex=r"https://.*\.vercel\.app").cors_origin_regex
    assert re.fullmatch(pattern, "https://hdsd-transcriber-git-abc123.vercel.app")
    assert re.fullmatch(pattern, "https://hdsd-transcriber.vercel.app")
    assert not re.fullmatch(pattern, "https://evil.com")


def test_inference_can_be_disabled_for_hosts_without_weights() -> None:
    assert defaults().inference_enabled is True
    assert defaults(inference_enabled=False).inference_enabled is False


# --- GPU memory settings ----------------------------------------------------
def test_base_sharing_is_on_by_default() -> None:
    """A 4 GB card needs it: sharing saves one ~973 MB FP32 copy."""
    assert defaults().share_base_model is True


def test_resident_model_limit_defaults_to_unlimited() -> None:
    assert defaults().max_resident_models == 0


def test_resident_model_limit_is_configurable() -> None:
    assert defaults(max_resident_models=1).max_resident_models == 1


# --- verification-only paths ------------------------------------------------
def test_verification_paths_point_at_the_frozen_artifacts() -> None:
    config = defaults()
    assert config.prepared_dir.name == "v1"
    assert config.final_v3_dir.name == "final_v3"
    assert config.hdsd_dataset_dir.name == "HDSD_Dataset"
