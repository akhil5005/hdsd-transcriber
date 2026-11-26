"""Guards around the read-only research folder.

The research archive contains private patient transcripts. This module is the
only sanctioned door into it: it exposes the two checkpoint folders and the
three benchmark CSVs named in the build spec, and nothing else. Every helper
here is read-only by construction -- no function in the codebase should write,
rename or delete a path under ``HDSD_ROOT``.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings, settings

#: The only benchmark files we are allowed to read.
ALLOWED_BENCHMARK_FILES: tuple[str, ...] = (
    "per_run.csv",
    "model_summary.csv",
    "efficiency.csv",
)

FULL_FT_WEIGHTS = "model.safetensors"
LORA_WEIGHTS = "adapter_model.safetensors"


class ResearchAssetError(RuntimeError):
    """A required research asset is missing or access to it is not permitted."""


def _require_file(directory: Path, filename: str, label: str) -> Path:
    candidate = directory / filename
    if not directory.is_dir():
        raise ResearchAssetError(
            f"{label} directory does not exist: {directory}\n"
            "Check the path in backend/.env (see .env.example)."
        )
    if not candidate.is_file():
        raise ResearchAssetError(
            f"{label} is missing its weights file: expected {candidate}\n"
            "Check the path in backend/.env (see .env.example)."
        )
    return candidate


def verify_checkpoints(config: Settings | None = None) -> dict[str, Path]:
    """Fail fast if either fine-tuned checkpoint is not where we expect it.

    Returns the resolved weights files, keyed by model key.
    """
    config = config or settings
    return {
        "full_ft": _require_file(config.full_ft_dir, FULL_FT_WEIGHTS, "Full fine-tune checkpoint"),
        "lora": _require_file(config.lora_dir, LORA_WEIGHTS, "LoRA adapter"),
    }


def benchmark_csv(filename: str, config: Settings | None = None) -> Path:
    """Resolve one allow-listed benchmark CSV inside ``BENCHMARK_DIR``.

    Raises if the name is not on the allow-list, so a typo or a future caller
    can never reach the private transcripts stored alongside these tables.
    """
    config = config or settings
    if filename not in ALLOWED_BENCHMARK_FILES:
        raise ResearchAssetError(
            f"{filename!r} is not an allow-listed benchmark file. "
            f"Allowed: {', '.join(ALLOWED_BENCHMARK_FILES)}"
        )
    path = config.benchmark_dir / filename
    if not path.is_file():
        raise ResearchAssetError(f"Benchmark table not found: {path}")
    return path


def describe_assets(config: Settings | None = None) -> dict[str, object]:
    """Non-fatal summary used by ``GET /health``."""
    config = config or settings
    full_weights = config.full_ft_dir / FULL_FT_WEIGHTS
    lora_weights = config.lora_dir / LORA_WEIGHTS
    return {
        "full_ft_dir": str(config.full_ft_dir),
        "full_ft_weights_present": full_weights.is_file(),
        "lora_dir": str(config.lora_dir),
        "lora_weights_present": lora_weights.is_file(),
        "benchmark_dir": str(config.benchmark_dir),
        "benchmark_files_present": {
            name: (config.benchmark_dir / name).is_file()
            for name in ALLOWED_BENCHMARK_FILES
        },
        "base_model": f"{config.base_model_id}@{config.base_model_revision}",
    }
