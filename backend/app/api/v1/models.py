"""`GET /models` -- the three models with metadata and live load state."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.models.registry import ModelRegistry
from app.schemas.transcription import ModelInfo
from app.services.transcription import (
    get_transcription_service,
    model_specs,
    resolve_device,
)

router = APIRouter(tags=["models"])

MODEL_ORDER = ("pretrained", "full_ft", "lora")


@router.get("/models", response_model=list[ModelInfo])
async def list_models(_: CurrentUser, db: DbSession) -> list[ModelInfo]:
    """Registry rows plus whether each model's weights are resident.

    Reading load state must not trigger a load -- that would turn a cheap
    listing into a multi-second, multi-GB operation.
    """
    rows = {
        entry.key: entry
        for entry in (await db.execute(select(ModelRegistry))).scalars()
    }
    loaded = set(get_transcription_service().loaded_keys())
    specs = model_specs()

    out: list[ModelInfo] = []
    for key in MODEL_ORDER:
        entry = rows.get(key)
        if entry is not None:
            info = ModelInfo.model_validate(entry)
        else:
            # Registry not seeded yet: fall back to the static spec so the
            # frontend still has something to render.
            spec = specs[key]
            info = ModelInfo(
                key=spec.key,
                display_name=spec.display_name,
                kind=spec.kind,
                learning_rate=spec.learning_rate,
                seed=spec.seed,
                path_or_revision=spec.source,
                trainable_params=0,
                checkpoint_bytes=None,
                is_active=True,
            )
        info.loaded = key in loaded
        out.append(info)
    return out


@router.get("/models/device", include_in_schema=False)
async def device_info(_: CurrentUser) -> dict[str, object]:
    """What the UI needs for the 'CPU will be slower' notice."""
    service = get_transcription_service()
    return {
        "resolved_device": resolve_device(),
        "loaded": service.loaded_keys(),
        "gen_config_hash": service.shared_gen_config_hash,
        "vram": service.peak_vram(),
    }
