"""Health endpoint: database reachability, research assets and model status.

Reading model status must never *cause* a load -- the probe is polled, and
loading three FP32 copies of whisper-small takes seconds and ~3 GB. So this
reports which models the service has already cached, nothing more.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter

from app.api.deps import DbSession
from app.core.config import settings
from app.core.research_assets import describe_assets
from app.services.transcription import get_transcription_service, model_specs

router = APIRouter(tags=["health"])


def _model_status() -> dict[str, Any]:
    service = get_transcription_service()
    loaded = set(service.loaded_keys())
    return {
        "available": [
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "kind": spec.kind,
                "seed": spec.seed,
                "loaded": spec.key in loaded,
            }
            for spec in model_specs().values()
        ],
        "loaded": sorted(loaded),
        "device_setting": settings.device,
        "gen_config_hash": service.shared_gen_config_hash,
    }


@router.get("/health")
async def health(db: DbSession) -> dict[str, Any]:
    try:
        await db.execute(sa.text("SELECT 1"))
        database = {"ok": True}
    except Exception as exc:  # pragma: no cover - depends on a live database
        database = {"ok": False, "error": type(exc).__name__}

    assets = describe_assets()
    assets_ok = bool(assets["full_ft_weights_present"]) and bool(
        assets["lora_weights_present"]
    )

    # With inference disabled (a host without the weights) the missing
    # checkpoints are expected, so they must not make the probe fail -- Render
    # would restart the service in a loop.
    healthy = database["ok"] and (assets_ok or not settings.inference_enabled)

    return {
        "status": "ok" if healthy else "degraded",
        "app": settings.app_name,
        "environment": settings.environment,
        "database": database,
        "inference_enabled": settings.inference_enabled,
        "research_assets": assets,
        "models": _model_status(),
    }
