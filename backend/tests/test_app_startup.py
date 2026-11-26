"""Lifespan behaviour around the research checkpoints.

Section 3 of the build spec requires a hard startup failure when a checkpoint
is missing. That must still hold locally, while a deployed host that carries no
weights at all (INFERENCE_ENABLED=false) has to boot normally.

``httpx.ASGITransport`` does not emit lifespan events, so these tests drive
``app.router.lifespan_context`` directly.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.core.research_assets import ResearchAssetError
from app.main import create_app


@pytest.fixture
def patched_settings(tmp_path, session_factory, monkeypatch):
    """Point the cached settings at empty paths, then restore them.

    Also redirects the worker's own session factory: the lifespan reconciles
    orphaned jobs on startup, which touches the database.
    """
    from app.services import jobs

    monkeypatch.setattr(jobs, "SessionLocal", session_factory)
    config = get_settings()
    saved = (
        config.inference_enabled,
        config.full_ft_dir,
        config.lora_dir,
        config.upload_dir,
    )
    config.full_ft_dir = tmp_path / "missing-full"
    config.lora_dir = tmp_path / "missing-lora"
    config.upload_dir = tmp_path / "uploads"
    yield config
    (
        config.inference_enabled,
        config.full_ft_dir,
        config.lora_dir,
        config.upload_dir,
    ) = saved


async def test_startup_fails_when_a_checkpoint_is_missing(patched_settings) -> None:
    patched_settings.inference_enabled = True
    app = create_app()

    with pytest.raises(ResearchAssetError, match="missing-full"):
        async with app.router.lifespan_context(app):
            pass


async def test_startup_error_names_the_env_file_to_fix(patched_settings) -> None:
    patched_settings.inference_enabled = True
    app = create_app()

    with pytest.raises(ResearchAssetError) as excinfo:
        async with app.router.lifespan_context(app):
            pass
    assert "backend/.env" in str(excinfo.value)


async def test_startup_succeeds_without_weights_when_inference_is_disabled(
    patched_settings,
) -> None:
    patched_settings.inference_enabled = False
    app = create_app()

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

    assert response.status_code == 200
    assert "not for diagnosis" in response.json()["disclaimer"].lower()


async def test_startup_creates_the_upload_directory(patched_settings) -> None:
    patched_settings.inference_enabled = False
    app = create_app()

    async with app.router.lifespan_context(app):
        assert patched_settings.resolved_upload_dir().is_dir()


# --- health probe -----------------------------------------------------------
async def test_health_is_ok_without_weights_when_inference_is_disabled(
    client, patched_settings
) -> None:
    """Render polls the health path; a 'degraded' loop would restart the app."""
    patched_settings.inference_enabled = False

    response = await client.get("/api/v1/health")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["inference_enabled"] is False
    assert body["research_assets"]["full_ft_weights_present"] is False


async def test_health_is_degraded_when_inference_is_on_but_weights_are_gone(
    client, patched_settings
) -> None:
    patched_settings.inference_enabled = True

    body = (await client.get("/api/v1/health")).json()
    assert body["status"] == "degraded"


async def test_health_lists_the_three_models_without_loading_them(
    client, patched_settings
) -> None:
    """Polling the probe must not trigger a multi-second, multi-GB load."""
    patched_settings.inference_enabled = False

    models = (await client.get("/api/v1/health")).json()["models"]

    assert [m["key"] for m in models["available"]] == [
        "pretrained",
        "full_ft",
        "lora",
    ]
    assert all(m["loaded"] is False for m in models["available"])
    assert models["loaded"] == []
    assert models["gen_config_hash"] is None
