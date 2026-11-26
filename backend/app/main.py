"""FastAPI application factory and lifespan.

Startup verifies the two fine-tuned checkpoints are reachable and fails loudly
if they are not, naming the offending path -- a missing checkpoint should be a
boot error, not a mystery at first transcription.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.research_assets import ResearchAssetError, verify_checkpoints
from app.db.session import engine
from app.services.jobs import get_worker

logger = get_logger(__name__)

DISCLAIMER = (
    "Research prototype. Exploratory results on the HDSD dataset. "
    "Not for diagnosis or clinical decisions."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Starting %s (%s)", settings.app_name, settings.environment)

    if settings.inference_enabled:
        # A missing checkpoint must be a boot error, not a mystery at the
        # first transcription.
        try:
            checkpoints = verify_checkpoints()
        except ResearchAssetError as exc:
            logger.error("Research asset check failed:\n%s", exc)
            raise

        for key, path in checkpoints.items():
            logger.info("Checkpoint %s -> %s", key, path)
        logger.info(
            "Pinned base model: %s@%s",
            settings.base_model_id,
            settings.base_model_revision,
        )
    else:
        # Hosts that cannot hold ~1 GB of weights (Render's free 512 MB
        # instance) run in this mode: auth, benchmark and history work, and
        # the transcription endpoints refuse cleanly with 503.
        logger.warning(
            "INFERENCE_ENABLED=false -- transcription endpoints will return 503. "
            "Auth, benchmark and history are unaffected."
        )

    settings.resolved_upload_dir()

    worker = get_worker()
    await worker.start()
    # A previous process may have died mid-job; reconcile before serving.
    await worker.recover_orphans()

    yield

    await worker.stop()
    await engine.dispose()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Hindi (Devanagari) transcription of dysarthric speech with three "
            f"Whisper-small variants, plus model comparison metrics.\n\n{DISCLAIMER}"
        ),
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Exact origins for local Vite; the regex additionally admits Vercel
    # preview deployments, whose hostname changes on every commit.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "app": settings.app_name,
            "api": settings.api_v1_prefix,
            "disclaimer": DISCLAIMER,
        }

    return app


app = create_app()
