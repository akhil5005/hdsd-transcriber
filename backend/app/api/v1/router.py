"""Aggregates the v1 routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import admin, auth, benchmark, health, models, transcriptions

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(models.router)
api_router.include_router(transcriptions.router)
api_router.include_router(benchmark.router)
api_router.include_router(admin.router)
