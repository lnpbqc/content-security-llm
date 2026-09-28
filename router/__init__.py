"""Assemble focused HTTP routers under the public API prefix."""

from fastapi import APIRouter

from .auth import router as auth_router
from .calls import router as calls_router
from .inference import router as inference_router
from .models import router as models_router


api_router = APIRouter(prefix="/api")
api_router.include_router(auth_router)
api_router.include_router(models_router)
api_router.include_router(inference_router)
api_router.include_router(calls_router)
