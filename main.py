"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import FastAPI

from config import Settings
from db.database import Database
from db.repositories import AuthRepository
from llm.models import ModelManager
from router import api_router
from service.auth import AuthService
from service.inference import InferenceService


def create_app(
    settings: Optional[Settings] = None,
    client_factory: Optional[Callable[..., Any]] = None,
    cipher: Optional[Any] = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime_settings = settings or Settings.load()
        database = Database(runtime_settings.database_path)
        database.initialize()
        application.state.database = database
        application.state.auth_service = AuthService(
            AuthRepository(database), runtime_settings.setup_secret
        )
        application.state.model_manager = ModelManager(
            database,
            runtime_settings.credential_key,
            client_factory=client_factory,
            cipher=cipher,
        )
        application.state.inference_service = InferenceService(application.state.model_manager)
        yield

    application = FastAPI(title="Content Security LLM", lifespan=lifespan)
    application.include_router(api_router)

    @application.get("/")
    def root():
        return {"message": "Content Security LLM"}

    return application


app = create_app()
