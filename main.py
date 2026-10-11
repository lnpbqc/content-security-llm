"""组装原有 API 和新增的治理任务 API、依赖及错误响应。"""

from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from uuid import uuid4

from config import Settings
from db.database import Database
from db.analysis import AnalysisRepository
from db.neural_trace import NeuralTraceRepository
from db.governance import GovernanceRepository
from db.governance_config import GovernanceConfigRepository
from db.repositories import AuthRepository
from db.training import TrainingRepository
from llm.models import ModelManager
from router import api_router
from router.analysis import router as analysis_router
from router.neural_trace import router as neural_trace_router
from router.model_frontend import router as model_frontend_router
from router.governance import router as governance_router
from router.governance_admin import router as governance_admin_router
from router.governance_frontend import router as governance_frontend_router
from router.training import router as training_router
from service.auth import AuthService
from service.analysis import AnalysisService
from service.neural_trace import NeuralTraceService
from service.business_users import BusinessUserService
from service.inference import InferenceService
from service.governance import GovernanceService
from service.governance_data import BusinessGovernanceData
from service.training import TrainingService
from utils.time import utc_now


def create_app(
    settings: Optional[Settings] = None,
    client_factory: Optional[Callable[..., Any]] = None,
    cipher: Optional[Any] = None,
) -> FastAPI:
    """创建应用，并接入治理任务服务及其 `/api/v1` 路由。"""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """启动时初始化数据库和治理任务使用的共享服务。"""
        runtime_settings = settings or Settings.load()
        database = Database(runtime_settings.database_path)
        database.initialize()
        application.state.database = database
        application.state.neural_trace_service = NeuralTraceService(NeuralTraceRepository(database))
        application.state.auth_service = AuthService(
            AuthRepository(database), runtime_settings.setup_secret
        )
        application.state.business_user_service = BusinessUserService(runtime_settings.business_api_base_url)
        application.state.model_manager = ModelManager(
            database,
            runtime_settings.credential_key,
            client_factory=client_factory,
            cipher=cipher,
        )
        application.state.inference_service = InferenceService(application.state.model_manager)
        analysis_repository = AnalysisRepository(database)
        analysis_repository.recover_running()
        application.state.analysis_service = AnalysisService(
            analysis_repository, application.state.model_manager)
        application.state.training_service = TrainingService(TrainingRepository(database))
        governance_config = GovernanceConfigRepository(database)
        application.state.governance_config = governance_config
        application.state.governance_service = GovernanceService(
            GovernanceRepository(database),
            BusinessGovernanceData(governance_config, application.state.model_manager.cipher,
                                   mock_samples=runtime_settings.governance_mock_samples),
            application.state.model_manager, governance_config,
        )
        yield

    application = FastAPI(title="Content Security LLM", lifespan=lifespan)
    application.include_router(api_router)
    application.include_router(governance_frontend_router)
    application.include_router(governance_router)
    application.include_router(governance_admin_router)
    application.include_router(training_router)
    application.include_router(analysis_router)
    application.include_router(neural_trace_router)
    application.include_router(model_frontend_router)

    @application.exception_handler(HTTPException)
    async def governance_http_error(request: Request, exc: HTTPException):
        """只将新接口的业务错误转换为前端约定的响应格式。"""
        if not request.url.path.startswith("/api/v1/"):
            return await http_exception_handler(request, exc)
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        body = {
            "code": exc.status_code, "message": detail.get("message", detail.get("code", "请求失败")),
            "data": None, "trace_id": request.headers.get("X-Trace-Id") or str(uuid4()),
            "timestamp": utc_now().isoformat(),
        }
        if isinstance(detail.get("code"), str):
            body["error_code"] = detail["code"]
        if "upstream_status" in detail:
            body["upstream_status"] = detail["upstream_status"]
        return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)

    @application.exception_handler(RequestValidationError)
    async def governance_validation_error(request: Request, exc: RequestValidationError):
        """只将新接口的参数校验错误转换为前端约定的响应格式。"""
        if not request.url.path.startswith("/api/v1/"):
            return await request_validation_exception_handler(request, exc)
        return JSONResponse(status_code=422, content={
            "code": 422, "message": "请求参数无效", "data": None, "error_code": "validation_error",
            "trace_id": request.headers.get("X-Trace-Id") or str(uuid4()),
            "timestamp": utc_now().isoformat(),
        })

    @application.get("/")
    def root():
        return {"message": "Content Security LLM"}

    return application


app = create_app()
