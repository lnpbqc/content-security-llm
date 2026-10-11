"""有记录、引用验证及令牌隔离的独立分析入口。"""

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from llm.analysis_contracts import AuditRequest, GovernanceRequest, Kind, ScenarioCode, ScenarioRequest
from llm.models import ModelNotFoundError
from router.dependencies import require_token
from router.governance import success
from service.analysis import AnalysisFailure, SCENARIOS

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"], dependencies=[Depends(require_token)])


@router.get("/options")
def options(request: Request):
    return success(request, {"analysis_types": ["scenario", "model_governance", "audit"],
        "audit_types": ["reasoning", "full_chain"], "scenarios": [
            {"code": code, "name": name, "input_requirements": instructions,
             "supported_analysis_types": ["scenario"]} for code, (name, _, instructions) in SCENARIOS.items()]})


def execute(request, kind, payload, token_hash):
    try:
        return success(request, request.app.state.analysis_service.execute(kind, payload, token_hash))
    except ModelNotFoundError as error:
        raise HTTPException(404, detail={"message": "模型不存在或已删除"}) from error
    except AnalysisFailure as failure:
        body = success(request, failure.record)
        body.update(code=failure.error.status_code, message=str(failure.error))
        return JSONResponse(status_code=failure.error.status_code, content=body)


@router.post("/scenarios")
def scenarios(request: Request, payload: ScenarioRequest, token_hash: str = Depends(require_token)):
    return execute(request, "scenario", payload, token_hash)


@router.post("/model-governance")
def model_governance(request: Request, payload: GovernanceRequest, token_hash: str = Depends(require_token)):
    return execute(request, "model_governance", payload, token_hash)


@router.post("/audits")
def audits(request: Request, payload: AuditRequest, token_hash: str = Depends(require_token)):
    return execute(request, "audit", payload, token_hash)


@router.get("/records")
def records(request: Request, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
            kind: Optional[Kind] = None, scenario_code: Optional[ScenarioCode] = None,
            status: Optional[Literal["running", "succeeded", "failed"]] = None,
            token_hash: str = Depends(require_token)):
    return success(request, request.app.state.analysis_service.repository.list(
        token_hash, page, page_size, kind, scenario_code, status))


@router.get("/records/{record_id}")
def record(request: Request, record_id: str, token_hash: str = Depends(require_token)):
    found = request.app.state.analysis_service.repository.get(record_id, token_hash)
    if found is None:
        raise HTTPException(404, detail={"message": "分析记录不存在"})
    return success(request, found)
