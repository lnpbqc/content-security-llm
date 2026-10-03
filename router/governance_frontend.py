"""补齐治理前端需要的只读方案、统计和导出接口。"""

from typing import Any, Dict, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from router.dependencies import require_token
from router.governance import get_service, query_scope, success, _get_result, value_kind, anomaly_kind, risk_kind
from service.governance import GovernanceService


router = APIRouter(prefix="/api/v1", tags=["governance-frontend"],
                   dependencies=[Depends(require_token)])


def _latest_versions(service: GovernanceService, kind: str, token_hash: str):
    """同一数据集版本只纳入最近一次保存结果，不累加重复分析。"""
    latest = {}
    for result in service.repository.list_results(kind, token_hash):
        scope = result["scope"]
        latest.setdefault((scope["dataset_id"], scope["version_id"]), result)
    return list(latest.values())


@router.get("/data-governance/options")
def governance_options(request: Request, kind: str = Query(...),
                       service: GovernanceService = Depends(get_service)):
    """目录继续由业务后端提供，本接口只展示本服务实际支持的方案。"""
    if kind == "governance-value":
        schemes = []
        for scheme_id in ("general-v1", "general-v2"):
            policy = service.data.policy(kind, scheme_id)
            schemes.append({"id": scheme_id, "name": policy["name"],
                            "description": "文化价值、信息价值、稀缺性、可信度、代表性五维等权分析",
                            "high_threshold": policy["high_threshold"],
                            "medium_threshold": policy["medium_threshold"]})
        return success(request, {"schemes": schemes})
    if kind != "governance-anomaly":
        raise HTTPException(400, detail={"code": "invalid_kind", "message": "任务类型与路径不一致"})
    policy = service.data.policy(kind, "anomaly-basic-v1")
    return success(request, {
        "schemes": [{"id": policy["scheme_id"], "name": policy["name"]}],
        "rules": [rule for rule in policy["rules"]
                  if rule["id"] in policy["model_applicable_rule_ids"]],
        "types": ["标签异常"], "read_only": True,
    })


@router.get("/data-governance/risk-options", dependencies=[risk_kind])
def risk_options(request: Request, service: GovernanceService = Depends(get_service)):
    """给风险页提供分级方案、筛选选项和已有只读动作。"""
    policy = service.data.policy("governance-risk", "risk-v1")
    return success(request, {
        "datasets": [],
        "schemes": [{"id": policy["scheme_id"], "name": policy["name"],
                     "levels": policy["levels"]}],
        "categories": policy["categories"], "page_sizes": [5, 10, 20],
        "default_scope": {"dataset_id": 0, "version_id": "", "language": "all",
                          "scheme_id": policy["scheme_id"]},
        "actions": ["detect", "history"], "review_statuses": ["未发起", "待复核", "无法评估"],
        "review_decisions": [], "reviewers": [], "read_only": True,
    })


@router.get("/kpis", dependencies=[value_kind])
def value_kpis(request: Request, kind: Literal["governance-value"] = Query(...),
               service: GovernanceService = Depends(get_service),
               token_hash: str = Depends(require_token)):
    """价值概览使用当前令牌的最新版本结果，按有效样本加权计算均分。"""
    results = _latest_versions(service, kind, token_hash)
    rows = [row for result in results for row in result["samples"]]
    valid = [row for row in rows if row["score"] is not None]
    high = sum(row["tier"] == "high" for row in valid)
    values = [round(sum(row["score"] for row in valid) / len(valid), 1) if valid else 0,
              round(high * 100 / len(valid), 1) if valid else 0,
              len(valid), len({row["language"] for row in rows})]
    labels = ["综合价值评分", "高价值语料占比", "已分析语料", "覆盖语种"]
    units = ["分", "%", "条", "种"]
    icons = ["Trophy", "Document", "Coin", "Position"]
    return success(request, [{"id": "value-{}".format(index), "label": labels[index],
                              "value": value, "unit": units[index], "change_rate": 0,
                              "icon": icons[index]} for index, value in enumerate(values)])


@router.get("/overview", dependencies=[anomaly_kind])
@router.get("/data-governance/anomaly-results/overview", dependencies=[anomaly_kind])
def anomaly_overview(request: Request, kind: Literal["governance-anomaly"] = Query("governance-anomaly"),
                     service: GovernanceService = Depends(get_service),
                     token_hash: str = Depends(require_token)):
    """异常概览只统计已保存的检测结果，不虚构人工复核或修复数量。"""
    results = _latest_versions(service, kind, token_hash)
    summaries = [result["summary"] for result in results]
    values = [sum(row["valid_count"] + row["unavailable_count"] + row["failed_count"] for row in summaries),
              sum(row["anomaly_count"] for row in summaries), 0, 0]
    labels = ["已检测语料", "检出异常", "待复核样本", "已完成修复"]
    return success(request, {
        "cards": [{"label": label, "value": value, "icon": "Document"}
                  for label, value in zip(labels, values)],
        "definitions": ["仅统计当前访问令牌产生的结果；同一数据集版本取最近一次已保存分析，不累加重复分析。",
                        "本服务只提供检测和建议，不提供复核工单或修复应用，相关数量为零。"],
    })


@router.get("/data-governance/risk-overview", dependencies=[risk_kind])
def risk_overview(request: Request, service: GovernanceService = Depends(get_service),
                  token_hash: str = Depends(require_token)):
    """风险概览只使用当前令牌的结果快照。"""
    results = _latest_versions(service, "governance-risk", token_hash)
    summaries = [result["summary"] for result in results]
    labels = ["有效检测语料", "风险候选", "高风险样本", "建议人工复核"]
    fields = ["valid_count", "risk_count", "high_count", "pending_count"]
    return success(request, {
        "cards": [{"label": label, "value": sum(row[field] for row in summaries), "icon": "Shield"}
                  for label, field in zip(labels, fields)],
        "definitions": ["仅统计当前访问令牌产生的结果；同一数据集版本取最近一次已保存分析，不累加重复分析。",
                        "风险等级和待复核数量是模型建议，本服务不执行人工复核。"],
        "records": [{"result_id": row["id"], "dataset_name": row["dataset_name"],
                     "version_id": row["scope"]["version_id"], "valid_count": row["valid_count"],
                     "risk_count": row["risk_count"], "finished_at": row["finished_at"]}
                    for row in summaries],
    })


@router.post("/data-governance/risk-results/{result_id}/export", dependencies=[risk_kind])
def export_risk(request: Request, result_id: str, payload: Dict[str, Any] = Body(default={}),
                service: GovernanceService = Depends(get_service),
                token_hash: str = Depends(require_token)):
    """导出当前保存结果的全部匹配样本，不调用模型、不受分页窗口影响。"""
    rows = _get_result(service, result_id, "governance-risk", token_hash)["samples"]
    keyword, level, status = (str(payload.get(key, "")) for key in ("keyword", "level", "status"))
    rows = [row for row in rows if (not keyword or keyword in row["id"] + row["text"]) and
            (not level or row["findings"] and row["maximum_suggested_level"] == level) and
            (not status or row["status"] == status)]
    return success(request, rows)


@router.get("/risk-knowledge", dependencies=[risk_kind])
def risk_knowledge(request: Request, result_id: str, sample_id: str, keyword: str = "",
                   service: GovernanceService = Depends(get_service),
                   token_hash: str = Depends(require_token)):
    """检索指定样本快照的规则，避免把检索请求转为新的模型调用。"""
    rows = _get_result(service, result_id, "governance-risk", token_hash)["samples"]
    sample = next((row for row in rows if row["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(404, detail={"message": "样本不存在"})
    keyword = keyword.lower()
    return success(request, [rule for rule in sample["rules"]
                             if keyword in "{}{}{}{}".format(rule["id"], rule["name"],
                                                             rule["category"], rule["text"]).lower()])


@router.get("/data-governance/change-sets/current", dependencies=[anomaly_kind])
def current_change_set(request: Request, scope: Dict[str, Any] = Depends(query_scope)):
    """允许只读页面初始化；本服务没有业务修改集，不能伪造待发布内容。"""
    return success(request, None)


@router.post("/data-governance/change-sets/{action}")
@router.post("/data-governance/anomaly-results/{result_id}/samples/{sample_id}/candidates/{action}")
@router.post("/data-governance/risk-results/{result_id}/samples/{sample_id}/reviews")
def read_only_operation():
    """业务写操作由原后端负责，本服务明确拒绝而非伪造成功。"""
    raise HTTPException(405, detail={"message": "当前模型结果仅支持分析查看，业务写操作请使用业务后端"})
