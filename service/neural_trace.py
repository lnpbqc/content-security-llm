"""溯源案例、输入快照与结果整理；本模块不导入模型运行库。"""

import csv
import hashlib
import json
import re
from pathlib import Path

from db.neural_trace import NeuralTraceRepository
from schemas.neural_trace import TraceResult


TRACE_ROOT = Path(__file__).resolve().parents[1] / "neurotrace-run"
MODEL_ROOT = TRACE_ROOT / "models" / "Qwen3-4B"
STAGES = ("document_trace", "prepare_inputs", "layer_scan", "activation_difference", "component_scan")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def public_task(task):
    return {"task_id": task["id"], **{key: task[key] for key in (
        "case_id", "status", "stage", "error_message", "created_at", "updated_at", "started_at", "finished_at",
    )}}


class NeuralTraceService:
    def __init__(self, repository: NeuralTraceRepository, data_root=None):
        self.repository = repository
        self.data_root = Path(data_root) if data_root else TRACE_ROOT / "data"

    def snapshot(self, case_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", case_id):
            raise KeyError(case_id)
        with (self.data_root / "manual_review.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next((row for row in csv.DictReader(stream) if row["case_id"] == case_id), None)
        if row is None or any(row.get(key, "").lower() != value for key, value in (
            ("review_status", "human_approved"), ("include_in_document", "yes"),
            ("include_in_neural_tracing", "yes"),
        )):
            raise KeyError(case_id)
        sources = {name: (self.data_root / "upstream" / case_id / f"{name}.json").read_text(
            encoding="utf-8-sig") for name in ("E1", "E2")}
        clean, poison = (json.loads(sources[name]) for name in ("E1", "E2"))
        fields = ("case_id", "query", "gold_answer", "poison_target_answer")
        if (clean["case_id"] != case_id or any(clean[key] != poison[key] for key in fields)
                or any(record["experiment_id"] != name or record["status"] != "success"
                       for record, name in ((clean, "E1"), (poison, "E2")))):
            raise ValueError("案例的 E1/E2 输入不一致。")
        return {"case": {key: clean[key] for key in fields}, "review": row, "sources": sources,
                "hashes": {name: digest(text) for name, text in sources.items()}}

    def cases(self):
        with (self.data_root / "manual_review.csv").open(encoding="utf-8-sig", newline="") as stream:
            ids = [row["case_id"] for row in csv.DictReader(stream)]
        return [self.snapshot(case_id)["case"] for case_id in ids]

    def create(self, token_hash, case_id):
        return self.repository.create(token_hash, case_id, self.snapshot(case_id))


def checked_result(path, case_id, stage):
    result = read_json(path)
    if result.get("case_id") != case_id:
        raise ValueError(f"{stage} 的案例 ID 不匹配。")
    if stage == "document_trace":
        passed = result.get("formal_result") is True and result.get("summary", {}).get("document_trace_success") is True
    elif stage == "prepare_inputs":
        passed = result.get("validation", {}).get("alignment_valid") is True
    else:
        passed = result.get("technical_checks", {}).get("technical_pass") is True
        if stage == "layer_scan":
            passed = (passed and result.get("intervention", {}).get("all_layers_scanned") is True
                      and result["intervention"].get("patch_scope") == "candidate_prediction_span")
        if stage == "component_scan":
            intervention = result.get("intervention", {})
            passed = (passed and intervention.get("actual_generation_performed") is True
                      and intervention.get("patch_scope") == "candidate_prediction_span"
                      and intervention.get("generation_patch_scope") == "generation_prompt_last_token")
    if not passed:
        raise ValueError(f"{stage} 未通过正式输入、因果恢复或技术检查；不继续发布结果。")
    return result


def collect_result(task, outputs, model_root=MODEL_ROOT):
    case_id = task["case_id"]
    document, alignment, layers, activation, components = (
        checked_result(outputs[stage], case_id, stage) for stage in STAGES
    )
    conditions = [{"name": name, "answer": value["generation"]["model_answer"],
                   "margin": value["margin"], "gold_answer_match": value["gold_answer_match"],
                   "poison_answer_match": value["poison_answer_match"],
                   "intervention_success": value.get("intervention_success")}
                  for name, value in document["conditions"].items()]
    # 显式挑选展示字段，不向前端暴露服务器路径或完整 token 激活。
    layer_rows = [{key: row[key] for key in ("layer_index", "crs", "patched_margin", "margin_delta")}
                  for row in layers["layer_results"]]
    activation_rows = [{key: row[key] for key in ("layer_index", "normalized_l2", "cosine_distance")}
                       for row in activation["layer_results"]]
    component_rows = []
    for row in components["layer_results"]:
        full = row["full_layer_reference"]
        values = {"layer": {**full, **full["generation"]}, **row["component_results"]}
        for name, value in values.items():
            component_rows.append({"layer_index": row["layer_index"], "component": name,
                                   **{key: value[key] for key in (
                                       "crs", "patched_margin", "patched_answer", "patched_answer_matches_gold",
                                       "patched_answer_matches_poison",
                                   )}})
    result = TraceResult(
        task_id=task["id"], case=task["input"]["case"],
        model={"repo_id": "Qwen/Qwen3-4B", "config_sha256": hashlib.sha256(
            (model_root / "config.json").read_bytes()).hexdigest(),
            "environment": {key: value for key, value in layers["environment"].items() if key != "model_path"}},
        input_hashes={**task["input"]["hashes"], **{key: value for key, value in alignment["prompt_files"].items()
                                                  if key.endswith("sha256")}},
        document_trace={"summary": document["summary"], "rows": conditions},
        layer_scan={"summary": layers["summary"], "rows": layer_rows},
        activation_difference={"summary": activation["summary"], "rows": activation_rows},
        component_scan={"summary": components["summary"], "rows": component_rows},
        baseline_generation=components["baseline_generation"],
    ).model_dump()
    json.dumps(result, allow_nan=False)
    return result
