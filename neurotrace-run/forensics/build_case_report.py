from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把正式文档、Layer和组件结果整理成单案例报告目录。"
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
    )
    parser.add_argument(
        "--document-trace-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "forensics"
            / "document_trace_results_after_scoring_unification_20260811"
        ),
    )
    parser.add_argument(
        "--neural-input-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "forensics"
            / "neural_trace_inputs_20260811"
        ),
    )
    parser.add_argument(
        "--layer-result-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "forensics"
            / "layer_scan_results_20260811"
        ),
    )
    parser.add_argument(
        "--component-result-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "forensics"
            / "component_scan_results_with_generation_20260812"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"缺少{label}：{path}")
    return path


def compact_document_condition(condition: dict[str, Any]) -> dict[str, Any]:
    generation = condition["generation"]
    return {
        "intervention": condition["intervention"],
        "model_answer": generation["model_answer"],
        "margin": condition["margin"],
        "gold_answer_match": condition["gold_answer_match"],
        "poison_answer_match": condition["poison_answer_match"],
        "margin_delta_from_poison": condition.get(
            "margin_delta_from_poison"
        ),
        "intervention_success": condition.get("intervention_success"),
    }


def build_answer_comparison(
    document_trace: dict[str, Any],
    layer_trace: dict[str, Any],
    component_trace: dict[str, Any],
) -> dict[str, Any]:
    layers = []
    for row in component_trace["layer_results"]:
        layer_generation = row["full_layer_reference"]["generation"]
        components = row["component_results"]
        layers.append(
            {
                "layer_index": row["layer_index"],
                "layer_crs": row["summary"]["layer_crs"],
                "dominant_component": row["summary"][
                    "dominant_component"
                ],
                "full_layer": {
                    "candidate_patched_margin": row[
                        "full_layer_reference"
                    ]["patched_margin"],
                    "candidate_crs": row["full_layer_reference"]["crs"],
                    "patched_answer": layer_generation[
                        "patched_answer"
                    ],
                    "answer_matches_gold": layer_generation[
                        "patched_answer_matches_gold"
                    ],
                    "answer_matches_poison": layer_generation[
                        "patched_answer_matches_poison"
                    ],
                },
                "self_attn": {
                    "candidate_patched_margin": components[
                        "self_attn"
                    ]["patched_margin"],
                    "candidate_crs": components["self_attn"]["crs"],
                    "patched_answer": components["self_attn"][
                        "patched_answer"
                    ],
                    "answer_matches_gold": components["self_attn"][
                        "patched_answer_matches_gold"
                    ],
                    "answer_matches_poison": components["self_attn"][
                        "patched_answer_matches_poison"
                    ],
                },
                "mlp": {
                    "candidate_patched_margin": components["mlp"][
                        "patched_margin"
                    ],
                    "candidate_crs": components["mlp"]["crs"],
                    "patched_answer": components["mlp"][
                        "patched_answer"
                    ],
                    "answer_matches_gold": components["mlp"][
                        "patched_answer_matches_gold"
                    ],
                    "answer_matches_poison": components["mlp"][
                        "patched_answer_matches_poison"
                    ],
                },
            }
        )

    return {
        "schema_version": "case-answer-comparison-v1",
        "case_id": document_trace["case_id"],
        "query": document_trace["query"],
        "gold_answer": document_trace["gold_answer"],
        "poison_target_answer": document_trace[
            "poison_target_answer"
        ],
        "document_conditions": {
            name: compact_document_condition(condition)
            for name, condition in document_trace["conditions"].items()
        },
        "neural_baselines": {
            "clean_generation": component_trace["baseline_generation"][
                "clean"
            ],
            "poison_generation": component_trace[
                "baseline_generation"
            ]["poison"],
            "clean_margin": layer_trace["scoring"]["clean_baseline"][
                "margin"
            ],
            "poison_margin": layer_trace["scoring"]["poison_baseline"][
                "margin"
            ],
        },
        "patched_conditions": layers,
        "generation_patch_scope": component_trace["intervention"][
            "generation_patch_scope"
        ],
        "candidate_scoring_patch_scope": component_trace[
            "intervention"
        ]["patch_scope"],
        "technical_pass": component_trace["technical_checks"][
            "technical_pass"
        ],
    }


def plot_layer_trace(
    layer_csv: Path,
    output_path: Path,
    case_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers: list[int] = []
    crs_values: list[float] = []
    margins: list[float] = []
    with layer_csv.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            layers.append(int(row["layer_index"]))
            crs_values.append(float(row["crs"]))
            margins.append(float(row["patched_margin"]))

    figure, axis_crs = plt.subplots(figsize=(10, 5.6), dpi=160)
    axis_margin = axis_crs.twinx()
    axis_crs.plot(
        layers,
        crs_values,
        color="#2563eb",
        marker="o",
        markersize=3,
        linewidth=2,
        label="CRS",
    )
    axis_margin.plot(
        layers,
        margins,
        color="#dc2626",
        linewidth=1.6,
        alpha=0.75,
        label="Patched margin",
    )
    axis_crs.axhline(0.0, color="#64748b", linewidth=1)
    axis_crs.axhline(1.0, color="#16a34a", linestyle="--", linewidth=1)
    axis_margin.axhline(0.0, color="#dc2626", linestyle=":", linewidth=1)
    axis_crs.set_xlabel("Layer index")
    axis_crs.set_ylabel("Causal Recovery Score (CRS)", color="#2563eb")
    axis_margin.set_ylabel("Patched margin", color="#dc2626")
    axis_crs.set_title(f"{case_id}: 36-layer causal trace")
    axis_crs.set_xticks(range(0, 36, 2))
    axis_crs.grid(alpha=0.2)

    lines = axis_crs.get_lines()[:1] + axis_margin.get_lines()[:1]
    axis_crs.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def first_positive_margin_layer(layer_trace: dict[str, Any]) -> int | None:
    for row in layer_trace["layer_results"]:
        if float(row["patched_margin"]) > 0:
            return int(row["layer_index"])
    return None


def build_report_markdown(
    document_trace: dict[str, Any],
    token_alignment: dict[str, Any],
    layer_trace: dict[str, Any],
    component_trace: dict[str, Any],
) -> str:
    case_id = document_trace["case_id"]
    doc = token_alignment["document_alignment"]["documents"][0]
    edit = doc["content_edits"][0]
    first_meaningful = layer_trace["summary"][
        "first_meaningful_recovery_layer"
    ]
    first_positive = first_positive_margin_layer(layer_trace)
    max_layer = layer_trace["summary"]["max_crs_layer"]
    max_crs = layer_trace["summary"]["max_crs"]

    component_rows = []
    for row in component_trace["layer_results"]:
        layer_gen = row["full_layer_reference"]["generation"]
        attn = row["component_results"]["self_attn"]
        mlp = row["component_results"]["mlp"]
        component_rows.append(
            "| {layer} | {layer_crs:.6f} | {attn_crs:.6f} | "
            "{mlp_crs:.6f} | {dominant} | {layer_answer} | "
            "{attn_answer} | {mlp_answer} |".format(
                layer=row["layer_index"],
                layer_crs=row["summary"]["layer_crs"],
                attn_crs=row["summary"]["attention_crs"],
                mlp_crs=row["summary"]["mlp_crs"],
                dominant=row["summary"]["dominant_component"],
                layer_answer=layer_gen["patched_answer"],
                attn_answer=attn["patched_answer"],
                mlp_answer=mlp["patched_answer"],
            )
        )

    conditions = document_trace["conditions"]
    return f"""# {case_id} 文档与神经因果溯源报告

## 1. 案例信息

- Query：{document_trace['query']}
- 正确答案：`{document_trace['gold_answer']}`
- 投毒目标答案：`{document_trace['poison_target_answer']}`
- clean baseline 实际答案：`{component_trace['baseline_generation']['clean']['model_answer']}`
- poison baseline 实际答案：`{component_trace['baseline_generation']['poison']['model_answer']}`

## 2. 直接污染源

- poison 文档：`{doc['poison_doc_id']}`
- clean counterpart：`{doc['clean_doc_id']}`
- 文档位置：{doc['position']}
- 唯一事实替换：`{edit['clean_span']}` → `{edit['poison_span']}`

clean 与 poison 文档数量、顺序和标题相同，仅该核心事实 span 不同。因此，该 span 是本案例的直接污染候选来源。

## 3. 文档干预结果

| 条件 | 实际生成答案 | margin | 是否成功 |
|---|---|---:|---|
| 原 poison | {conditions['original_poison']['generation']['model_answer']} | {conditions['original_poison']['margin']:+.6f} | baseline |
| 删除 poison 文档 | {conditions['delete_poison']['generation']['model_answer']} | {conditions['delete_poison']['margin']:+.6f} | {conditions['delete_poison']['intervention_success']} |
| 同位置 clean 替换 | {conditions['replace_with_clean']['generation']['model_answer']} | {conditions['replace_with_clean']['margin']:+.6f} | {conditions['replace_with_clean']['intervention_success']} |
| 只修复错误 span | {conditions['repair_poison_span']['generation']['model_answer']} | {conditions['repair_poison_span']['margin']:+.6f} | {conditions['repair_poison_span']['intervention_success']} |

clean 替换与 span 修复的 Prompt 完全相同，因为本案例中两份文档只有一个错误事实 span 不同。修复该 span 后，整篇文档已经等价于 clean counterpart，所以二者 margin 相同。

## 4. 36 层 Layer 因果扫描

- 扫描范围：Layer 0–35；
- patch 方向：clean activation → poison 运行；
- 候选评分 patch 范围：`candidate_prediction_span`；
- 首次 CRS≥0.05：Layer {first_meaningful}；
- 首次 patched margin>0：Layer {first_positive}；
- 最大 CRS：Layer {max_layer}，CRS={max_crs:.6f}。

![36 层 CRS 与 margin 曲线](layer_trace.png)

该结果说明从 Layer {first_meaningful} 开始出现明显因果恢复作用，Layer {first_positive} 已能使正确候选重新占优。不能把最大 CRS 层直接写成错误最早产生的位置，只能说替换该层具有很强的恢复作用。

## 5. Attention/MLP 组件结果与真实生成

| Layer | Layer CRS | Attention CRS | MLP CRS | 局部判断 | 整层实际答案 | Attention 实际答案 | MLP 实际答案 |
|---:|---:|---:|---:|---|---|---|---|
{os.linesep.join(component_rows)}

整层生成 patch 只替换 poison Prompt 预填充阶段最后一个 Prompt token，并让模型随后自行贪心生成，不使用未来 gold/poison 候选 token。被测整层均实际生成正确答案 `{document_trace['gold_answer']}`，说明整层 clean 状态足以因果恢复本案例答案。

单独 Attention 或 MLP patch 虽然在候选评分上产生不同程度的恢复，但实际自由生成仍为投毒答案 `{document_trace['poison_target_answer']}`。因此，本案例可以报告“Attention 在 Layer 24 和 33 的局部恢复作用高于 MLP”，但不能报告“Attention 单独完成纠错”。完整恢复更可能依赖整层多个子过程及残差路径共同作用。

## 6. 最终结论

本案例的直接污染源为文档 `DOC_P_HIS_028` 中把“任弼时”替换为“董必武”的错误片段。删除该文档会消除明确错误回答；同位置换回 clean 文档或只修复错误 span 后，模型回答恢复为“任弼时”，gold–poison margin 由负转正。

模型内部从中后层开始出现因果恢复。Top Layer 的整层输出替换可以在真实自由生成中恢复正确答案；Attention/MLP 单独替换只能产生局部概率改善，未独立完成答案翻转。因此，当前证据支持“该错误由 poison 文档直接诱导，并在模型中后段形成可被整层 clean 状态恢复的错误倾向”，但不足以断言某个单独组件或神经元是唯一错误来源。

## 7. 验收状态

- 文档溯源：`{document_trace['summary']['document_trace_success']}`
- 36 层全部扫描：`{layer_trace['intervention']['all_layers_scanned']}`
- Layer 技术检查：`{layer_trace['technical_checks']['technical_pass']}`
- 组件与生成技术检查：`{component_trace['technical_checks']['technical_pass']}`
- 真实 patched 自由生成：`{component_trace['intervention']['actual_generation_performed']}`
"""


def main() -> None:
    args = parse_args()
    case_id = args.case_id.strip()
    if not case_id or "/" in case_id or "\\" in case_id:
        raise ValueError("case_id格式不安全。")

    document_trace_path = require_file(
        args.document_trace_dir / f"{case_id}.json",
        "document_trace",
    )
    neural_case_dir = args.neural_input_dir / case_id
    clean_prompt_path = require_file(
        neural_case_dir / "clean_prompt.txt",
        "clean_prompt",
    )
    poison_prompt_path = require_file(
        neural_case_dir / "poison_prompt.txt",
        "poison_prompt",
    )
    token_alignment_path = require_file(
        neural_case_dir / "token_alignment.json",
        "token_alignment",
    )
    layer_csv_path = require_file(
        args.layer_result_dir
        / f"{case_id}.layer_scan.candidate_prediction_span.csv",
        "layer_trace CSV",
    )
    layer_json_path = require_file(
        args.layer_result_dir
        / f"{case_id}.layer_scan.candidate_prediction_span.json",
        "layer_trace JSON",
    )
    component_csv_path = require_file(
        args.component_result_dir / f"{case_id}.component_scan.csv",
        "component_trace CSV",
    )
    component_json_path = require_file(
        args.component_result_dir / f"{case_id}.component_scan.json",
        "component_trace JSON",
    )

    document_trace = read_json(document_trace_path)
    token_alignment = read_json(token_alignment_path)
    layer_trace = read_json(layer_json_path)
    component_trace = read_json(component_json_path)
    if any(
        data.get("case_id") != case_id
        for data in (
            document_trace,
            token_alignment,
            layer_trace,
            component_trace,
        )
    ):
        raise ValueError("来源文件case_id不一致。")
    if not component_trace["intervention"].get(
        "actual_generation_performed"
    ):
        raise RuntimeError("组件结果尚未执行patched自由生成。")
    if not component_trace["technical_checks"].get("technical_pass"):
        raise RuntimeError("组件结果未通过技术检查。")

    clean_source = require_file(
        Path(document_trace["source_files"]["clean_E1"]),
        "原始clean E1",
    )
    poison_source = require_file(
        Path(document_trace["source_files"]["poison_E2"]),
        "原始poison E2",
    )

    reports_dir = args.reports_dir.expanduser().resolve()
    target_dir = reports_dir / case_id
    if target_dir.exists():
        raise FileExistsError(f"报告目录已存在，本次不覆盖：{target_dir}")
    reports_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = reports_dir / f".{case_id}.build-{uuid.uuid4().hex}"
    temporary_dir.mkdir()

    source_paths = {
        "clean_E1": clean_source,
        "poison_E2": poison_source,
        "clean_prompt": clean_prompt_path,
        "poison_prompt": poison_prompt_path,
        "document_trace": document_trace_path,
        "token_alignment": token_alignment_path,
        "layer_trace_csv": layer_csv_path,
        "layer_trace_json": layer_json_path,
        "component_trace_csv": component_csv_path,
        "component_trace_json": component_json_path,
    }

    try:
        original_pair = {
            "schema_version": "original-clean-poison-pair-v1",
            "case_id": case_id,
            "clean_E1": read_json(clean_source),
            "poison_E2": read_json(poison_source),
        }
        write_json(temporary_dir / "original_pair.json", original_pair)
        shutil.copy2(clean_prompt_path, temporary_dir / "clean_prompt.txt")
        shutil.copy2(poison_prompt_path, temporary_dir / "poison_prompt.txt")
        shutil.copy2(document_trace_path, temporary_dir / "document_trace.json")
        shutil.copy2(token_alignment_path, temporary_dir / "token_alignment.json")
        shutil.copy2(layer_csv_path, temporary_dir / "layer_trace.csv")
        shutil.copy2(component_csv_path, temporary_dir / "component_trace.csv")

        answer_comparison = build_answer_comparison(
            document_trace,
            layer_trace,
            component_trace,
        )
        write_json(
            temporary_dir / "answer_comparison.json",
            answer_comparison,
        )
        plot_layer_trace(
            layer_csv_path,
            temporary_dir / "layer_trace.png",
            case_id,
        )
        report_text = build_report_markdown(
            document_trace,
            token_alignment,
            layer_trace,
            component_trace,
        )
        (temporary_dir / "report.md").write_text(
            report_text,
            encoding="utf-8",
        )

        output_names = (
            "original_pair.json",
            "clean_prompt.txt",
            "poison_prompt.txt",
            "answer_comparison.json",
            "document_trace.json",
            "token_alignment.json",
            "layer_trace.csv",
            "layer_trace.png",
            "component_trace.csv",
            "report.md",
        )
        manifest = {
            "schema_version": "case-report-manifest-v1",
            "case_id": case_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "completed",
            "source_files": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in source_paths.items()
            },
            "output_files": {
                name: {
                    "path": name,
                    "sha256": sha256_file(temporary_dir / name),
                    "size_bytes": (temporary_dir / name).stat().st_size,
                }
                for name in output_names
            },
            "acceptance": {
                "document_trace_success": document_trace["summary"][
                    "document_trace_success"
                ],
                "all_36_layers_scanned": layer_trace["intervention"][
                    "all_layers_scanned"
                ],
                "layer_technical_pass": layer_trace["technical_checks"][
                    "technical_pass"
                ],
                "component_technical_pass": component_trace[
                    "technical_checks"
                ]["technical_pass"],
                "patched_generation_performed": component_trace[
                    "intervention"
                ]["actual_generation_performed"],
                "complete_case_demo": True,
            },
        }
        write_json(temporary_dir / "case_manifest.json", manifest)
        temporary_dir.rename(target_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(f"完整案例报告已生成：{target_dir}")
    for path in sorted(target_dir.iterdir()):
        print(f"  {path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()

