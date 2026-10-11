from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.candidate_scoring import (
    _build_chat_prompt,
    _tokenize_prompt_and_candidate,
)
from core.model_runtime import MODEL_PATH
from core.prompt_builder import build_prompt


DEFAULT_CASE_IDS = [
    "CASE_HIS_028_V01",
    "CASE_HIS_041_V02",
    "CASE_MAR_031_V01",
    "CASE_MAR_059_V01",
    "CASE_NAT_030_V02",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "为正式神经因果溯源保存clean/poison Prompt并验证token对齐。"
        )
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("manual_review.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/e0_e4_knowledge_qa"),
    )
    parser.add_argument(
        "--document-trace-dir",
        type=Path,
        default=Path(
            "forensics/"
            "document_trace_results_after_scoring_unification_20260811"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("forensics/neural_trace_inputs_20260811"),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="可重复传入；省略时准备默认5个正式案例。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有输出目录。",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON文件不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_review_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"审核表不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {row["case_id"]: row for row in rows}


def require_formal_neural_approval(row: dict[str, str]) -> None:
    expected = {
        "review_status": "human_approved",
        "include_in_document": "yes",
        "include_in_neural_tracing": "yes",
    }
    for field, value in expected.items():
        actual = str(row.get(field) or "").strip().lower()
        if actual != value:
            raise RuntimeError(
                f"{row.get('case_id')}不满足神经溯源条件："
                f"{field}={actual!r}，期待{value!r}。"
            )


def validate_document_trace(
    case_id: str,
    path: Path,
) -> dict[str, Any]:
    result = load_json(path)
    if result.get("case_id") != case_id:
        raise RuntimeError(f"文档溯源结果case_id不一致：{path}")
    if result.get("formal_result") is not True:
        raise RuntimeError(f"{case_id}的formal_result不是true。")
    if (
        result.get("summary", {}).get("document_trace_success")
        is not True
    ):
        raise RuntimeError(
            f"{case_id}的document_trace_success不是true。"
        )
    return result


def validate_source_result(
    result: dict[str, Any],
    case_id: str,
    experiment_id: str,
) -> None:
    if result.get("case_id") != case_id:
        raise RuntimeError(
            f"{case_id}/{experiment_id}的case_id不一致。"
        )
    if result.get("experiment_id") != experiment_id:
        raise RuntimeError(
            f"{case_id}/{experiment_id}的experiment_id不一致。"
        )
    if result.get("status") != "success":
        raise RuntimeError(f"{case_id}/{experiment_id}不是success。")
    if not result.get("retrieved_documents"):
        raise RuntimeError(f"{case_id}/{experiment_id}没有检索文档。")


def format_document(document: dict[str, Any]) -> str:
    title = str(document.get("title") or "").strip()
    content = str(document.get("content") or "").strip()
    if not content:
        raise RuntimeError(
            f"文档{document.get('doc_id')!r}的content为空。"
        )
    if title:
        return f"标题：{title}\n正文：{content}"
    return content


def reconstruct_rag_prompt(result: dict[str, Any]) -> str:
    return build_prompt(
        [
            format_document(document)
            for document in result["retrieved_documents"]
        ],
        result["query"],
    )


def compare_documents(
    clean_documents: list[dict[str, Any]],
    poison_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(clean_documents) != len(poison_documents):
        raise RuntimeError(
            "clean与poison检索文档数量不同："
            f"{len(clean_documents)} != {len(poison_documents)}。"
        )

    comparisons: list[dict[str, Any]] = []
    differing_document_positions: list[int] = []

    for position, (clean, poison) in enumerate(
        zip(clean_documents, poison_documents)
    ):
        clean_pair = str(clean.get("pair_id") or "")
        poison_pair = str(poison.get("pair_id") or "")
        clean_title = str(clean.get("title") or "").strip()
        poison_title = str(poison.get("title") or "").strip()
        clean_content = str(clean.get("content") or "")
        poison_content = str(poison.get("content") or "")

        pair_equal = clean_pair == poison_pair and bool(clean_pair)
        title_equal = clean_title == poison_title
        content_equal = clean_content == poison_content
        if not pair_equal:
            raise RuntimeError(
                f"位置{position}的pair_id不一致："
                f"{clean_pair!r} != {poison_pair!r}。"
            )
        if not title_equal:
            raise RuntimeError(
                f"位置{position}的文档标题不一致。"
            )
        if not content_equal:
            differing_document_positions.append(position)

        matcher = difflib.SequenceMatcher(
            a=clean_content,
            b=poison_content,
            autojunk=False,
        )
        content_edits = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            content_edits.append(
                {
                    "operation": tag,
                    "clean_start": i1,
                    "clean_end": i2,
                    "clean_span": clean_content[i1:i2],
                    "poison_start": j1,
                    "poison_end": j2,
                    "poison_span": poison_content[j1:j2],
                }
            )

        comparisons.append(
            {
                "position": position,
                "clean_doc_id": clean.get("doc_id"),
                "poison_doc_id": poison.get("doc_id"),
                "pair_id": clean_pair,
                "clean_variant": clean.get("variant"),
                "poison_variant": poison.get("variant"),
                "pair_id_equal": pair_equal,
                "title_equal": title_equal,
                "content_equal": content_equal,
                "clean_content_length": len(clean_content),
                "poison_content_length": len(poison_content),
                "content_edits": content_edits,
            }
        )

    return {
        "document_count": len(clean_documents),
        "document_count_equal": True,
        "differing_document_positions": differing_document_positions,
        "only_one_document_differs": (
            len(differing_document_positions) == 1
        ),
        "documents": comparisons,
    }


def common_prefix_length(
    clean_ids: list[int],
    poison_ids: list[int],
) -> int:
    length = 0
    for clean_id, poison_id in zip(clean_ids, poison_ids):
        if clean_id != poison_id:
            break
        length += 1
    return length


def common_suffix_length(
    clean_ids: list[int],
    poison_ids: list[int],
    prefix_length: int,
) -> int:
    maximum = min(len(clean_ids), len(poison_ids)) - prefix_length
    length = 0
    while (
        length < maximum
        and clean_ids[-1 - length] == poison_ids[-1 - length]
    ):
        length += 1
    return length


def token_opcodes(
    tokenizer,
    clean_ids: list[int],
    poison_ids: list[int],
) -> list[dict[str, Any]]:
    matcher = difflib.SequenceMatcher(
        a=clean_ids,
        b=poison_ids,
        autojunk=False,
    )
    records = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        clean_span_ids = clean_ids[i1:i2]
        poison_span_ids = poison_ids[j1:j2]
        records.append(
            {
                "operation": tag,
                "clean_start": i1,
                "clean_end": i2,
                "clean_token_ids": clean_span_ids,
                "clean_tokens": tokenizer.convert_ids_to_tokens(
                    clean_span_ids
                ),
                "clean_text": tokenizer.decode(clean_span_ids),
                "poison_start": j1,
                "poison_end": j2,
                "poison_token_ids": poison_span_ids,
                "poison_tokens": tokenizer.convert_ids_to_tokens(
                    poison_span_ids
                ),
                "poison_text": tokenizer.decode(poison_span_ids),
            }
        )
    return records


def candidate_tokenization(
    tokenizer,
    clean_full_prompt: str,
    poison_full_prompt: str,
    answers: dict[str, str],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for answer_name, answer_text in answers.items():
        clean_prompt_ids, clean_candidate_ids, clean_input_ids = (
            _tokenize_prompt_and_candidate(
                tokenizer,
                clean_full_prompt,
                answer_text,
            )
        )
        poison_prompt_ids, poison_candidate_ids, poison_input_ids = (
            _tokenize_prompt_and_candidate(
                tokenizer,
                poison_full_prompt,
                answer_text,
            )
        )
        if clean_candidate_ids != poison_candidate_ids:
            raise RuntimeError(
                f"{answer_name}在clean/poison边界的candidate token不同。"
            )
        records[answer_name] = {
            "answer": answer_text,
            "candidate_token_count": len(clean_candidate_ids),
            "candidate_token_ids": clean_candidate_ids,
            "candidate_tokens": tokenizer.convert_ids_to_tokens(
                clean_candidate_ids
            ),
            "clean_prompt_token_count": len(clean_prompt_ids),
            "poison_prompt_token_count": len(poison_prompt_ids),
            "clean_input_token_count": len(clean_input_ids),
            "poison_input_token_count": len(poison_input_ids),
            "candidate_tokens_equal_across_prompts": True,
        }
    return records


def prepare_case(
    case_id: str,
    review_row: dict[str, str],
    results_dir: Path,
    document_trace_dir: Path,
    tokenizer,
) -> dict[str, Any]:
    require_formal_neural_approval(review_row)

    case_dir = results_dir / case_id
    clean_path = case_dir / "E1.json"
    poison_path = case_dir / "E2.json"
    document_trace_path = document_trace_dir / f"{case_id}.json"

    clean = load_json(clean_path)
    poison = load_json(poison_path)
    validate_source_result(clean, case_id, "E1")
    validate_source_result(poison, case_id, "E2")
    validate_document_trace(case_id, document_trace_path)

    scalar_fields = [
        "case_id",
        "pair_id",
        "query_id",
        "query",
        "gold_answer",
        "poison_target_answer",
    ]
    field_checks = {
        field: clean.get(field) == poison.get(field)
        for field in scalar_fields
    }
    if not all(field_checks.values()):
        failed = [field for field, passed in field_checks.items() if not passed]
        raise RuntimeError(
            f"{case_id}的E1/E2字段不一致：{failed}。"
        )

    clean_reconstructed = reconstruct_rag_prompt(clean)
    poison_reconstructed = reconstruct_rag_prompt(poison)
    prompt_reconstruction = {
        "clean_matches_saved_e1": (
            clean_reconstructed == clean["rag_prompt"]
        ),
        "poison_matches_saved_e2": (
            poison_reconstructed == poison["rag_prompt"]
        ),
    }
    if not all(prompt_reconstruction.values()):
        raise RuntimeError(f"{case_id}的Prompt重建与E1/E2不一致。")

    document_alignment = compare_documents(
        clean["retrieved_documents"],
        poison["retrieved_documents"],
    )
    if not document_alignment["only_one_document_differs"]:
        raise RuntimeError(
            f"{case_id}不是只有一个目标文档存在差异。"
        )

    clean_rag_prompt = clean["rag_prompt"]
    poison_rag_prompt = poison["rag_prompt"]
    clean_full_prompt = _build_chat_prompt(tokenizer, clean_rag_prompt)
    poison_full_prompt = _build_chat_prompt(tokenizer, poison_rag_prompt)

    clean_token_ids = tokenizer(
        clean_full_prompt,
        add_special_tokens=False,
    )["input_ids"]
    poison_token_ids = tokenizer(
        poison_full_prompt,
        add_special_tokens=False,
    )["input_ids"]
    if not clean_token_ids or not poison_token_ids:
        raise RuntimeError(f"{case_id}的Prompt没有有效token。")

    prefix_length = common_prefix_length(
        clean_token_ids,
        poison_token_ids,
    )
    suffix_length = common_suffix_length(
        clean_token_ids,
        poison_token_ids,
        prefix_length,
    )
    counts_equal = len(clean_token_ids) == len(poison_token_ids)
    positionwise_differences = []
    if counts_equal:
        positionwise_differences = [
            {
                "position": position,
                "clean_token_id": clean_id,
                "clean_token": tokenizer.convert_ids_to_tokens(
                    [clean_id]
                )[0],
                "poison_token_id": poison_id,
                "poison_token": tokenizer.convert_ids_to_tokens(
                    [poison_id]
                )[0],
            }
            for position, (clean_id, poison_id) in enumerate(
                zip(clean_token_ids, poison_token_ids)
            )
            if clean_id != poison_id
        ]

    token_differences = token_opcodes(
        tokenizer,
        clean_token_ids,
        poison_token_ids,
    )
    last_token_equal = clean_token_ids[-1] == poison_token_ids[-1]

    candidates = candidate_tokenization(
        tokenizer,
        clean_full_prompt,
        poison_full_prompt,
        {
            "gold_answer": clean["gold_answer"],
            "poison_target_answer": clean["poison_target_answer"],
        },
    )

    alignment_valid = all(
        [
            all(field_checks.values()),
            all(prompt_reconstruction.values()),
            document_alignment["document_count_equal"],
            document_alignment["only_one_document_differs"],
            counts_equal,
            last_token_equal,
            bool(token_differences),
        ]
    )
    if not alignment_valid:
        raise RuntimeError(f"{case_id}的token alignment未通过。")

    return {
        "schema_version": "neural-trace-input-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "case_id": case_id,
        "pair_id": clean["pair_id"],
        "query_id": clean["query_id"],
        "query": clean["query"],
        "gold_answer": clean["gold_answer"],
        "poison_target_answer": clean["poison_target_answer"],
        "review_snapshot": {
            "review_status": review_row["review_status"],
            "include_in_document": review_row["include_in_document"],
            "include_in_neural_tracing": review_row[
                "include_in_neural_tracing"
            ],
            "reviewer": review_row["reviewer"],
            "review_date": review_row["review_date"],
        },
        "source_files": {
            "clean_E1": str(clean_path.resolve()),
            "clean_E1_sha256": sha256_file(clean_path),
            "poison_E2": str(poison_path.resolve()),
            "poison_E2_sha256": sha256_file(poison_path),
            "document_trace": str(document_trace_path.resolve()),
            "document_trace_sha256": sha256_file(document_trace_path),
            "candidate_scoring": str(
                (PROJECT_ROOT / "core/candidate_scoring.py").resolve()
            ),
            "candidate_scoring_sha256": sha256_file(
                PROJECT_ROOT / "core/candidate_scoring.py"
            ),
        },
        "prompt_files": {
            "clean_prompt": "clean_prompt.txt",
            "clean_prompt_sha256": sha256_text(clean_rag_prompt),
            "poison_prompt": "poison_prompt.txt",
            "poison_prompt_sha256": sha256_text(poison_rag_prompt),
            "stored_prompt_type": "raw_rag_prompt_before_chat_template",
            "tokenization_basis": (
                "candidate_scoring._build_chat_prompt output"
            ),
        },
        "validation": {
            "formal_document_trace_passed": True,
            "field_checks": field_checks,
            "prompt_reconstruction": prompt_reconstruction,
            "alignment_valid": alignment_valid,
        },
        "document_alignment": document_alignment,
        "token_alignment": {
            "model_path": MODEL_PATH,
            "clean_full_prompt_sha256": sha256_text(clean_full_prompt),
            "poison_full_prompt_sha256": sha256_text(poison_full_prompt),
            "clean_token_count": len(clean_token_ids),
            "poison_token_count": len(poison_token_ids),
            "token_counts_equal": counts_equal,
            "clean_token_ids": clean_token_ids,
            "poison_token_ids": poison_token_ids,
            "clean_tokens": tokenizer.convert_ids_to_tokens(
                clean_token_ids
            ),
            "poison_tokens": tokenizer.convert_ids_to_tokens(
                poison_token_ids
            ),
            "common_prefix_token_count": prefix_length,
            "common_suffix_token_count": suffix_length,
            "positionwise_difference_count": len(
                positionwise_differences
            ),
            "positionwise_differences": positionwise_differences,
            "difference_opcodes": token_differences,
            "last_prompt_token_index": len(clean_token_ids) - 1,
            "clean_last_prompt_token_id": clean_token_ids[-1],
            "poison_last_prompt_token_id": poison_token_ids[-1],
            "last_prompt_token_equal": last_token_equal,
            "candidate_tokenization": candidates,
            "alignment_valid": alignment_valid,
        },
        "clean_rag_prompt": clean_rag_prompt,
        "poison_rag_prompt": poison_rag_prompt,
    }


def write_outputs(
    prepared_cases: list[dict[str, Any]],
    output_dir: Path,
    overwrite: bool,
) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"输出目录已存在：{output_dir}。"
            "如确认覆盖请显式增加--overwrite。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_cases = []
    for prepared in prepared_cases:
        case_dir = output_dir / prepared["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)

        (case_dir / "clean_prompt.txt").write_text(
            prepared["clean_rag_prompt"],
            encoding="utf-8",
            newline="\n",
        )
        (case_dir / "poison_prompt.txt").write_text(
            prepared["poison_rag_prompt"],
            encoding="utf-8",
            newline="\n",
        )

        json_payload = dict(prepared)
        json_payload.pop("clean_rag_prompt")
        json_payload.pop("poison_rag_prompt")
        (case_dir / "token_alignment.json").write_text(
            json.dumps(json_payload, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

        alignment = prepared["token_alignment"]
        summary_cases.append(
            {
                "case_id": prepared["case_id"],
                "clean_token_count": alignment["clean_token_count"],
                "poison_token_count": alignment[
                    "poison_token_count"
                ],
                "positionwise_difference_count": alignment[
                    "positionwise_difference_count"
                ],
                "common_prefix_token_count": alignment[
                    "common_prefix_token_count"
                ],
                "common_suffix_token_count": alignment[
                    "common_suffix_token_count"
                ],
                "last_prompt_token_index": alignment[
                    "last_prompt_token_index"
                ],
                "alignment_valid": alignment["alignment_valid"],
            }
        )

    summary = {
        "schema_version": "neural-trace-input-summary-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "case_count": len(prepared_cases),
        "all_alignments_valid": all(
            item["alignment_valid"] for item in summary_cases
        ),
        "cases": summary_cases,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    case_ids = args.case_ids or DEFAULT_CASE_IDS
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("--case-id中存在重复案例。")

    review_rows = load_review_rows(args.review_csv)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    prepared_cases = []
    for index, case_id in enumerate(case_ids, start=1):
        print(f"[{index}/{len(case_ids)}] 检查{case_id}……")
        if case_id not in review_rows:
            raise KeyError(f"审核表中找不到案例：{case_id}")
        prepared = prepare_case(
            case_id,
            review_rows[case_id],
            args.results_dir,
            args.document_trace_dir,
            tokenizer,
        )
        prepared_cases.append(prepared)
        alignment = prepared["token_alignment"]
        print(
            "  token数："
            f"{alignment['clean_token_count']}/"
            f"{alignment['poison_token_count']}；"
            "位置差异："
            f"{alignment['positionwise_difference_count']}；"
            "alignment_valid=True"
        )

    write_outputs(prepared_cases, args.output_dir, args.overwrite)
    print("\n===== Neural Trace Input 汇总 =====")
    print(f"案例数：{len(prepared_cases)}")
    print("全部alignment_valid：True")
    print(f"结果目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
