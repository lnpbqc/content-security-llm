"""由逐样本模型估计汇总多语种语义权重和风险分层。"""

from itertools import combinations
from math import log2


LEVELS = ("safe", "low", "medium", "high")


def risk_level(score, thresholds):
    if score < thresholds["low"]:
        return "safe"
    if score < thresholds["medium"]:
        return "low"
    if score < thresholds["high"]:
        return "medium"
    return "high"


def _similarity(left, right):
    midpoint = [(a + b) / 2 for a, b in zip(left, right)]

    def divergence(values):
        return sum(value * log2(value / middle) for value, middle in zip(values, midpoint) if value)

    return 1 - (divergence(left) + divergence(right)) / 2


def build_snapshot(base, policy, samples, outcomes, mask_text, previous=None):
    rows = []
    failed = unassessable = 0
    for sample in samples:
        outcome = outcomes[sample["id"]]
        if outcome["status"] == "error":
            failed += 1
            continue
        output = outcome["output"]
        row = {"id": sample["id"], "dataset_id": sample["dataset_id"],
               "version_id": sample["version_id"], "revision_id": sample["revision_id"],
               "text": mask_text(sample["text"]), "language": sample["language"],
               "status": output["status"], "semantic_weights": output["semanticWeights"],
               "risk_score": output["riskScore"], "risk_level": "unknown",
               "evidence": output["evidence"],
               **({"reference_risk_level": sample["metadata"]["reference_risk_level"]}
                  if "reference_risk_level" in sample["metadata"] else {})}
        if output["status"] == "assessed":
            row["risk_level"] = risk_level(output["riskScore"], policy["thresholds"])
        else:
            unassessable += 1
            row["unassessable_reason"] = output["unassessableReason"]
        rows.append(row)

    valid = [row for row in rows if row["status"] == "assessed"]
    by_language = {}
    for row in valid:
        by_language.setdefault(row["language"], []).append(row)
    vectors = {}
    language_distribution = []
    for language, group in sorted(by_language.items()):
        vector = [sum(row["semantic_weights"][key] for row in group) / len(group)
                  for key in ("safe", "neutral", "risk")]
        vector = [value / sum(vector) for value in vector]
        vectors[language] = vector
        tenths = [round(value * 1000) for value in vector]
        tenths[tenths.index(max(tenths))] += 1000 - sum(tenths)
        language_distribution.append({"language": language, "sample_count": len(group),
                                      "safe_weight": tenths[0] / 10,
                                      "neutral_weight": tenths[1] / 10,
                                      "risk_weight": tenths[2] / 10})
    comparisons = [_similarity(a, b) for a, b in combinations(vectors.values(), 2)]
    consistency = round(sum(comparisons) / len(comparisons), 2) if comparisons else None
    mean_score = sum(row["risk_score"] for row in valid) / len(valid) if valid else None
    high_weight = round(sum(row["semantic_weights"]["risk"] for row in valid) /
                        len(valid) * 100, 1) if valid else None
    distribution = [{"level": level,
                     "count": sum(row["risk_level"] == level for row in valid),
                     "percent": round(sum(row["risk_level"] == level for row in valid) /
                                      len(valid) * 100, 1) if valid else 0.0}
                    for level in LEVELS]
    evidence = [{"sample_id": row["id"], "language": row["language"],
                 "quote": item["quote"], "reason": item["reason"],
                 "semantic_weight": round(row["semantic_weights"]["risk"], 2),
                 "risk_score": round(row["risk_score"], 2), "risk_level": row["risk_level"]}
                for row in valid for item in row["evidence"]]
    evidence.sort(key=lambda item: (-item["semantic_weight"], -item["risk_score"], item["sample_id"]))
    labeled = [sample for sample in samples if "reference_risk_level" in sample["metadata"]]
    predicted = {row["id"]: row["risk_level"] for row in rows}
    correct = sum(predicted.get(sample["id"]) == sample["metadata"]["reference_risk_level"]
                  for sample in labeled)
    summary = {**base, "data_source": "business_dataset", "scheme_name": policy["name"],
               "thresholds": policy["thresholds"],
               "target_count": len(samples), "valid_count": len(valid),
               "unassessable_count": unassessable, "failed_count": failed,
               "accuracy_correct_count": correct, "accuracy_test_count": len(labeled),
               "data_weight_classification_accuracy": (round(correct / len(labeled), 4)
                                                       if labeled else None),
               "overall_risk_score": round(mean_score, 2) if mean_score is not None else None,
               "overall_risk_level": (risk_level(mean_score, policy["thresholds"])
                                      if mean_score is not None else "unknown"),
               "semantic_consistency": consistency,
               "semantic_consistency_method": "pairwise_jensen_shannon_similarity",
               "high_risk_semantic_weight": high_weight,
               "covered_language_count": len(by_language),
               "language_distribution": language_distribution,
               "risk_distribution": distribution, "risk_evidence": evidence[:20],
               "semantic_consistency_delta": (round(consistency - previous["semantic_consistency"], 2)
                                              if consistency is not None and previous and
                                              previous.get("semantic_consistency") is not None else None),
               "high_risk_semantic_weight_delta": (round(high_weight - previous["high_risk_semantic_weight"], 1)
                                                   if high_weight is not None and previous and
                                                   previous.get("high_risk_semantic_weight") is not None else None)}
    return summary, rows
