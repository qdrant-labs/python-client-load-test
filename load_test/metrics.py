from typing import List

import numpy as np


def compute_recall(ann_results: List[List], exact_results: List[List]) -> List[float]:
    recalls = []
    for ann, exact in zip(ann_results, exact_results):
        if not exact:
            continue
        ann_set = set(map(str, ann))
        exact_set = set(map(str, exact))
        recalls.append(len(ann_set & exact_set) / len(exact_set))
    return recalls


def compute_metric(recalls: List[float], metric: str) -> float:
    if not recalls:
        return 0.0
    pmap = {"p01": 1, "p05": 5, "p10": 10, "p50": 50, "p95": 95, "p99": 99}
    if metric in pmap:
        return float(np.percentile(recalls, pmap[metric]))
    if metric == "mean":
        return float(np.mean(recalls))
    raise ValueError(f"Unknown metric {metric!r}. Use: p01 p05 p10 p50 p95 p99 mean")


def recall_histogram_data(recalls: List[float]) -> dict:
    """Compute 100-bucket histogram + percentile bucket indices for annotation lines."""
    counts = [0] * 100
    for r in recalls:
        counts[min(int(r * 100), 99)] += 1
    p01 = compute_metric(recalls, "p01")
    p05 = compute_metric(recalls, "p05")
    p10 = compute_metric(recalls, "p10")
    p50 = compute_metric(recalls, "p50")
    return {
        "counts": counts,
        "p01Bucket": min(int(p01 * 100), 99),
        "p05Bucket": min(int(p05 * 100), 99),
        "p10Bucket": min(int(p10 * 100), 99),
        "p50Bucket": min(int(p50 * 100), 99),
        "p01Recall": round(p01, 4),
        "p05Recall": round(p05, 4),
        "p10Recall": round(p10, 4),
        "p50Recall": round(p50, 4),
    }
