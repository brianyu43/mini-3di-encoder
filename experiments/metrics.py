"""Query retrieval metrics with explicit relevance denominators and deterministic ties."""

from collections import defaultdict

import numpy as np


def query_metrics(query_id, scores, relevant):
    if not relevant:
        raise ValueError("Every evaluated query must have at least one relevant target")
    if query_id in relevant:
        raise ValueError("Self hits must not be relevant")
    ranked = sorted(scores, key=lambda row: (-row["raw_score"], row["target_id"]))
    if len({r["target_id"] for r in ranked}) != len(ranked):
        raise ValueError("Duplicate retrieval targets")
    positives, precision_sum = 0, 0.0
    for rank, row in enumerate(ranked, 1):
        if row["target_id"] in relevant:
            positives += 1
            precision_sum += positives / rank
    result = {
        "query_id": query_id,
        "relevant_count": len(relevant),
        "positive_score_targets": len(ranked),
        "AP": precision_sum / len(relevant),
        "top1": float(bool(ranked) and ranked[0]["target_id"] in relevant),
        "top_score_tied": bool(
            len(ranked) > 1 and ranked[0]["raw_score"] == ranked[1]["raw_score"]
        ),
    }
    for k in (1, 5, 10):
        result[f"recall_at_{k}"] = sum(r["target_id"] in relevant for r in ranked[:k]) / len(
            relevant
        )
    # Expected AP under uniform random ordering inside each exact-score tie group.
    groups = defaultdict(list)
    for row in ranked:
        groups[row["raw_score"]].append(row["target_id"] in relevant)
    before_n = before_p = 0
    expected = 0.0
    for labels in groups.values():
        n, p = len(labels), sum(labels)
        for k in range(1, n + 1):
            preceding = (k - 1) * (p - 1) / (n - 1) if n > 1 and p else 0.0
            expected += (p / n) * (before_p + 1 + preceding) / (before_n + k)
        before_n += n
        before_p += p
    result["tie_expected_AP"] = expected / len(relevant)
    return result


def retrieval_metrics(scores, queries, targets):
    rows = []
    for query in queries:
        relevant = {r["record_id"] for r in targets if r["superfamily"] == query["superfamily"]}
        row = query_metrics(
            query["record_id"], [s for s in scores if s["query_id"] == query["record_id"]], relevant
        )
        rows.append({**row, "superfamily": query["superfamily"], "fold": query["fold"]})
    values = {
        "MAP": float(np.mean([r["AP"] for r in rows])),
        "tie_expected_MAP": float(np.mean([r["tie_expected_AP"] for r in rows])),
        "top1_rate": float(np.mean([r["top1"] for r in rows])),
        "top_score_tied_queries": sum(r["top_score_tied"] for r in rows),
    }
    for k in (1, 5, 10):
        values[f"recall_at_{k}"] = float(np.mean([r[f"recall_at_{k}"] for r in rows]))
    return {"summary": values, "per_query": rows}


def grouped_bootstrap(values, *, field="AP", seed=20260929, repeats=2000):
    groups = defaultdict(list)
    for row in values:
        groups[row["superfamily"]].append(row[field])
    means = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(0, len(means), size=(repeats, len(means)))].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "groups": len(means),
        "resamples": repeats,
        "scope": "Exploratory percentile bootstrap over superfamilies; fixed trained model",
    }
