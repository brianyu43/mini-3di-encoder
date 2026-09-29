"""Stratified retrieval with explicit hard negatives and fold-cluster uncertainty."""

from collections import defaultdict

import numpy as np

from experiments.metrics import query_metrics

VIEWS = ("all", "cross_family", "same_fold", "other_fold")


def eligible_targets(query, targets, view):
    if view == "all":
        return targets
    if view == "cross_family":
        return [t for t in targets if t["family"] != query["family"]]
    if view == "same_fold":
        return [t for t in targets if t["fold"] == query["fold"]]
    if view == "other_fold":
        return [
            t
            for t in targets
            if t["superfamily"] == query["superfamily"] or t["fold"] != query["fold"]
        ]
    raise ValueError(view)


def cluster_interval(rows, field="AP", repeats=2000, seed=20260929):
    groups = defaultdict(list)
    for row in rows:
        groups[row["fold"]].append(row[field])
    if not groups:
        return None
    values = [groups[k] for k in sorted(groups)]
    totals = np.array([sum(v) for v in values])
    counts = np.array([len(v) for v in values])
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(repeats, len(values)))
    means = totals[sample].sum(1) / counts[sample].sum(1)
    return {
        "mean": float(totals.sum() / counts.sum()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "folds": len(groups),
        "queries": len(rows),
        "resamples": repeats,
        "scope": "Resample whole folds; query-weighted mean; fixed models",
    }


def evaluate(scores, queries, targets):
    by_query = defaultdict(list)
    for score in scores:
        by_query[score["query_id"]].append(score)
    outputs = {}
    for view in VIEWS:
        metrics, curves = [], []
        for query in queries:
            eligible = eligible_targets(query, targets, view)
            ids = {r["record_id"] for r in eligible}
            positives = {
                r["record_id"] for r in eligible if r["superfamily"] == query["superfamily"]
            }
            if not positives:
                raise ValueError("Every view must retain at least one relevant target")
            ranked = sorted(
                (s for s in by_query[query["record_id"]] if s["target_id"] in ids),
                key=lambda s: (-s["raw_score"], s["target_id"]),
            )
            row = query_metrics(query["record_id"], ranked, positives)
            row.update(
                {
                    "fold": query["fold"],
                    "superfamily": query["superfamily"],
                    "family": query["family"],
                    "eligible_targets": len(eligible),
                    "published_sid_absent": not (
                        query["published_sid_train"] or query["published_sid_validation"]
                    ),
                    "published_pdb_absent": not query["published_pdb_present"],
                    "positive_ranks": {
                        s["target_id"]: i
                        for i, s in enumerate(ranked, 1)
                        if s["target_id"] in positives
                    },
                }
            )
            metrics.append(row)
            hit = np.cumsum([s["target_id"] in positives for s in ranked])
            recalls = hit / len(positives)
            precision = hit / np.arange(1, len(hit) + 1)
            curves.append(
                [
                    float(precision[recalls >= r].max()) if (recalls >= r).any() else 0.0
                    for r in np.linspace(0, 1, 101)
                ]
            )
        summary = {
            "MAP": float(np.mean([r["AP"] for r in metrics])),
            "recall_at_10": float(np.mean([r["recall_at_10"] for r in metrics])),
            "top1_rate": float(np.mean([r["top1"] for r in metrics])),
            "tie_expected_MAP": float(np.mean([r["tie_expected_AP"] for r in metrics])),
            "query_count": len(metrics),
        }
        strata = {}
        for field in ["published_sid_absent", "published_pdb_absent"]:
            subset = [r for r in metrics if r[field]]
            strata[field] = {
                "queries": len(subset),
                "MAP": float(np.mean([r["AP"] for r in subset])) if subset else None,
                "recall_at_10": float(np.mean([r["recall_at_10"] for r in subset]))
                if subset
                else None,
            }
        outputs[view] = {
            "summary": summary,
            "per_query": metrics,
            "MAP_bootstrap": cluster_interval(metrics),
            "strata": strata,
            "pr_recall": np.linspace(0, 1, 101).tolist(),
            "pr_precision_macro_interpolated": np.mean(curves, axis=0).tolist(),
        }
    return outputs
