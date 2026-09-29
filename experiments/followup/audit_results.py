"""Independent recount from score TSVs and hierarchy labels, including neutral hits."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.common import read, save, sha

from .common import OUT


def main():
    out = OUT
    freeze = read(out / "test-freeze.json")
    for path, digest in freeze["files"].items():
        assert sha(out / path) == digest, path
    rows = read(out / "manifest-v2.json")
    queries = [r for r in rows if r["split"] == "test" and r["role"] == "query"]
    targets = {r["record_id"]: r for r in rows if r["split"] == "test" and r["role"] == "target"}
    result = read(out / "R1-results.json")
    tested, failures = 0, []
    for condition in result["rows"]:
        variant, (go, ge) = condition["variant"], condition["gap"]
        baseline_scores = None
        for repeat in range(3):
            directory = out / f"search-test/r{repeat}-{variant}-g{go}-{ge}"
            with (directory / "scores.tsv").open() as stream:
                scores = list(csv.DictReader(stream, delimiter="\t"))
            if baseline_scores is None:
                baseline_scores = scores
            else:
                assert scores == baseline_scores
            assert len({(r["query_id"], r["target_id"]) for r in scores}) == len(scores)
            reported = read(directory / "report.json")
            per_query = defaultdict(list)
            for score in scores:
                per_query[score["query_id"]].append(score)
            for view in ["all", "cross_family", "same_fold", "other_fold"]:
                values = []
                fold_values = defaultdict(list)
                reported_queries = {
                    q["query_id"]: q for q in reported["metrics"][view]["per_query"]
                }
                for query in queries:
                    eligible = set(targets)
                    if view == "cross_family":
                        eligible = {k for k, t in targets.items() if t["family"] != query["family"]}
                    elif view == "same_fold":
                        eligible = {k for k, t in targets.items() if t["fold"] == query["fold"]}
                    elif view == "other_fold":
                        eligible = {
                            k
                            for k, t in targets.items()
                            if t["fold"] != query["fold"]
                            or t["superfamily"] == query["superfamily"]
                        }
                    positive = {
                        k for k in eligible if targets[k]["superfamily"] == query["superfamily"]
                    }
                    ranked = sorted(
                        (h for h in per_query[query["record_id"]] if h["target_id"] in eligible),
                        key=lambda h: (-int(h["raw_score"]), h["target_id"]),
                    )
                    flags = [h["target_id"] in positive for h in ranked]
                    ap = sum(sum(flags[:i]) / i for i, hit in enumerate(flags, 1) if hit) / len(
                        positive
                    )
                    recall = sum(flags[:10]) / len(positive)
                    record = reported_queries[query["record_id"]]
                    assert abs(record["AP"] - ap) < 1e-12
                    assert abs(record["recall_at_10"] - recall) < 1e-12
                    assert record["eligible_targets"] == len(eligible)
                    assert record["relevant_count"] == len(positive)
                    values.append(ap)
                    fold_values[query["fold"]].append(ap)
                    if repeat == 0 and view == "all" and ap < 0.5:
                        failures.append(
                            {
                                "variant": variant,
                                "gap": [go, ge],
                                "query_id": query["record_id"],
                                "fold": query["fold"],
                                "superfamily": query["superfamily"],
                                "AP": ap,
                                "recall_at_10": recall,
                                "relevant_count": len(positive),
                                "top10": [
                                    {
                                        "target_id": h["target_id"],
                                        "score": int(h["raw_score"]),
                                        "kind": "positive"
                                        if h["target_id"] in positive
                                        else (
                                            "same_fold"
                                            if targets[h["target_id"]]["fold"] == query["fold"]
                                            else "other_fold"
                                        ),
                                    }
                                    for h in ranked[:10]
                                ],
                            }
                        )
                assert (
                    abs(sum(values) / len(values) - condition["metrics"][view]["summary"]["MAP"])
                    < 1e-12
                )
                if repeat == 0:
                    groups = [fold_values[f] for f in sorted(fold_values)]
                    rng = np.random.default_rng(20260929)
                    samples = []
                    for _ in range(2000):
                        selected = rng.integers(0, len(groups), len(groups))
                        merged = [value for index in selected for value in groups[index]]
                        samples.append(sum(merged) / len(merged))
                    np.testing.assert_allclose(
                        np.quantile(samples, [0.025, 0.975]),
                        condition["metrics"][view]["MAP_bootstrap"]["ci95"],
                        atol=1e-12,
                    )
                tested += len(queries)
    save(
        out / "R1-independent-audit.json",
        {
            "passed": True,
            "query_view_run_checks": tested,
            "conditions": len(result["rows"]),
            "poor_query_examples": failures,
            "no_metric_helpers_reused": True,
            "script_sha256": sha(Path(__file__)),
            "freeze_sha256": sha(out / "test-freeze.json"),
            "results_sha256": sha(out / "R1-results.json"),
        },
    )
    print("R1 independent recount passed", tested, "query/view/run checks", flush=True)


if __name__ == "__main__":
    main()
