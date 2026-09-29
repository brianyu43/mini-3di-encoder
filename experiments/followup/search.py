"""Cost preflight, validation gap selection and frozen R1 exhaustive comparison."""

import argparse
import resource
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from experiments.common import SEARCH_ROOT, bounded_cpu, read, save, sha

from .common import OUT
from .metrics import cluster_interval, evaluate


def query_batches(queries, target_length, limit=1_000_000_000):
    """Preserve every pair while honoring the unchanged engine's per-call DP ceiling."""
    batch, cells = [], 0
    for query in queries:
        cost = len(query.three_di) * target_length
        if cost > limit:
            raise ValueError("One query exceeds the search engine's per-call budget")
        if batch and cells + cost > limit:
            yield batch
            batch, cells = [], 0
        batch.append(query)
        cells += cost
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validation", "test"])
    args = parser.parse_args()
    out = OUT
    bounded_cpu(out)
    import numpy as np

    sys.path.insert(0, str(SEARCH_ROOT / "src"))
    from mini3di_search.align_numba import _score_batch, warmup
    from mini3di_search.index import IndexConfig, build_index
    from mini3di_search.index import digest as record_digest
    from mini3di_search.io import read_records
    from mini3di_search.pipeline import SearchConfig
    from mini3di_search.records import Alphabet
    from mini3di_search.scoring import Scoring, load_matrix
    from mini3di_search.search_numba import (
        FastResult,
        metadata,
        prepare_search,
        search,
        write_outputs,
    )

    manifest = read(out / "manifest-v2.json")
    protocol = read(out / "protocol-v2.json")
    data = read(out / "data-freeze.json")
    assert sha(out / "manifest-v2.json") == data["manifest_sha256"]
    assert sha(out / "protocol-v2.json") == data["protocol_sha256"]
    queries_meta = [r for r in manifest if r["split"] == args.stage and r["role"] == "query"]
    targets_meta = [r for r in manifest if r["split"] == args.stage and r["role"] == "target"]
    destination = out / f"search-{args.stage}"
    destination.mkdir(exist_ok=False)
    if args.stage == "test":
        frozen = read(out / "test-freeze.json")
        for name, digest in frozen["files"].items():
            assert sha(out / name) == digest, name
        if sha(Path(__file__)) != frozen["search_code_sha256"]:
            repair = read(out / "execution-repair.json")
            assert repair["original_freeze_sha256"] == sha(out / "test-freeze.json")
            assert repair["original_search_sha256"] == frozen["search_code_sha256"]
            assert repair["corrected_search_sha256"] == sha(Path(__file__))
        assert sha(Path(__file__).with_name("metrics.py")) == frozen["metrics_code_sha256"]
        for name, digest in frozen["search_engine_sources"].items():
            assert sha(SEARCH_ROOT / "src/mini3di_search" / name) == digest
    contexts = {}
    for variant in protocol["variants"]:
        alphabet = "learned" if variant == "learned_refit" else "official"
        queries = read_records(out / f"encoded/{alphabet}-{args.stage}-query.jsonl")
        targets = read_records(out / f"encoded/{alphabet}-{args.stage}-target.jsonl")
        index = build_index(targets, IndexConfig(3), allow_real=True)
        for word, postings in index.postings.items():
            assert "X" not in word
            for tid, offset in postings:
                assert all(index.targets[tid].valid_seed_mask[offset : offset + 3])
        matrix_path = out / f"matrices/{variant}.mat"
        assert sha(matrix_path) == data["matrices"][matrix_path.name]
        matrix = load_matrix(
            matrix_path,
            kind=Alphabet.THREE_DI,
            source=f"R1/{variant}; previous train-only matrix",
            synthetic=False,
        )
        contexts[variant] = queries, targets, index, matrix
    save(destination / "jit.json", warmup())

    def prepared(variant, gap):
        queries, targets, index, matrix = contexts[variant]
        target_length = sum(len(t.three_di) for t in targets)
        assert (
            sum(len(q.three_di) for q in queries) * target_length
            <= protocol["max_search_cells_per_condition"]
        )
        return [
            prepare_search(
                batch,
                targets,
                Scoring(matrix, *gap),
                allow_real=True,
                max_total_cells=1_000_000_000,
            )
            for batch in query_batches(queries, target_length)
        ]

    if args.stage == "validation":
        p = prepared("official_original", protocol["common_gap"])[0]
        candidate_ids = np.arange(len(p.targets), dtype=np.int64)
        cells = 0
        began = perf_counter()
        while cells < protocol["preflight_score_cells"]:
            for q in p.query_arrays:
                scores = _score_batch(
                    q, p.flat_targets, p.offsets, candidate_ids, p.matrix, np.int64(10), np.int64(1)
                )
                assert (scores >= 0).all()
                cells += len(q) * len(p.flat_targets)
                if cells >= protocol["preflight_score_cells"]:
                    break
        seconds = perf_counter() - began
        test_queries = [r for r in manifest if r["split"] == "test" and r["role"] == "query"]
        test_targets = [r for r in manifest if r["split"] == "test" and r["role"] == "target"]
        test_cells = sum(r["length"] for r in test_queries) * sum(r["length"] for r in test_targets)
        # 9 validation runs + 6 conditions x 3 repeats, with a traceback reserve.
        estimate = seconds / cells * (9 * p.exhaustive_dp_cells + 18 * test_cells) * 2 + 600
        similarity = read(out / "similarity-summary.json")["wall_seconds_this_invocation"]
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        allowed = (
            estimate + similarity < protocol["cpu_limit_seconds"]
            and peak < protocol["memory_limit_bytes"]
        )
        save(
            out / "cost-preflight.json",
            {
                "cells": cells,
                "seconds": seconds,
                "validation_data_only": True,
                "test_cells_per_condition": test_cells,
                "estimated_search_seconds_with_reserve": estimate,
                "completed_similarity_wall_seconds": similarity,
                "peak_process_rss_bytes_macos": peak,
                "passed": allowed,
                "method": (
                    "Warm validation score throughput; 2x score reserve +600s; "
                    "serial wall is conservative CPU budget proxy"
                ),
            },
        )
        if not allowed:
            raise RuntimeError("R1 cost preflight exceeded the frozen resource ceiling")

    started = perf_counter()

    def run(variant, gap, name):
        if perf_counter() - started > protocol["cpu_limit_seconds"]:
            raise RuntimeError("R1 search time ceiling exceeded")
        begin_prepare = perf_counter()
        batches = prepared(variant, gap)
        preparation_seconds = perf_counter() - begin_prepare
        parts = [
            search(batch, contexts[variant][2], SearchConfig("exhaustive", 3, 64, None), top_k=1)
            for batch in batches
        ]
        first = parts[0]
        result = FastResult(
            hits=tuple(h for part in parts for h in part.hits),
            scores=tuple(s for part in parts for s in part.scores),
            diagnostics=tuple(d for part in parts for d in part.diagnostics),
            config=first.config,
            index_id=first.index_id,
            scoring_id=first.scoring_id,
            query_hash=record_digest([asdict(q) for q in contexts[variant][0]]),
            target_hash=first.target_hash,
            top_k=1,
            synthetic=False,
            wall_seconds=sum(part.wall_seconds for part in parts),
        )
        metrics = evaluate(result.scores, queries_meta, targets_meta)
        report = {
            "variant": variant,
            "gap": gap,
            **metadata(result),
            "metrics": metrics,
            "query_batches": len(batches),
            "prepare_seconds": preparation_seconds,
        }
        write_outputs(destination / name, result, name)
        save(destination / name / "report.json", report)
        return report

    if args.stage == "validation":
        selection = {}
        summaries = []
        for variant in protocol["variants"]:
            runs = []
            for gap in protocol["validation_gap_grid"]:
                result = run(variant, gap, f"{variant}-g{gap[0]}-{gap[1]}")
                runs.append(result)
                summaries.append({"variant": variant, "gap": gap, "metrics": result["metrics"]})
                print(variant, gap, result["metrics"]["all"]["summary"], flush=True)
            best = min(runs, key=lambda r: (-r["metrics"]["all"]["summary"]["MAP"], *r["gap"]))
            selection[variant] = best["gap"]
        save(
            out / "validation-selection.json",
            {"selected_gaps": selection, "results": summaries, "test_used": False},
        )
        paths = [
            "manifest-v2.json",
            "protocol-v2.json",
            "data-freeze.json",
            "encoding.json",
            "validation-selection.json",
            "cost-preflight.json",
        ]
        paths += [str(p.relative_to(out)) for p in (out / "encoded").glob("*.jsonl")]
        paths += [str(p.relative_to(out)) for p in (out / "matrices").glob("*.mat")]
        save(
            out / "test-freeze.json",
            {
                "files": {p: sha(out / p) for p in paths},
                "selected_gaps": selection,
                "search_code_sha256": sha(Path(__file__)),
                "metrics_code_sha256": sha(Path(__file__).with_name("metrics.py")),
                "search_engine_sources": {
                    p.name: sha(p) for p in (SEARCH_ROOT / "src/mini3di_search").glob("*.py")
                },
                "test_has_not_run": True,
            },
        )
        print("test protocol frozen", flush=True)
        return
    results = {}
    # Evaluate distinct configurations; alias identical common/selected settings.
    for repeat in range(protocol["timing_repeats"]):
        variants = protocol["variants"][repeat:] + protocol["variants"][:repeat]
        for variant in variants:
            gaps = [protocol["common_gap"], frozen["selected_gaps"][variant]]
            for gap in dict.fromkeys(tuple(g) for g in gaps):
                result = run(variant, list(gap), f"r{repeat}-{variant}-g{gap[0]}-{gap[1]}")
                key = variant, gap
                results.setdefault(key, []).append(result)
            print("test repeated", repeat, variant, flush=True)
    rows = []
    for (variant, gap), runs in results.items():
        assert all(r["metrics"] == runs[0]["metrics"] for r in runs)
        first = runs[0]
        rows.append(
            {
                "variant": variant,
                "gap": list(gap),
                "settings": [
                    name
                    for name, g in [
                        ("common", protocol["common_gap"]),
                        ("selected", frozen["selected_gaps"][variant]),
                    ]
                    if list(gap) == g
                ],
                "metrics": first["metrics"],
                "dp_cells": first["dp_cells"],
                "aligned_pairs": first["aligned_pairs"],
                "search_seconds_median": statistics.median(r["search_seconds"] for r in runs),
                "search_seconds_range": [
                    min(r["search_seconds"] for r in runs),
                    max(r["search_seconds"] for r in runs),
                ],
                "stage_seconds_median": {
                    key: statistics.median(r["stage_seconds"][key] for r in runs)
                    for key in first["stage_seconds"]
                },
            }
        )
    differences = {}
    for setting in ["common", "selected"]:
        chosen = {r["variant"]: r for r in rows if setting in r["settings"]}
        for baseline in ["official_original", "official_refit"]:
            for view in ["all", "cross_family", "same_fold", "other_fold"]:
                reference = {
                    r["query_id"]: r["AP"] for r in chosen[baseline]["metrics"][view]["per_query"]
                }
                delta = [
                    {**r, "AP": r["AP"] - reference[r["query_id"]]}
                    for r in chosen["learned_refit"]["metrics"][view]["per_query"]
                ]
                differences[f"{setting}:learned-minus-{baseline}:{view}"] = cluster_interval(delta)
    save(
        out / "R1-results.json",
        {
            "rows": rows,
            "paired_differences": differences,
            "query_count": len(queries_meta),
            "target_count": len(targets_meta),
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wall_seconds": perf_counter() - started,
            "test_freeze_sha256": sha(out / "test-freeze.json"),
            "script_sha256": sha(Path(__file__)),
        },
    )
    print("R1 test finished", flush=True)


if __name__ == "__main__":
    main()
