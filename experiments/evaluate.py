"""Validation-only gap selection followed by frozen, repeated held-out search."""

import argparse
import resource
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from mini3di_encoder.learned import load_learned_model, predict_states

from .common import ALPHABET, DEFAULT_OUT, SEARCH_ROOT, bounded_cpu, read, save, sha
from .metrics import grouped_bootstrap, retrieval_metrics

VARIANTS = ("official_original", "official_refit", "learned_refit")
MODES = ("exhaustive", "single", "double", "double-ungapped")
PROTOCOL = {
    "primary": "MAP over all positive-score ranked targets; denominator all six relevant targets",
    "relevance": "same SCOPe superfamily; no self-hit or shared PDB",
    "ties": "descending raw score, ascending record_id; tie-expected AP also reported",
    "validation_selection": "maximum MAP, then gap_open ascending, gap_extend ascending",
    "traceback": "Top 1 Python traceback/rescore; scores saved for every positive hit",
    "timing": "3 in-process repetitions, warmed Numba, no disk I/O; no cold-cache speedup claim",
    "bootstrap": "2000 superfamily-cluster percentile resamples, model held fixed",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validation", "test"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    bounded_cpu(out)
    sys.path.insert(0, str(SEARCH_ROOT / "src"))
    from mini3di_search.align_numba import warmup
    from mini3di_search.index import IndexConfig, build_index, save_index, seed_windows
    from mini3di_search.pipeline import SearchConfig
    from mini3di_search.prefilter import collect_hits
    from mini3di_search.records import Alphabet, ProteinRecord
    from mini3di_search.scoring import Scoring, load_matrix
    from mini3di_search.search_numba import metadata, prepare_search, search, write_outputs

    directory = out / f"search-{args.stage}"
    directory.mkdir(exist_ok=False)
    manifest = read(out / "manifest.json")
    config = read(out / "experiment_config.json")
    freeze = read(out / "data-freeze.json")
    assert sha(out / "manifest.json") == freeze["manifest_sha256"]
    assert sha(out / "experiment_config.json") == freeze["experiment_config_sha256"]
    selected_seed = read(out / "models/selection.json")["selected_seed"]
    encoder_path = out / f"models/seed-{selected_seed}/encoder.json"
    layers, spec = load_learned_model(encoder_path)
    if args.stage == "validation":
        save(out / "evaluation_protocol.json", PROTOCOL)
    else:
        test_freeze = read(out / "test-freeze.json")
        assert sha(Path(__file__)) == test_freeze["script_sha256"]
        assert sha(Path(__file__).with_name("metrics.py")) == test_freeze["metrics_code_sha256"]
        for path, digest in test_freeze["files"].items():
            assert sha(out / path) == digest, path
        assert test_freeze["selected_seed"] == selected_seed
    rows = [r for r in manifest if r["split"] == args.stage]
    queries_meta = [r for r in rows if r["role"] == "query"]
    targets_meta = [r for r in rows if r["role"] == "target"]
    encoded = {}
    encoding_times = {}
    for alphabet in ["official", "learned"]:
        records = []
        began = perf_counter()
        for row in rows:
            with np.load(out / f"features/{row['record_id']}.npz") as arrays:
                mask = arrays["mask"]
                states = (
                    arrays["states"]
                    if alphabet == "official"
                    else predict_states(layers, spec, arrays["x"])
                )
                text = "".join(
                    ALPHABET[s] if ok else "X" for s, ok in zip(states, mask, strict=True)
                )
                records.append(
                    ProteinRecord(row["record_id"], row["aa"], text, tuple(map(bool, mask)), False)
                )
        encoding_times[alphabet] = perf_counter() - began
        encoded[alphabet] = {r.record_id: r for r in records}
        for role in ["query", "target"]:
            path = directory / f"{alphabet}-{role}.jsonl"
            import json

            path.write_text(
                "".join(
                    json.dumps(asdict(encoded[alphabet][r["record_id"]])) + "\n"
                    for r in rows
                    if r["role"] == role
                )
            )
    save(directory / "jit.json", warmup())
    contexts, index_times = {}, {}
    mask_audits = {}
    for variant in VARIANTS:
        alphabet = "learned" if variant == "learned_refit" else "official"
        queries = [encoded[alphabet][r["record_id"]] for r in queries_meta]
        targets = [encoded[alphabet][r["record_id"]] for r in targets_meta]
        began = perf_counter()
        index = build_index(targets, IndexConfig(config["k"]), allow_real=True)
        index_times[variant] = perf_counter() - began
        save_index(directory / f"{variant}-index.json", index)
        audited_postings = audited_hits = 0
        for word, entries in index.postings.items():
            for tid, start in entries:
                record = index.targets[tid]
                assert all(record.valid_seed_mask[start : start + config["k"]]) and "X" not in word
                audited_postings += 1
        for query in queries:
            for start, word in seed_windows(query, config["k"]):
                assert all(query.valid_seed_mask[start : start + config["k"]]) and "X" not in word
            for hit in collect_hits(query, index):
                target = index.targets[hit.target_numeric_id]
                assert all(query.valid_seed_mask[hit.query_start : hit.query_start + config["k"]])
                assert all(
                    target.valid_seed_mask[hit.target_start : hit.target_start + config["k"]]
                )
                audited_hits += 1
        mask_audits[variant] = {"postings": audited_postings, "hits": audited_hits, "invalid": 0}
        matrix = load_matrix(
            out / f"matrices/{variant}.mat",
            kind=Alphabet.THREE_DI,
            source=f"sessions-11-15/{variant}; train-only refit where applicable",
            synthetic=False,
        )
        contexts[variant] = (queries, targets, index, matrix)

    def run(variant, gap, mode, name):
        queries, targets, index, matrix = contexts[variant]
        scoring = Scoring(matrix, *gap)
        prepared = prepare_search(
            queries, targets, scoring, allow_real=True, max_total_cells=200_000_000
        )
        settings = SearchConfig(
            mode,
            config["k"],
            config["window"],
            config["ungapped_threshold"] if mode == "double-ungapped" else None,
        )
        result = search(prepared, index, settings, top_k=1)
        metrics = retrieval_metrics(result.scores, queries_meta, targets_meta)
        report = {"variant": variant, "gap": gap, "mode": mode, **metadata(result), **metrics}
        write_outputs(directory / name, result, name)
        save(directory / name / "report.json", report)
        return report, result

    if args.stage == "validation":
        reports, selected = [], {}
        for variant in VARIANTS:
            candidates = []
            for gap in config["gap_grid"]:
                report, _ = run(variant, gap, "exhaustive", f"{variant}-g{gap[0]}-{gap[1]}")
                candidates.append(report)
                reports.append(report)
                print(variant, gap, report["summary"], flush=True)
            best = min(candidates, key=lambda r: (-r["summary"]["MAP"], *r["gap"]))
            selected[variant] = best["gap"]
        save(
            directory / "selection.json",
            {"selected_gaps": selected, "reports": reports, "test_used": False},
        )
        paths = [
            "manifest.json",
            "experiment_config.json",
            "models/selection.json",
            f"models/seed-{selected_seed}/encoder.json",
            "evaluation_protocol.json",
            "search-validation/selection.json",
            "alignment_audit.json",
            *[f"matrices/{v}.mat" for v in VARIANTS],
        ]
        save(
            out / "test-freeze.json",
            {
                "selected_seed": selected_seed,
                "selected_gaps": selected,
                "files": {p: sha(out / p) for p in paths},
                "script_sha256": sha(Path(__file__)),
                "metrics_code_sha256": sha(Path(__file__).with_name("metrics.py")),
                "test_search_has_not_run": True,
            },
        )
        return

    selected = test_freeze["selected_gaps"]
    reports, baseline = defaultdict(list), {}
    for repeat in range(config["timing_repeats"]):
        order = VARIANTS[repeat % len(VARIANTS) :] + VARIANTS[: repeat % len(VARIANTS)]
        for variant in order:
            for mode in MODES:
                report, result = run(
                    variant, selected[variant], mode, f"r{repeat}-{variant}-{mode}"
                )
                if mode == "exhaustive":
                    baseline[variant] = result
                exact = baseline[variant]
                expected = {(r["query_id"], r["target_id"]): r["raw_score"] for r in exact.scores}
                assert all(
                    expected[(r["query_id"], r["target_id"])] == r["raw_score"]
                    for r in result.scores
                )
                retained = denominator = 0
                for d in result.diagnostics:
                    top = [r["target_id"] for r in exact.scores if r["query_id"] == d["query_id"]][
                        :10
                    ]
                    retained += len(set(top) & set(d["candidate_ids"]))
                    denominator += len(top)
                report.update(
                    {
                        "retain_exact_at_10": retained / denominator,
                        "retained_exact_top10": retained,
                        "exact_top10_denominator": denominator,
                        "score_disagreements": 0,
                    }
                )
                save(directory / f"r{repeat}-{variant}-{mode}" / "report.json", report)
                reports[(variant, mode)].append(report)
            print("test repetition", repeat, variant, flush=True)
    summaries = []
    for (variant, mode), repeats in reports.items():
        first = repeats[0]
        assert all(r["summary"] == first["summary"] for r in repeats)
        timing_keys = ["sw_score", "candidate", "ungapped", "traceback_recompute", "rescore"]
        summaries.append(
            {
                "variant": variant,
                "mode": mode,
                "gap": selected[variant],
                **first["summary"],
                "retain_exact_at_10": first["retain_exact_at_10"],
                "retained_exact_top10": first["retained_exact_top10"],
                "exact_top10_denominator": first["exact_top10_denominator"],
                "aligned_pairs": first["aligned_pairs"],
                "dp_cells": first["dp_cells"],
                "search_seconds_median": statistics.median(r["search_seconds"] for r in repeats),
                "search_seconds_range": [
                    min(r["search_seconds"] for r in repeats),
                    max(r["search_seconds"] for r in repeats),
                ],
                "stage_seconds_median": {
                    key: statistics.median(r["stage_seconds"][key] for r in repeats)
                    for key in timing_keys
                },
                "MAP_bootstrap": grouped_bootstrap(
                    first["per_query"], repeats=config["bootstrap_repeats"]
                ),
                "per_query": first["per_query"],
            }
        )
    exacts = {r["variant"]: r for r in summaries if r["mode"] == "exhaustive"}
    differences = {}
    for a, b in [
        ("learned_refit", "official_original"),
        ("learned_refit", "official_refit"),
        ("official_refit", "official_original"),
    ]:
        by_id = {r["query_id"]: r for r in exacts[b]["per_query"]}
        rows_delta = [
            {**r, "AP": r["AP"] - by_id[r["query_id"]]["AP"]} for r in exacts[a]["per_query"]
        ]
        differences[f"{a}-minus-{b}"] = grouped_bootstrap(
            rows_delta, repeats=config["bootstrap_repeats"]
        )
    save(
        directory / "summary.json",
        {
            "queries": len(queries_meta),
            "targets": len(targets_meta),
            "groups": 8,
            "repetitions": config["timing_repeats"],
            "rows": summaries,
            "paired_MAP_differences": differences,
            "mask_audits": mask_audits,
            "index_seconds": index_times,
            "record_materialization_seconds": encoding_times,
            "materialization_caveat": (
                "Official states cached; learned forward computed. Not encoding speed comparison."
            ),
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "memory_scope": "Whole evaluation Python process; cumulative peak, not per variant",
            "freeze_sha256": sha(out / "test-freeze.json"),
            "script_sha256": sha(Path(__file__)),
            "search_source_sha256": {
                p.name: sha(p) for p in (SEARCH_ROOT / "src/mini3di_search").glob("*.py")
            },
        },
    )
    print("test complete", [(r["variant"], r["MAP"]) for r in exacts.values()], flush=True)


if __name__ == "__main__":
    main()
