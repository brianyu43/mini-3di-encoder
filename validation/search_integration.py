"""Audit encoder JSONL with the separately installed mini-3di-search engine.

Run with that engine's Python environment. No source changes or cache writes are
made in its checkout. This is a tiny development integration, not a benchmark.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

MATRIX_SHA256 = "63cdc9b17de248c790e934ffb7739d067a59c7c6cdb73a2c9c99de63e662b557"
MATRIX_SOURCE = (
    "https://github.com/steineggerlab/foldseek/blob/"
    "941cd33ff0771cd2e3f144e3293e22a2b87e9fda/data/mat3di.out"
)
MODES = ("exhaustive", "single", "double", "double-ungapped")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tsv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoded-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.encoded_dir.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["NUMBA_CACHE_DIR"] = str(out / "numba-cache")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"

    import mini3di_search
    from mini3di_search.index import IndexConfig, load_index, seed_windows
    from mini3di_search.io import read_records
    from mini3di_search.prefilter import collect_hits
    from mini3di_search.records import ProteinRecord

    def command(name, *arguments):
        argv = [sys.executable, "-m", "mini3di_search.cli", *map(str, arguments)]
        began = perf_counter()
        process = subprocess.run(argv, capture_output=True, text=True)
        save(
            out / f"{name}.command.json",
            {
                "argv": argv,
                "returncode": process.returncode,
                "wall_seconds": perf_counter() - began,
                "stdout": process.stdout,
                "stderr": process.stderr,
            },
        )
        process.check_returncode()

    config = json.loads((source / "config.json").read_text())
    provenance = json.loads((source / "export_provenance.json").read_text())
    assert sha(args.matrix) == MATRIX_SHA256
    assert sha(source / "config.json") == provenance["dataset_sha256"]
    for role in ("query", "target"):
        assert sha(source / f"{role}.jsonl") == provenance[f"{role}_sha256"]
        command(f"validate-{role}", "validate-records", source / f"{role}.jsonl")
    queries = read_records(source / "query.jsonl")
    targets = read_records(source / "target.jsonl")
    for record in queries + targets:
        encoded_path = source / record.record_id / "encoded.json"
        origin = next(p for p in provenance["inputs"] if p["record_id"] == record.record_id)
        assert sha(encoded_path) == origin["encoded_sha256"]
        encoded = json.loads(encoded_path.read_text())
        assert record.aa == encoded["aa"] and record.three_di == encoded["three_di"]
        assert record.valid_seed_mask == tuple(encoded["valid_seed_mask"])
        assert not record.synthetic
        assert len(record.aa) == len(encoded["rows"])
        for i, row in enumerate(encoded["rows"]):
            assert row["index"] == i and row["aa"] == record.aa[i]
            assert row["seed_valid"] == record.valid_seed_mask[i]
            assert record.three_di[i] == (row["letter"] if row["seed_valid"] else "X")

    k = config["k"]
    command(
        "index",
        "index",
        "--records",
        source / "target.jsonl",
        "--out",
        out / "index.json",
        "--k",
        k,
        "--real",
    )
    index = load_index(out / "index.json", expected=IndexConfig(k), allow_real=True)

    def safe(record, start):
        assert 0 <= start <= len(record.three_di) - k
        assert all(record.valid_seed_mask[start : start + k])
        assert "X" not in record.three_di[start : start + k]

    postings = hits = windows = 0
    for word, entries in index.postings.items():
        for target_id, start in entries:
            target = index.targets[target_id]
            safe(target, start)
            assert target.three_di[start : start + k] == word
            postings += 1
    for query in queries:
        for start, _ in seed_windows(query, k):
            safe(query, start)
            windows += 1
        for hit in collect_hits(query, index):
            target = index.targets[hit.target_numeric_id]
            safe(query, hit.query_start)
            safe(target, hit.target_start)
            assert (
                query.three_di[hit.query_start : hit.query_start + k]
                == target.three_di[hit.target_start : hit.target_start + k]
            )
            hits += 1
    # A raw D can be valid OR invalid. The explicit mask must decide, not D itself.
    probe = ProteinRecord("mask-probe", "A" * 8, "D" * 8, (False, *([True] * 6), False), True)
    probe_starts = [start for start, _ in seed_windows(probe, 3)]
    assert probe_starts == [1, 2, 3, 4]
    save(
        out / "mask_audit.json",
        {
            "target_postings": postings,
            "query_windows": windows,
            "seed_hits": hits,
            "invalid_seed_count": 0,
            "raw_D_probe_starts_k3": probe_starts,
            "scope": "Every posting and seed hit; neutral X alignment can span X.",
        },
    )

    reports, scores, ranks = {}, {}, {}
    for mode in MODES:
        arguments = [
            "search",
            "--queries",
            source / "query.jsonl",
            "--db",
            out / "index.json",
            "--matrix",
            args.matrix,
            "--matrix-source",
            MATRIX_SOURCE,
            "--real",
            "--mode",
            mode,
            "--k",
            k,
            "--window",
            config["double_window"],
            "--gap-open",
            config["gap_open"],
            "--gap-extend",
            config["gap_extend"],
            "--top-k",
            len(targets),
            "--out",
            out / mode,
        ]
        if mode == "double-ungapped":
            arguments.extend(["--ungapped-threshold", config["ungapped_threshold"]])
        command(mode, *arguments)
        reports[mode] = json.loads((out / mode / "run.json").read_text())
        scores[mode] = {
            (r["query_id"], r["target_id"]): int(r["raw_score"])
            for r in tsv(out / mode / "scores.tsv")
        }
        ranks[mode] = {
            (r["query_id"], r["target_id"]): int(r["rank"]) for r in tsv(out / mode / "hits.tsv")
        }
        for pair, score in scores[mode].items():
            assert scores["exhaustive"][pair] == score
        assert reports[mode]["scoring_id"] == reports["exhaustive"]["scoring_id"]
        print(mode, reports[mode]["aligned_pairs"], flush=True)
    assert reports["exhaustive"]["aligned_pairs"] == len(queries) * len(targets)
    rows = []
    for mode in MODES:
        report = reports[mode]
        retained = denominator = 0
        for query in queries:
            exact = sorted(
                (
                    (score, tid)
                    for (qid, tid), score in scores["exhaustive"].items()
                    if qid == query.record_id
                ),
                key=lambda p: (-p[0], p[1]),
            )[:10]
            candidates = next(
                d["candidate_ids"]
                for d in report["diagnostics"]
                if d["query_id"] == query.record_id
            )
            retained += sum(tid in candidates for _, tid in exact)
            denominator += len(exact)
        rows.append(
            {
                "mode": mode,
                "aligned_pairs": report["aligned_pairs"],
                "dp_cells": report["dp_cells"],
                "positive_hits": len(scores[mode]),
                "retained_exact_top_min10": retained,
                "exact_top_denominator": denominator,
                "search_seconds": report["search_seconds"],
                "memory": report["memory"],
                "related_pair_ranks": [
                    {"query": q, "target": t, "rank": ranks[mode].get((q, t))}
                    for q, t in config["related_pairs"]
                ],
            }
        )
    engine = Path(mini3di_search.__file__).resolve().parent
    save(
        out / "summary.json",
        {
            "scope": "6x6 development integration; no generalization or speedup claim",
            "query_count": len(queries),
            "target_count": len(targets),
            "modes": rows,
            "matrix_sha256": sha(args.matrix),
            "dataset_sha256": provenance["dataset_sha256"],
            "export_provenance_sha256": sha(source / "export_provenance.json"),
            "integration_script_sha256": sha(Path(__file__)),
            "python": sys.version,
            "search_source_sha256": {p.name: sha(p) for p in sorted(engine.glob("*.py"))},
            "score_disagreements_with_exhaustive": 0,
            "invalid_seed_count": 0,
        },
    )


if __name__ == "__main__":
    main()
