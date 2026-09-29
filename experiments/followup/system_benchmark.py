"""Serial controlled backend benchmarks with candidates, scores and paths held fixed."""

import hashlib
import json
import resource
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.common import ROOT, SEARCH_ROOT, bounded_cpu, read, save, sha

from .common import OUT
from .search import query_batches


def main():
    out = OUT / "R3"
    bounded_cpu(out)
    sys.path.insert(0, str(SEARCH_ROOT / "src"))
    from mini3di_search.align_numba import _score_batch, warmup
    from mini3di_search.align_reference import align
    from mini3di_search.index import IndexConfig, build_index
    from mini3di_search.prefilter import collect_hits, filter_ungapped, supported_diagonals
    from mini3di_search.records import Alphabet
    from mini3di_search.scoring import Scoring, load_matrix
    from mini3di_search.search_numba import prepare_search
    from mini3di_search.traceback import rescore_alignment

    from .evaluation import records
    from .system_kernels import (
        decoded_alignment,
        packed_postings,
        support_kernel,
        traceback_kernel,
        ungapped_kernel,
    )

    data = read(out / "data-freeze.json")
    protocol = data["protocol"]
    assert sha(ROOT / protocol["model"]) == data["encoder_sha256"]
    assert sha(ROOT / protocol["matrix"]) == data["matrix_sha256"]
    destination = out / "benchmark"
    destination.mkdir(exist_ok=False)
    save(
        destination / "execution-freeze.json",
        {
            "data_freeze_sha256": sha(out / "data-freeze.json"),
            "script_sha256": sha(__file__),
            "kernels_sha256": sha(Path(__file__).with_name("system_kernels.py")),
            "source_engine": {
                p.name: sha(p) for p in (SEARCH_ROOT / "src/mini3di_search").glob("*.py")
            },
            "timed_execution_is_serial": True,
        },
    )
    sc = Scoring(
        load_matrix(
            ROOT / protocol["matrix"],
            kind=Alphabet.THREE_DI,
            source="fixed released bundle",
            synthetic=False,
        ),
        *protocol["gap"],
    )
    assert "".join(sc.matrix.alphabet) == "ACDEFGHIKLMNPQRSTVWYX"
    arrays = {}
    encoding = read(out / "encoding.json")
    for r in data["queries"] + data["targets"]:
        path = out / f"encoded/{r['record_id']}.npz"
        assert sha(path) == encoding["encoded_files"][path.name]
        with np.load(path) as a:
            arrays[r["record_id"]] = {k: a[k] for k in ["states", "mask"]}
    qrecords = records(data["queries"], arrays)
    by_id = {r["record_id"]: r for r in data["targets"]}
    begun = perf_counter()
    preexisting_cache = list((out / "numba-cache").rglob("*.nbc"))
    jit = warmup()
    qsmall = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    empty_offsets = np.zeros(21**3 + 1, dtype=np.int64)
    empty_postings = np.empty((0, 2), dtype=np.int64)
    support_kernel(qsmall, np.ones(6, dtype=bool), empty_offsets, empty_postings, 6)
    ungapped_kernel(
        qsmall,
        qsmall,
        np.array([0, 6], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array(sc.matrix.values),
        20,
    )
    traceback_kernel(qsmall, qsmall, np.array(sc.matrix.values), *protocol["gap"])
    save(
        destination / "jit.json",
        {
            "seconds": perf_counter() - begun,
            "preexisting_cache_files": len(preexisting_cache),
            "score_kernel": jit,
            "scope": "first calls in this process; fresh dedicated cache if count is zero",
        },
    )

    def context(condition):
        targets = records([by_id[s] for s in condition["target_ids"]], arrays)
        began = perf_counter()
        index = build_index(targets, IndexConfig(3), allow_real=True)
        index_seconds = perf_counter() - began
        began = perf_counter()
        po, postings = packed_postings(index)
        packed_seconds = perf_counter() - began
        began = perf_counter()
        parts = [
            prepare_search(b, targets, sc, allow_real=True)
            for b in query_batches(qrecords, sum(len(t.three_di) for t in targets))
        ]
        p = parts[0]
        qs = [
            (q, a) for part in parts for q, a in zip(part.queries, part.query_arrays, strict=True)
        ]
        return {
            "p": p,
            "qs": qs,
            "index": index,
            "po": po,
            "postings": postings,
            "max_length": max(len(t.three_di) for t in targets),
            "preparation_seconds": perf_counter() - began,
            "index_seconds": index_seconds,
            "packed_index_seconds": packed_seconds,
        }

    def run(ctx, mode, backend, trace):
        p, index = ctx["p"], ctx["index"]
        began = perf_counter()
        stages = dict.fromkeys(["candidate", "ungapped", "score", "ranking", "traceback"], 0.0)
        signatures, paths, candidates, rankings, cells = [], [], {}, {}, 0
        for q, qa in ctx["qs"]:
            start = perf_counter()
            support = {}
            if mode == "exhaustive":
                ids = np.arange(len(index.targets), dtype=np.int64)
                tids = diagonals = np.empty(0, dtype=np.int64)
            elif backend == "baseline":
                support = supported_diagonals(collect_hits(q, index), k=3, window=64, double=True)
            else:
                tids, diagonals = support_kernel(
                    qa, np.array(q.valid_seed_mask), ctx["po"], ctx["postings"], ctx["max_length"]
                )
                if backend in ["numba_seed", "preencoded_ungapped"]:
                    for tid, diagonal in zip(tids, diagonals, strict=True):
                        support.setdefault(int(tid), []).append(int(diagonal))
            stages["candidate"] += perf_counter() - start
            start = perf_counter()
            if mode != "exhaustive":
                if backend in ["baseline", "numba_seed"]:
                    accepted, _ = filter_ungapped(q, index, support, sc, 20)
                    ids = np.array(accepted, dtype=np.int64)
                elif backend == "preencoded_ungapped":
                    accepted = []
                    for tid, diagonals_list in support.items():
                        best = 0
                        target = p.flat_targets[p.offsets[tid] : p.offsets[tid + 1]]
                        for diagonal in diagonals_list:
                            current = 0
                            for i in range(max(0, -diagonal), min(len(qa), len(target) - diagonal)):
                                current = max(
                                    0, current + int(p.matrix[qa[i], target[i + diagonal]])
                                )
                                best = max(best, current)
                        if best >= 20:
                            accepted.append(tid)
                    ids = np.array(accepted, dtype=np.int64)
                else:
                    ids, _ = ungapped_kernel(
                        qa, p.flat_targets, p.offsets, tids, diagonals, p.matrix, 20
                    )
            stages["ungapped"] += perf_counter() - start
            start = perf_counter()
            values = _score_batch(
                qa,
                p.flat_targets,
                p.offsets,
                ids,
                p.matrix,
                np.int64(sc.gap_open),
                np.int64(sc.gap_extend),
            )
            stages["score"] += perf_counter() - start
            start = perf_counter()
            ranked = sorted(
                [
                    (int(v), index.targets[int(tid)].record_id, int(tid))
                    for tid, v in zip(ids, values, strict=True)
                    if v > 0
                ],
                key=lambda item: (-item[0], item[1]),
            )
            stages["ranking"] += perf_counter() - start
            start = perf_counter()
            if trace and ranked:
                value, tid, numeric = ranked[0]
                target = index.targets[numeric]
                if backend == "numba_traceback":
                    ta = p.flat_targets[p.offsets[numeric] : p.offsets[numeric + 1]]
                    result = decoded_alignment(
                        q.three_di,
                        target.three_di,
                        traceback_kernel(qa, ta, p.matrix, sc.gap_open, sc.gap_extend),
                    )
                else:
                    result = align(
                        q.sequence(Alphabet.THREE_DI), target.sequence(Alphabet.THREE_DI), sc
                    )
                assert (
                    rescore_alignment(
                        q.sequence(Alphabet.THREE_DI),
                        target.sequence(Alphabet.THREE_DI),
                        result,
                        sc,
                    )
                    == value
                )
                paths.append({"query": q.record_id, "target": tid, **asdict(result)})
            stages["traceback"] += perf_counter() - start
            signatures.append((q.record_id, ids.tolist(), values.tolist()))
            candidates[q.record_id] = [index.targets[int(t)].record_id for t in ids]
            rankings[q.record_id] = [(score, sid) for score, sid, _ in ranked]
            cells += len(qa) * sum(int(p.offsets[i + 1] - p.offsets[i]) for i in ids)
        seconds = perf_counter() - began
        digest = hashlib.sha256(json.dumps(signatures, separators=(",", ":")).encode()).hexdigest()
        return {
            "seconds": seconds,
            "stages": stages,
            "candidate_score_sha256": digest,
            "candidate_pairs": sum(map(len, candidates.values())),
            "dp_cells": cells,
            "paths": paths,
            "candidates": candidates,
            "rankings": rankings,
        }

    # Full baseline smallest-size runs in each length stratum, before timing the scaling sweep.
    preflights = {}
    for c in data["conditions"]:
        if c["target_count"] != 50:
            continue
        ctx = context(c)
        measurements = [run(ctx, mode, "baseline", True)["seconds"] for mode in protocol["modes"]]
        preflights[str(c["length"])] = max(measurements)
    estimate = 0.0
    for c in data["conditions"]:
        base = preflights.get(str(c["length"]), max(preflights.values()))
        estimate += base * c["target_count"] / 50 * 3 * 2 * len(protocol["backends"]) * 5 * 2
    already = data["selection_seconds"] + encoding["seconds"]
    save(
        destination / "preflight.json",
        {
            "smallest_top1_seconds": preflights,
            "estimated_sweep_seconds_with_3x_reserve": estimate,
            "completed_preparation_seconds": already,
            "passed": estimate + already < protocol["cpu_seconds_limit"],
        },
    )
    if estimate + already >= protocol["cpu_seconds_limit"]:
        raise RuntimeError("R3 preflight exceeds fixed CPU ceiling")
    started, outputs = perf_counter(), []
    for condition in data["conditions"]:
        ctx = context(condition)
        conditions = {}
        for mode in protocol["modes"]:
            oracle = run(ctx, mode, "baseline", True)
            records_by_backend = {b: {"score_only": [], "top1": []} for b in protocol["backends"]}
            for repeat in range(5):
                order = protocol["backends"][repeat:] + protocol["backends"][:repeat]
                for backend in order:
                    for trace in [False, True]:
                        result = run(ctx, mode, backend, trace)
                        assert result["candidate_score_sha256"] == oracle["candidate_score_sha256"]
                        if trace:
                            assert result["paths"] == oracle["paths"]
                        records_by_backend[backend]["top1" if trace else "score_only"].append(
                            {k: result[k] for k in ["seconds", "stages", "candidate_score_sha256"]}
                        )
            conditions[mode] = {"repeats": records_by_backend, "oracle": oracle}
        exhaustive = conditions["exhaustive"]["oracle"]["rankings"]
        filtered = conditions["double-ungapped"]["oracle"]["candidates"]
        retention = []
        for sid, ranked in exhaustive.items():
            top = [tid for _, tid in ranked[:10]]
            retention.append(sum(t in filtered[sid] for t in top) / len(top) if top else 1.0)
        summary = {
            "name": condition["name"],
            "length": condition["length"],
            "query_count": len(qrecords),
            "target_count": condition["target_count"],
            "index_seconds": ctx["index_seconds"],
            "packed_index_seconds": ctx["packed_index_seconds"],
            "preparation_seconds": ctx["preparation_seconds"],
            "retain_exact_at_10": float(np.mean(retention)),
            "modes": {},
        }
        for mode, values in conditions.items():
            summary["modes"][mode] = {
                "candidate_pairs": values["oracle"]["candidate_pairs"],
                "dp_cells": values["oracle"]["dp_cells"],
                "backends": {},
            }
            for backend, timings in values["repeats"].items():
                summary["modes"][mode]["backends"][backend] = {}
                for label, repetitions in timings.items():
                    seconds = [r["seconds"] for r in repetitions]
                    summary["modes"][mode]["backends"][backend][label] = {
                        "median": statistics.median(seconds),
                        "minimum": min(seconds),
                        "maximum": max(seconds),
                        "repeats": repetitions,
                    }
        output_start = perf_counter()
        save(destination / f"{condition['name']}-raw.json", conditions)
        summary["output_seconds"] = perf_counter() - output_start
        outputs.append(summary)
        save(destination / f"{condition['name']}.json", summary)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > protocol["peak_rss_bytes_limit"] or perf_counter() - started + already > 14400:
            raise RuntimeError("R3 measured resource ceiling exceeded")
        print(
            "R3",
            condition["name"],
            "retain",
            round(summary["retain_exact_at_10"], 3),
            "speedup",
            round(
                summary["modes"]["double-ungapped"]["backends"]["baseline"]["top1"]["median"]
                / summary["modes"]["double-ungapped"]["backends"]["numba_traceback"]["top1"][
                    "median"
                ],
                2,
            ),
            flush=True,
        )
    save(
        out / "system-results.json",
        {
            "conditions": outputs,
            "skipped": data["skipped"],
            "all_candidates_scores_paths_equal": True,
            "wall_seconds": perf_counter() - started,
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "script_sha256": sha(__file__),
        },
    )


if __name__ == "__main__":
    main()
