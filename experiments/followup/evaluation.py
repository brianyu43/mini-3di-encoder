"""Research-only exhaustive score tables and independently checkable filter ablations."""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.common import ALPHABET, SEARCH_ROOT, save

from .metrics import evaluate
from .search import query_batches

sys.path.insert(0, str(SEARCH_ROOT / "src"))
from mini3di_search.align_numba import _score_batch  # noqa: E402
from mini3di_search.index import IndexConfig, build_index, seed_windows  # noqa: E402
from mini3di_search.prefilter import SeedHit, collect_hits, supported_diagonals  # noqa: E402
from mini3di_search.records import Alphabet, ProteinRecord  # noqa: E402
from mini3di_search.scoring import Scoring, load_matrix  # noqa: E402
from mini3di_search.search_numba import prepare_search  # noqa: E402


def records(rows, arrays):
    return [
        ProteinRecord(
            r["record_id"],
            r["aa"],
            "".join(
                ALPHABET[int(s)] if ok else "X"
                for s, ok in zip(
                    arrays[r["record_id"]]["states"], arrays[r["record_id"]]["mask"], strict=True
                )
            ),
            tuple(bool(x) for x in arrays[r["record_id"]]["mask"]),
            False,
        )
        for r in rows
    ]


def exhaustive(qrows, trows, arrays, matrix_path, destination):
    """Same integer rolling SW as the engine. Timing explicitly excludes traceback."""
    queries, targets = records(qrows, arrays), records(trows, arrays)
    scoring = Scoring(
        load_matrix(
            Path(matrix_path),
            kind=Alphabet.THREE_DI,
            source="frozen research matrix",
            synthetic=False,
        ),
        10,
        1,
    )
    began = perf_counter()
    batches = [
        prepare_search(b, targets, scoring, allow_real=True)
        for b in query_batches(queries, sum(len(t.three_di) for t in targets))
    ]
    prepared_seconds = perf_counter() - began
    began = perf_counter()
    scores, total_cells = [], 0
    for p in batches:
        ids = np.arange(len(p.targets), dtype=np.int64)
        total_cells += p.exhaustive_dp_cells
        for q, qa in zip(p.queries, p.query_arrays, strict=True):
            values = _score_batch(
                qa, p.flat_targets, p.offsets, ids, p.matrix, np.int64(10), np.int64(1)
            )
            ranked = sorted(
                [(int(v), t.record_id) for v, t in zip(values, p.targets, strict=True) if v > 0],
                key=lambda x: (-x[0], x[1]),
            )
            scores.extend(
                {"query_id": q.record_id, "target_id": tid, "rank": rank, "raw_score": score}
                for rank, (score, tid) in enumerate(ranked, 1)
            )
    seconds = perf_counter() - began
    metrics = evaluate(scores, qrows, trows)
    destination = Path(destination)
    destination.mkdir(exist_ok=False, parents=True)
    write_scores(destination / "scores.tsv", scores)
    summary = {
        "metrics": metrics,
        "score_and_ranking_seconds": seconds,
        "prepare_seconds": prepared_seconds,
        "dp_cells": total_cells,
        "aligned_pairs": len(qrows) * len(trows),
        "traceback_included": False,
    }
    save(destination / "results.json", summary)
    return scores, summary, queries, targets


def write_scores(path, scores):
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["query_id", "target_id", "rank", "raw_score"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(scores)


def soft_hits(query, index, second, margins, threshold):
    """At most one alternate token in each valid query window; hits are deduplicated."""
    hits = set(collect_hits(query, index))
    for start, word in seed_windows(query, index.config.k):
        for offset in range(index.config.k):
            i = start + offset
            if margins[i] <= threshold:
                altered = word[:offset] + ALPHABET[int(second[i])] + word[offset + 1 :]
                hits.update(
                    SeedHit(tid, start, pos) for tid, pos in index.postings.get(altered, ())
                )
    return tuple(sorted(hits))


def filter_scores(scores, queries, targets, arrays, *, threshold=None):
    began = perf_counter()
    index = build_index(targets, IndexConfig(3), allow_real=True)
    index_seconds = perf_counter() - began
    began = perf_counter()
    sets, hit_counts = {}, {}
    for q in queries:
        a = arrays[q.record_id]
        hits = (
            collect_hits(q, index)
            if threshold is None
            else soft_hits(q, index, a["second"], a["margin"], threshold)
        )
        support = supported_diagonals(hits, k=3, window=64, double=True)
        sets[q.record_id] = {index.targets[i].record_id for i in support}
        hit_counts[q.record_id] = len(hits)
    seconds = perf_counter() - began
    grouped = defaultdict(list)
    for s in scores:
        grouped[s["query_id"]].append(s)
    retained = []
    for q in queries:
        top = grouped[q.record_id][:10]
        retained.append(
            sum(s["target_id"] in sets[q.record_id] for s in top) / len(top) if top else 1.0
        )
    filtered = [s for s in scores if s["target_id"] in sets[s["query_id"]]]
    return filtered, {
        "threshold": threshold,
        "retain_exact_at_10": float(np.mean(retained)),
        "candidate_pairs": sum(map(len, sets.values())),
        "candidate_fraction": sum(map(len, sets.values())) / (len(queries) * len(targets)),
        "index_seconds": index_seconds,
        "candidate_seconds": seconds,
        "seed_hit_count": sum(hit_counts.values()),
        "candidates": {k: sorted(v) for k, v in sets.items()},
        "timing_scope": "index and Python candidate generation only; filtered SW not timed",
    }


def independent_recount(scores, qrows, trows, result):
    """No metric helper: independently derive AP and recall from positive ranks."""
    for view in ["all", "cross_family", "same_fold", "other_fold"]:
        values, recalls = [], []
        for q in qrows:
            eligible = {
                t["record_id"]: t
                for t in trows
                if (view != "cross_family" or t["family"] != q["family"])
                and (view != "same_fold" or t["fold"] == q["fold"])
                and (
                    view != "other_fold"
                    or t["fold"] != q["fold"]
                    or t["superfamily"] == q["superfamily"]
                )
            }
            positives = {k for k, t in eligible.items() if t["superfamily"] == q["superfamily"]}
            ranked = sorted(
                [
                    s
                    for s in scores
                    if s["query_id"] == q["record_id"] and s["target_id"] in eligible
                ],
                key=lambda s: (-s["raw_score"], s["target_id"]),
            )
            ranks = [i for i, s in enumerate(ranked, 1) if s["target_id"] in positives]
            values.append(sum(i / rank for i, rank in enumerate(ranks, 1)) / len(positives))
            recalls.append(sum(r <= 10 for r in ranks) / len(positives))
        np.testing.assert_allclose(np.mean(values), result[view]["summary"]["MAP"], atol=1e-12)
        np.testing.assert_allclose(
            np.mean(recalls), result[view]["summary"]["recall_at_10"], atol=1e-12
        )
