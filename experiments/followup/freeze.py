"""Freeze an audited dataset and the outcome-independent R1 comparison protocol."""

import itertools
import shutil
from collections import Counter
from pathlib import Path

from experiments.common import ROOT, read, save, sha

from .common import ORIGINAL, OUT


def main():
    out = OUT
    if (out / "data-freeze.json").exists():
        raise FileExistsError("Dataset already frozen")
    audit = read(out / "similarity-summary.json")
    assert audit["completed"] == audit["sequence_pairs"]
    assert sha(out / "similarity-journal.jsonl") == audit["journal_sha256"]
    assert not audit["flags"], "Near duplicates need explicit blind-data repair before freezing"
    assert sha(out / "candidate-manifest.json") == audit["candidate_manifest_sha256"]
    rows = read(out / "candidate-manifest.json")
    old = read(ORIGINAL / "manifest.json")
    test = [r for r in rows if r["split"] == "test"]
    for key in ["fold", "pdb", "aa_sha256", "shape_sha256", "structure_sha256"]:
        assert not {r[key] for r in test} & {r[key] for r in old}
    for key in ["fold", "superfamily", "pdb", "aa_sha256", "shape_sha256", "structure_sha256"]:
        sets = [
            {r[key] for r in rows if r["split"] == split}
            for split in ["train", "validation", "test"]
        ]
        assert all(not a & b for a, b in itertools.combinations(sets, 2)), key
    assert len([r for r in test if r["role"] == "query"]) == 100
    assert len({r["superfamily"] for r in test if r["role"] == "query"}) >= 30
    assert len([r for r in test if r["role"] == "target"]) <= 1000
    shutil.copyfile(out / "candidate-manifest.json", out / "manifest-v2.json")
    shutil.copyfile(out / "candidate-denominators.json", out / "denominators-v2.json")
    source = ORIGINAL / "matrices"
    (out / "matrices").mkdir()
    for variant in ["official_original", "official_refit", "learned_refit"]:
        shutil.copyfile(source / f"{variant}.mat", out / f"matrices/{variant}.mat")
    protocol = {
        "experiment": "R1 hard-negative evaluation of three fixed representations/matrices",
        "common_gap": [10, 1],
        "validation_gap_grid": [[6, 1], [10, 1], [14, 2]],
        "gap_selection": "max validation all-view query MAP; then smaller open/extend",
        "model_selection": "Previous seed29/epoch39 fixed before this experiment; no R1 retraining",
        "variants": ["official_original", "official_refit", "learned_refit"],
        "views": ["all", "cross_family", "same_fold", "other_fold"],
        "primary": "All-view query-macro MAP and macro Recall@10; same SF positives",
        "cross_family": "Remove same-family targets entirely; remaining same-SF targets positive",
        "same_fold": "Keep only same-fold targets; other SFs are the hard-negative class",
        "other_fold": "Keep same-SF positives and other-fold negatives; exclude hard negatives",
        "published_absent_stratum": (
            "Query SID and PDB membership absence separately; not proof of weight non-exposure"
        ),
        "ties": "Raw score descending then record_id ascending; tie-expected AP secondary",
        "zero_score": "Do not retrieve score-zero pairs; missing positives remain in denominators",
        "bootstrap": {
            "unit": "fold",
            "repeats": 2000,
            "seed": 20260929,
            "statistic": "query-weighted ratio after whole-fold resampling",
        },
        "pr_curve": (
            "Macro mean of per-query 101-point interpolated precision envelopes; not MAP area"
        ),
        "backend": "Existing read-only mini-3di-search int64 Numba rolling SW, 1 compute thread",
        "traceback": "Python top1 with independent path rescoring; all positive scores retained",
        "timing_repeats": 3,
        "cpu_limit_seconds": 7200,
        "memory_limit_bytes": 4 * 1024**3,
        "preflight_score_cells": 1_000_000_000,
        "max_search_cells_per_condition": 4_000_000_000,
        "test_single_evaluation": "One frozen test evaluation; exact repeats only for time",
        "R2_gate": (
            "Common-gap learned MAP at least 0.02 below official_refit, "
            "or concrete encoding/ranking failures warrant analysis"
        ),
        "R3_gate": (
            "Candidate/ungapped/traceback overhead remains dominant "
            "in controlled timing; benchmark separately"
        ),
    }
    save(out / "protocol-v2.json", protocol)
    shutil.copyfile(ROOT / "FOLLOWUP_PLAN.md", out / "plan-at-data-freeze.md")
    save(
        out / "data-freeze.json",
        {
            "manifest_sha256": sha(out / "manifest-v2.json"),
            "denominators_sha256": sha(out / "denominators-v2.json"),
            "similarity_summary_sha256": sha(out / "similarity-summary.json"),
            "similarity_journal_sha256": sha(out / "similarity-journal.jsonl"),
            "protocol_sha256": sha(out / "protocol-v2.json"),
            "learned_encoder_sha256": sha(ORIGINAL / "models/seed-29/encoder.json"),
            "matrices": {p.name: sha(p) for p in (out / "matrices").glob("*.mat")},
            "counts": dict(Counter(r["split"] + ":" + r["role"] for r in rows)),
            "test_search_has_not_run": True,
            "script_sha256": sha(Path(__file__)),
            "plan_sha256": sha(out / "plan-at-data-freeze.md"),
        },
    )
    print("R1 data and protocol frozen", flush=True)


if __name__ == "__main__":
    main()
