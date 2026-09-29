"""Audit all length-eligible cross-split pairs for near sequence/structure duplication."""

import itertools
import json
import re
import resource
import subprocess
from pathlib import Path
from time import perf_counter

from experiments.common import read, save, sha

from .common import OUT, order


def main():
    from Bio.Align import PairwiseAligner

    out = OUT
    rows = read(out / "candidate-manifest.json")
    old = read(out / "previous-exposure-manifest.json")
    by_id = {r["record_id"]: r for r in rows + old}
    splits = {s: [r for r in rows if r["split"] == s] for s in ["train", "validation", "test"]}
    pairs = set()
    for left, right in itertools.combinations(splits, 2):
        pairs.update(
            tuple(sorted((a["record_id"], b["record_id"])))
            for a in splits[left]
            for b in splits[right]
        )
    pairs.update(
        tuple(sorted((a["record_id"], b["record_id"]))) for a in old for b in splits["test"]
    )
    ordered = sorted(pairs, key=lambda p: order(":".join(p)))
    lengths = {sid: row["length"] for sid, row in by_id.items()}
    checks = []
    for a, b in ordered:
        ratio = min(lengths[a], lengths[b]) / max(lengths[a], lengths[b])
        if ratio >= 0.9:
            checks.append((a, b, ratio >= 0.95))
    contract = {
        "candidate_manifest_sha256": sha(out / "candidate-manifest.json"),
        "exposure_manifest_sha256": sha(out / "previous-exposure-manifest.json"),
        "script_sha256": sha(Path(__file__)),
        "tmalign_sha256": sha(out / "TMalign"),
        "total_cross_pairs": len(pairs),
        "sequence_pairs": len(checks),
        "structure_pairs": sum(x[2] for x in checks),
        "identity": 0.8,
        "both_sequence_coverage": 0.9,
        "min_bidirectional_TM": 0.95,
        "filter": "Length ratio is a necessary bound, not an approximate similarity filter",
    }
    plan_path = out / "similarity-contract.json"
    if plan_path.exists():
        assert read(plan_path) == contract
    else:
        save(plan_path, contract)
    print(contract, flush=True)
    path = out / "similarity-journal.jsonl"
    completed = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            completed[(r["a"], r["b"])] = r
    aligner = PairwiseAligner(
        mode="global", match_score=2, mismatch_score=-1, open_gap_score=-10, extend_gap_score=-1
    )
    started = perf_counter()
    with path.open("a", buffering=1) as journal:
        for number, (a, b, structure_check) in enumerate(checks, 1):
            if (a, b) in completed:
                continue
            aa, bb = by_id[a]["aa"], by_id[b]["aa"]
            began = perf_counter()
            alignment = aligner.align(aa, bb)[0]
            count = matched = 0
            for (ai, aj), (bi, bj) in zip(*alignment.aligned, strict=True):
                assert aj - ai == bj - bi
                count += int(aj - ai)
                matched += sum(
                    x == y and x != "X" for x, y in zip(aa[ai:aj], bb[bi:bj], strict=True)
                )
            identity = matched / count if count else 0.0
            covers = [count / len(aa), count / len(bb)]
            scores, seconds = None, 0.0
            if structure_check:
                start_tm = perf_counter()
                process = subprocess.run(
                    [
                        out / "TMalign",
                        out / f"structures/{a}.pdb",
                        out / f"structures/{b}.pdb",
                        "-ter",
                        "0",
                        "-mol",
                        "protein",
                        "-outfmt",
                        "0",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                process.check_returncode()
                seconds = perf_counter() - start_tm
                scores = [float(x) for x in re.findall(r"TM-score=\s*([0-9.]+)", process.stdout)]
                assert len(scores) == 2
                # Full raw output only for flagged or deterministic audit sample pairs.
                if min(scores) >= 0.95 or number % 100 == 0:
                    save(
                        out / f"similarity-raw/{a}__{b}.json",
                        {"stdout": process.stdout, "stderr": process.stderr},
                    )
            result = {
                "a": a,
                "b": b,
                "identity": identity,
                "coverage": covers,
                "tm_scores": scores,
                "near_sequence": identity >= 0.8 and min(covers) >= 0.9,
                "near_structure": scores is not None and min(scores) >= 0.95,
                "seconds": perf_counter() - began,
                "tmalign_seconds": seconds,
            }
            completed[(a, b)] = result
            journal.write(json.dumps(result) + "\n")
            if number % 250 == 0:
                elapsed = perf_counter() - started
                print(
                    "similarity", number, "/", len(checks), "elapsed", round(elapsed, 1), flush=True
                )
                save(
                    out / "similarity-progress.json",
                    {
                        "completed": len(completed),
                        "required": len(checks),
                        "elapsed_seconds_this_invocation": elapsed,
                    },
                )
                if elapsed > 5400:
                    raise RuntimeError("Similarity audit reached the preallocated 90-minute limit")
    assert len(completed) == len(checks)
    flags = [r for r in completed.values() if r["near_sequence"] or r["near_structure"]]
    save(
        out / "similarity-summary.json",
        {
            **contract,
            "completed": len(completed),
            "flags": flags,
            "sequence_length_excluded_pairs": len(pairs) - len(checks),
            "structure_length_excluded_pairs": len(pairs) - sum(x[2] for x in checks),
            "wall_seconds_this_invocation": perf_counter() - started,
            "tmalign_seconds_sum": sum(r["tmalign_seconds"] for r in completed.values()),
            "peak_parent_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "journal_sha256": sha(path),
        },
    )
    print("similarity audit complete; flagged", len(flags), flush=True)


if __name__ == "__main__":
    main()
