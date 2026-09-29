"""Cache fixed geometric features, then make masked paired training examples."""

import argparse
import re
import resource
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.encode import encode_chain

from .common import DEFAULT_OUT, bounded_cpu, command, read, save, sha


def parse_alignment(text, aa, bb):
    scores = [float(v) for v in re.findall(r"TM-score=\s*([0-9.]+)", text)]
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "denotes residue pairs" in line)
    first, marks, second = lines[start + 1 : start + 4]
    if first.replace("-", "") != aa or second.replace("-", "") != bb:
        raise ValueError("TM-align/parser residue correspondence differs")
    if len(first) != len(second) or len(marks) != len(first) or len(scores) != 2:
        raise ValueError("Unexpected TM-align output shape")
    i = j = 0
    pairs = []
    for a, mark, b in zip(first, marks, second, strict=True):
        if mark == ":":
            if a == "-" or b == "-":
                raise ValueError("A distance match cannot contain a gap")
            pairs.append((i, j))
        i += a != "-"
        j += b != "-"
    return scores, np.array(pairs, dtype=np.int64).reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    bounded_cpu(out)
    records = read(out / "manifest.json")
    freeze = read(out / "data-freeze.json")
    config = read(out / "experiment_config.json")
    assert sha(out / "manifest.json") == freeze["manifest_sha256"]
    assert sha(out / "pair_plan.json") == freeze["pair_plan_sha256"]
    assert sha(out / "experiment_config.json") == freeze["experiment_config_sha256"]
    features = out / "features"
    features.mkdir(exist_ok=False)
    cache, encode_seconds = {}, 0.0
    for n, row in enumerate(records, 1):
        path = out / row["path"]
        assert sha(path) == row["structure_sha256"]
        chain = read_backbone(path, row["chain"])
        assert chain.aa == row["aa"] and len(chain.residues) == row["length"]
        start = perf_counter()
        encoded = encode_chain(chain, row["record_id"])
        encode_seconds += perf_counter() - start
        x = np.array([r["features"] if r["valid"] else [0.0] * 10 for r in encoded["rows"]])
        mask = np.array(encoded["valid_seed_mask"], dtype=bool)
        arrays = {
            "x": x,
            "mask": mask,
            "states": np.array([r["state"] for r in encoded["rows"]]),
            "feature_mask": np.array(encoded["feature_valid_mask"], dtype=bool),
            "aa": np.array(chain.aa),
            "pdb_numbers": np.array([r[0] for r in chain.residues]),
            "ca": chain.xyz[:, 1],
        }
        np.savez_compressed(features / f"{row['record_id']}.npz", **arrays)
        cache[row["record_id"]] = arrays
        if n % 64 == 0:
            print("encoded", n, flush=True)
    alignment_dir = out / "alignments"
    alignment_dir.mkdir(exist_ok=False)
    rows_by_id = {r["record_id"]: r for r in records}
    tmalign = out / "tools/usalign/TMalign"
    audit, examples = [], {"train": [], "validation": []}
    for number, pair in enumerate(read(out / "pair_plan.json"), 1):
        a, b, split = pair["a"], pair["b"], pair["split"]
        assert rows_by_id[a]["split"] == rows_by_id[b]["split"] == split != "test"
        assert rows_by_id[a]["superfamily"] == rows_by_id[b]["superfamily"]
        text = command(
            [
                tmalign,
                out / rows_by_id[a]["path"],
                out / rows_by_id[b]["path"],
                "-ter",
                "0",
                "-mol",
                "protein",
                "-outfmt",
                "0",
            ],
            alignment_dir / f"{a}__{b}.json",
            timeout=30,
        )
        scores, indices = parse_alignment(text, rows_by_id[a]["aa"], rows_by_id[b]["aa"])
        valid = cache[a]["mask"][indices[:, 0]] & cache[b]["mask"][indices[:, 1]]
        usable = indices[valid]
        accepted = min(scores) >= config["pair_min_tm"] and len(usable) > 0
        audit.append(
            {
                **pair,
                "tm_scores": scores,
                "close_pairs": len(indices),
                "valid_pairs": len(usable),
                "accepted": accepted,
                "reason": "accepted" if accepted else "tm_or_no_valid_pairs",
                "indices": usable.tolist() if accepted else [],
            }
        )
        if accepted:
            xa, xb = cache[a]["x"][usable[:, 0]], cache[b]["x"][usable[:, 1]]
            examples[split].append((xa, xb))
        if number % 80 == 0:
            print("aligned", number, flush=True)
    for split, chunks in examples.items():
        if not chunks:
            raise ValueError(f"No paired examples for {split}; dataset construction failed")
        x = np.concatenate([a for a, b in chunks] + [b for a, b in chunks]).astype(np.float32)
        y = np.concatenate([b for a, b in chunks] + [a for a, b in chunks]).astype(np.float32)
        np.savez_compressed(out / f"pairs-{split}.npz", x=x, y=y)
    save(out / "alignment_audit.json", audit)
    save(
        out / "feature_summary.json",
        {
            "structures": len(records),
            "residues": sum(r["length"] for r in records),
            "encode_seconds": encode_seconds,
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "attempted_pairs": dict(Counter(r["split"] for r in audit)),
            "accepted_pairs": dict(Counter(r["split"] for r in audit if r["accepted"])),
            "directed_feature_pairs": {
                s: int(np.load(out / f"pairs-{s}.npz")["x"].shape[0]) for s in examples
            },
            "accepted_superfamilies": {
                s: len({r["superfamily"] for r in audit if r["split"] == s and r["accepted"]})
                for s in examples
            },
            "tmalign_sha256": sha(tmalign),
            "script_sha256": sha(Path(__file__)),
            "mask_policy": "Both strict search masks true; ':' distances <5A; symmetric directions",
            "feature_files": {p.name: sha(p) for p in sorted(features.glob("*.npz"))},
            "test_used_for_training_pairs": False,
        },
    )
    print(read(out / "feature_summary.json")["directed_feature_pairs"], flush=True)


if __name__ == "__main__":
    main()
