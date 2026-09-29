"""Independently recount frozen scores, alignments, matrices and exported states."""

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

from .common import DEFAULT_OUT, ROOT, SEARCH_ROOT, bounded_cpu, read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    bounded_cpu(out)
    import numpy as np
    import torch

    from mini3di_encoder.atoms import read_backbone
    from mini3di_encoder.learned import embed_batch, load_learned_model, predict_states

    from .model import PairedVQVAE

    torch.set_num_threads(1)
    freeze = read(out / "test-freeze.json")
    for name, digest in freeze["files"].items():
        assert sha(out / name) == digest, name
    for name, key in [("evaluate.py", "script_sha256"), ("metrics.py", "metrics_code_sha256")]:
        assert sha(ROOT / "experiments" / name) == freeze[key]
    manifest = read(out / "manifest.json")
    by_id = {r["record_id"]: r for r in manifest}
    assert len(by_id) == len(manifest) == 384
    for key in ["fold", "superfamily", "pdb", "aa_sha256", "shape_sha256", "structure_sha256"]:
        groups = [
            {r[key] for r in manifest if r["split"] == s} for s in ["train", "validation", "test"]
        ]
        assert all(not (a & b) for a, b in itertools.combinations(groups, 2)), key
    feature_files = read(out / "feature_summary.json")["feature_files"]
    cache = {}
    for row in manifest:
        sid = row["record_id"]
        assert sha(out / row["path"]) == row["structure_sha256"]
        chain = read_backbone(out / row["path"], row["chain"])
        assert chain.aa == row["aa"] and len(chain.residues) == row["length"]
        assert chain.backbone_valid.all() and np.mean(chain.peptide_links()) >= 0.95
        assert sha(out / f"features/{sid}.npz") == feature_files[f"{sid}.npz"]
        with np.load(out / f"features/{sid}.npz") as data:
            cache[sid] = {key: data[key] for key in data.files}

    seed = freeze["selected_seed"]
    layers, spec = load_learned_model(out / f"models/seed-{seed}/encoder.json")
    model = PairedVQVAE()
    model.load_state_dict(torch.load(out / f"models/seed-{seed}/checkpoint.pt", weights_only=True))
    model.eval()
    mismatches = tested = 0
    max_error = 0.0
    for data in cache.values():
        data["learned"] = predict_states(layers, spec, data["x"])
        x = data["x"][data["feature_mask"]].astype(np.float32)
        with torch.no_grad():
            latent = model.encoder(torch.from_numpy(x)).numpy()
        deployed = embed_batch(layers, x)
        centers = np.array(spec["centroids"])
        states = np.argmin(np.sum((latent[:, None].astype(float) - centers) ** 2, 2), 1)
        mismatches += int(np.count_nonzero(states != predict_states(layers, spec, x)))
        tested += len(x)
        max_error = max(max_error, float(np.max(np.abs(latent - deployed))))
    assert mismatches == 0 and max_error < 1e-4

    # Read the raw TM-align alignment anew; do not call features.parse_alignment.
    audit = read(out / "alignment_audit.json")
    chunks = defaultdict(list)
    counts = {k: np.zeros((20, 20), dtype=np.int64) for k in ["official_refit", "learned_refit"]}
    for pair in audit:
        a, b, split = pair["a"], pair["b"], pair["split"]
        assert by_id[a]["split"] == by_id[b]["split"] == split != "test"
        assert by_id[a]["superfamily"] == by_id[b]["superfamily"]
        lines = read(out / f"alignments/{a}__{b}.json")["stdout"].splitlines()
        offset = next(i for i, line in enumerate(lines) if "denotes residue pairs" in line)
        aa, marks, bb = lines[offset + 1 : offset + 4]
        assert aa.replace("-", "") == by_id[a]["aa"]
        assert bb.replace("-", "") == by_id[b]["aa"]
        scores = [
            float(line.split("=")[1].split()[0]) for line in lines if line.startswith("TM-score=")
        ]
        assert scores == pair["tm_scores"]
        ii = jj = 0
        valid = []
        for left, mark, right in zip(aa, marks, bb, strict=True):
            if mark == ":" and cache[a]["mask"][ii] and cache[b]["mask"][jj]:
                assert left != "-" and right != "-"
                valid.append([ii, jj])
            ii += left != "-"
            jj += right != "-"
        accepted = min(scores) >= 0.6 and bool(valid)
        assert accepted == pair["accepted"]
        assert (valid if accepted else []) == pair["indices"]
        if not accepted:
            continue
        xy = [(cache[a]["x"][i], cache[b]["x"][j]) for i, j in valid]
        chunks[split].extend(xy)
        if split == "train":
            for i, j in valid:
                for variant, key in [("official_refit", "states"), ("learned_refit", "learned")]:
                    u, v = cache[a][key][i], cache[b][key][j]
                    counts[variant][u, v] += 1
                    counts[variant][v, u] += 1
    for split, pairs in chunks.items():
        x = np.array([a for a, b in pairs] + [b for a, b in pairs], dtype=np.float32)
        y = np.array([b for a, b in pairs] + [a for a, b in pairs], dtype=np.float32)
        with np.load(out / f"pairs-{split}.npz") as data:
            assert np.array_equal(x, data["x"]) and np.array_equal(y, data["y"])
    for variant, table in counts.items():
        assert np.array_equal(table, np.load(out / f"matrices/{variant}-counts.npy"))
        matrix = np.loadtxt(
            out / f"matrices/{variant}.mat", skiprows=1, usecols=range(1, 22), dtype=int
        )
        total = int(table.sum()) + 400 * 0.5
        for i, j in itertools.product(range(20), repeat=2):
            joint = (int(table[i, j]) + 0.5) / total
            pi = (int(table[i].sum()) + 20 * 0.5) / total
            pj = (int(table[j].sum()) + 20 * 0.5) / total
            assert matrix[i, j] == round(2 * math.log2(joint / (pi * pj)))
        assert not matrix[20].any() and not matrix[:, 20].any()

    summary = read(out / "search-test/summary.json")
    for name, digest in summary["search_source_sha256"].items():
        assert sha(SEARCH_ROOT / "src/mini3di_search" / name) == digest
    queries = [r for r in manifest if r["split"] == "test" and r["role"] == "query"]
    targets = [r for r in manifest if r["split"] == "test" and r["role"] == "target"]
    recounted = []
    failures = []
    for row in summary["rows"]:
        variant, mode = row["variant"], row["mode"]
        exact_path = out / f"search-test/r0-{variant}-exhaustive/scores.tsv"
        with exact_path.open() as stream:
            exact = list(csv.DictReader(stream, delimiter="\t"))
        all_exact = {(r["query_id"], r["target_id"]): int(r["raw_score"]) for r in exact}
        for repeat in range(3):
            directory = out / f"search-test/r{repeat}-{variant}-{mode}"
            report = read(directory / "report.json")
            with (directory / "scores.tsv").open() as stream:
                hits = list(csv.DictReader(stream, delimiter="\t"))
            assert len(hits) == len({(r["query_id"], r["target_id"]) for r in hits})
            assert all(
                int(h["raw_score"]) == all_exact[h["query_id"], h["target_id"]] for h in hits
            )
            candidates = {d["query_id"]: set(d["candidate_ids"]) for d in report["diagnostics"]}
            aps, retained = [], 0
            for query in queries:
                qid = query["record_id"]
                relevant = {
                    t["record_id"] for t in targets if t["superfamily"] == query["superfamily"]
                }
                ranked = sorted(
                    (h for h in hits if h["query_id"] == qid),
                    key=lambda h: (-int(h["raw_score"]), h["target_id"]),
                )
                flags = [h["target_id"] in relevant for h in ranked]
                ap = sum(sum(flags[:i]) / i for i, ok in enumerate(flags, 1) if ok) / len(relevant)
                aps.append(ap)
                per_query = next(p for p in report["per_query"] if p["query_id"] == qid)
                assert abs(ap - per_query["AP"]) < 1e-12
                for k in [1, 5, 10]:
                    assert sum(flags[:k]) / len(relevant) == per_query[f"recall_at_{k}"]
                exact_top = sorted(
                    (h for h in exact if h["query_id"] == qid),
                    key=lambda h: (-int(h["raw_score"]), h["target_id"]),
                )[:10]
                retained += len({h["target_id"] for h in exact_top} & candidates[qid])
                if repeat == 0 and (ap < 0.999999 or not relevant <= candidates[qid]):
                    failures.append(
                        {
                            "variant": variant,
                            "mode": mode,
                            "query": qid,
                            "superfamily": query["superfamily"],
                            "length": query["length"],
                            "AP": ap,
                            "missed_relevant_candidates": sorted(relevant - candidates[qid]),
                            "relevant_ranks": {
                                h["target_id"]: i
                                for i, h in enumerate(ranked, 1)
                                if h["target_id"] in relevant
                            },
                        }
                    )
            assert abs(sum(aps) / len(aps) - row["MAP"]) < 1e-12
            assert retained == row["retained_exact_top10"]
            recounted.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "repeat": repeat,
                    "MAP": sum(aps) / len(aps),
                    "retained": retained,
                    "score_rows": len(hits),
                }
            )
    save(
        out / "independent_audit.json",
        {
            "passed": True,
            "structures_checked": len(manifest),
            "alignment_outputs_checked": len(audit),
            "checkpoint_export": {
                "valid_features": tested,
                "max_embedding_error": max_error,
                "state_mismatches": mismatches,
            },
            "matrix_cells_recomputed": 800,
            "recounted_runs": recounted,
            "failures": failures,
            "scope": (
                "Separate score/AP/matrix/alignment recount; "
                "same parser and model for geometry/export checks"
            ),
            "script_sha256": sha(Path(__file__)),
            "freeze_sha256": sha(out / "test-freeze.json"),
        },
    )
    print("independent audit passed", len(recounted), "runs; exported states", tested, flush=True)


if __name__ == "__main__":
    main()
