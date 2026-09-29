"""Reopen persisted scores and diagnose every nonidentical exported state."""

import csv
from pathlib import Path

import numpy as np

from experiments.common import bounded_cpu, read, save, sha

from .common import OUT


def main():
    out = OUT / "R2"
    bounded_cpu(out)
    import torch

    from experiments.model import PairedVQVAE
    from mini3di_encoder.learned import embed_batch, load_learned_model

    from .evaluation import independent_recount
    from .metrics import cluster_interval

    manifest = read(OUT / "manifest-v2.json")
    curve = read(out / "curve-results.json")
    persisted_checks = []

    def recount(directory, split, metrics, name="scores.tsv"):
        with (directory / name).open() as stream:
            scores = list(csv.DictReader(stream, delimiter="\t"))
        for s in scores:
            s["raw_score"], s["rank"] = int(s["raw_score"]), int(s["rank"])
        qr = [r for r in manifest if r["split"] == split and r["role"] == "query"]
        tr = [r for r in manifest if r["split"] == split and r["role"] == "target"]
        independent_recount(scores, qr, tr, metrics)
        for q in qr:
            actual = [s for s in scores if s["query_id"] == q["record_id"]]
            assert [s["rank"] for s in actual] == list(range(1, len(actual) + 1))
            assert len({s["target_id"] for s in actual}) == len(actual)
        persisted_checks.append(
            {
                "path": str((directory / name).relative_to(OUT)),
                "sha256": sha(directory / name),
                "score_rows": len(scores),
            }
        )

    for run in curve["runs"]:
        for split in ["validation", "test"]:
            recount(out / "search" / run["name"] / split, split, run["splits"][split]["metrics"])
    stability = read(out / "stability-results.json")
    for run in stability["runs"]:
        directory = out / "stability" / run["condition"] / run["model"] / run["split"]
        recount(directory, run["split"], run["exhaustive"]["metrics"])
        for label, value in run["filters"].items():
            recount(directory, run["split"], value["metrics"], f"scores-{label}.tsv")
    training = read(out / "training-results.json")
    export_cases = []
    torch.set_num_threads(1)
    for run in training["runs"]:
        for split, check in run["export_checks"].items():
            if not check["state_mismatches"]:
                continue
            directory = out / f"pairs-{run['structure_pairs']}-seed-{run['seed']}"
            layers, spec = load_learned_model(directory / "encoder.json")
            model = PairedVQVAE()
            model.load_state_dict(torch.load(directory / "checkpoint.pt", weights_only=True))
            model.eval()
            assert split == "train"  # Any unexpected validation mismatch needs an explicit audit.
            with np.load(out / f"pairs-{run['structure_pairs']}.npz") as a:
                x = a["x"]
            with torch.no_grad():
                original = model.encoder(torch.from_numpy(x)).numpy().astype(float)
            deployed = embed_batch(layers, x).astype(float)
            centers = np.array(spec["centroids"])
            before = ((original[:, None] - centers) ** 2).sum(2)
            after = ((deployed[:, None] - centers) ** 2).sum(2)
            mismatches = np.flatnonzero(before.argmin(1) != after.argmin(1))
            assert len(mismatches) == check["state_mismatches"]
            for i in mismatches:
                a, b = int(before[i].argmin()), int(after[i].argmin())
                export_cases.append(
                    {
                        "structure_pairs": run["structure_pairs"],
                        "seed": run["seed"],
                        "split": split,
                        "row": int(i),
                        "torch_state": a,
                        "exported_state": b,
                        "max_latent_delta": float(np.max(np.abs(original[i] - deployed[i]))),
                        "before_distance_gap_to_competitor": float(before[i, b] - before[i, a]),
                        "after_distance_gap_to_competitor": float(after[i, a] - after[i, b]),
                        "matrix_uses_exported_states": True,
                        "interpretation": "BN folding / float32 crossed a center boundary",
                    }
                )
    r1 = read(OUT / "R1-results.json")
    common = {r["variant"]: r for r in r1["rows"] if "common" in r["settings"]}
    paired = {}
    for view in ["all", "cross_family", "same_fold", "other_fold"]:
        ref = {q["query_id"]: q for q in common["official_refit"]["metrics"][view]["per_query"]}
        differences = [
            {"fold": q["fold"], "AP": q["AP"] - ref[q["query_id"]]["AP"]}
            for q in common["learned_refit"]["metrics"][view]["per_query"]
        ]
        paired[view] = cluster_interval(differences)
    # Reproducing official R1 exhaustive scores via batch inference is also a complete cross-check.
    official_stable = next(
        r
        for r in stability["runs"]
        if r["model"] == "official" and r["condition"] == "unchanged" and r["split"] == "test"
    )
    assert official_stable["exhaustive"]["metrics"] == common["official_original"]["metrics"]
    save(
        out / "final-audit.json",
        {
            "passed": True,
            "persisted_tables_checked": persisted_checks,
            "export_boundary_cases": export_cases,
            "R1_paired_fold_difference": paired,
            "R1_official_batch_metrics_exactly_equal": True,
            "script_sha256": sha(Path(__file__)),
        },
    )
    print(
        "Final R2 audit passed",
        len(persisted_checks),
        "score tables; export cases",
        export_cases,
        flush=True,
    )


if __name__ == "__main__":
    main()
