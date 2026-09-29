"""Freeze all trained models before the prespecified repeated-test ablation."""

import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.common import SEARCH_ROOT, bounded_cpu, read, save, sha
from mini3di_encoder.learned import embed_batch, load_learned_model

from .common import OUT


def encoded_arrays(layers, spec, features):
    result = {}
    centers = np.array(spec["centroids"], dtype=np.float64)
    for sid, a in features.items():
        z = embed_batch(layers, a["x"])
        distances = ((z[:, None].astype(float) - centers) ** 2).sum(2)
        order = np.argsort(distances, axis=1, kind="stable")
        first, second = order[:, 0], order[:, 1]
        d1, d2 = distances[np.arange(len(z)), first], distances[np.arange(len(z)), second]
        result[sid] = {
            **a,
            "states": first,
            "second": second,
            "margin": (d2 - d1) / np.maximum(d2, 1e-12),
        }
    return result


def main():
    out = OUT / "R2"
    bounded_cpu(out)
    sys.path.insert(0, str(SEARCH_ROOT / "src"))
    from mini3di_search.align_numba import warmup

    from .evaluation import exhaustive, independent_recount

    training = read(out / "training-results.json")
    features = {}
    manifest = [r for r in read(OUT / "manifest-v2.json") if r["split"] != "train"]
    for r in manifest:
        with np.load(OUT / f"encoded/{r['record_id']}.npz") as a:
            features[r["record_id"]] = {k: a[k] for k in ["x", "mask", "partners"]}
    freeze = {
        "training_results_sha256": sha(out / "training-results.json"),
        "manifest_sha256": sha(OUT / "manifest-v2.json"),
        "script_sha256": sha(__file__),
        "evaluation_sha256": sha(Path(__file__).with_name("evaluation.py")),
        "primary_seed": training["primary_selected_seed"],
        "selection_fixed_before_curve_test": True,
        "models": {},
    }
    for r in training["runs"]:
        name = f"pairs-{r['structure_pairs']}-seed-{r['seed']}"
        freeze["models"][name] = {
            p: sha(out / name / p) for p in ["encoder.json", "substitution.mat"]
        }
    save(out / "search-freeze.json", freeze)
    save(out / "search-jit.json", warmup())
    results, started = [], perf_counter()
    for r in training["runs"]:
        name = f"pairs-{r['structure_pairs']}-seed-{r['seed']}"
        arrays = encoded_arrays(*load_learned_model(out / name / "encoder.json"), features)
        entry = {
            "name": name,
            "structure_pairs": r["structure_pairs"],
            "seed": r["seed"],
            "validation_loss": r["validation_loss"],
            "splits": {},
        }
        for split in ["validation", "test"]:
            qr = [r for r in manifest if r["split"] == split and r["role"] == "query"]
            tr = [r for r in manifest if r["split"] == split and r["role"] == "target"]
            scores, summary, _, _ = exhaustive(
                qr, tr, arrays, out / name / "substitution.mat", out / "search" / name / split
            )
            independent_recount(scores, qr, tr, summary["metrics"])
            entry["splits"][split] = summary
        results.append(entry)
        print(
            name,
            {s: v["metrics"]["all"]["summary"]["MAP"] for s, v in entry["splits"].items()},
            flush=True,
        )
    full = [r for r in results if r["structure_pairs"] == 152]
    alternate = min(
        full,
        key=lambda r: (-r["splits"]["validation"]["metrics"]["all"]["summary"]["MAP"], r["seed"]),
    )
    primary = next(r for r in full if r["seed"] == training["primary_selected_seed"])
    save(
        out / "curve-results.json",
        {
            "runs": results,
            "primary_seed": primary["seed"],
            "alternative_validation_MAP_seed": alternate["seed"],
            "selection_comparison_validation_only": {
                "loss_selected_MAP": primary["splits"]["validation"]["metrics"]["all"]["summary"][
                    "MAP"
                ],
                "MAP_selected_MAP": alternate["splits"]["validation"]["metrics"]["all"]["summary"][
                    "MAP"
                ],
            },
            "independent_recounts_passed": True,
            "wall_seconds": perf_counter() - started,
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "test_reused_for_preregistered_ablation": True,
        },
    )


if __name__ == "__main__":
    main()
