"""Fixed coordinate stress tests and validation-only alternative-seed selection."""

import hashlib
import resource
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.common import SEARCH_ROOT, bounded_cpu, read, save, sha
from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.encode import encode_chain
from mini3di_encoder.learned import load_learned_model
from mini3di_encoder.network import official_model

from .common import OUT
from .evaluate_curve import encoded_arrays


def perturb(xyz, sid, condition):
    result = xyz.copy()
    if condition in {"rigid_float64", "rigid_then_3_decimal_round"}:
        axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        angle = 0.713
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = (
            np.cos(angle) * np.eye(3)
            + (1 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * cross
        )
        result = result @ rotation.T + np.array([13.125, -8.25, 3.5])
        if condition == "rigid_then_3_decimal_round":
            result = np.round(result, 3)
    elif condition.startswith("gaussian_"):
        sigma = float(condition.split("_")[1])
        seed = int.from_bytes(hashlib.sha256(("R2-noise:" + sid).encode()).digest()[:8], "little")
        noise = np.random.default_rng(seed).normal(size=result.shape)
        # Missing atoms remain NaN and are approximated only by the unchanged geometry code.
        result = result + sigma * noise
    elif condition != "unchanged":
        raise ValueError(condition)
    return result


def stability_counts(before, after):
    common = before["mask"] & after["mask"]
    partner = before["partners"] != after["partners"]
    state = before["states"] != after["states"]
    same_partner = common & ~partner
    windows = np.array([before["mask"][i : i + 3].all() for i in range(len(common) - 2)])
    broken = np.array(
        [
            not (
                after["mask"][i : i + 3].all()
                and np.array_equal(before["states"][i : i + 3], after["states"][i : i + 3])
            )
            for i in range(len(common) - 2)
        ]
    )
    result = Counter(
        {
            "positions": len(common),
            "common_valid": int(common.sum()),
            "mask_changes": int(np.count_nonzero(before["mask"] != after["mask"])),
            "partner_changes": int(np.count_nonzero(common & partner)),
            "state_changes": int(np.count_nonzero(common & state)),
            "same_partner_positions": int(same_partner.sum()),
            "state_changes_same_partner": int(np.count_nonzero(same_partner & state)),
            "valid_3mers": int(windows.sum()),
            "broken_3mers": int(np.count_nonzero(windows & broken)),
        }
    )
    for low, high in [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, 1.000001)]:
        selected = same_partner & (before["margin"] >= low) & (before["margin"] < high)
        result[f"margin_{low}_{high}_positions"] = int(selected.sum())
        result[f"margin_{low}_{high}_changes"] = int(np.count_nonzero(selected & state))
    return result


def main():
    out = OUT / "R2"
    bounded_cpu(out)
    sys.path.insert(0, str(SEARCH_ROOT / "src"))
    from mini3di_search.align_numba import warmup

    from .evaluation import exhaustive, filter_scores, independent_recount, write_scores
    from .metrics import evaluate

    protocol = read(out / "protocol.json")
    seed = read(out / "training-results.json")["primary_selected_seed"]
    learned_path = out / f"pairs-152-seed-{seed}/encoder.json"
    models = {"official": official_model(), "learned": load_learned_model(learned_path)}
    matrices = {
        "official": OUT / "matrices/official_original.mat",
        "learned": out / f"pairs-152-seed-{seed}/substitution.mat",
    }
    manifest = [r for r in read(OUT / "manifest-v2.json") if r["split"] != "train"]
    directory = out / "stability"
    directory.mkdir(exist_ok=False)
    save(
        directory / "freeze.json",
        {
            "protocol_sha256": sha(out / "protocol.json"),
            "script_sha256": sha(__file__),
            "evaluation_sha256": sha(Path(__file__).with_name("evaluation.py")),
            "manifest_sha256": sha(OUT / "manifest-v2.json"),
            "encoder_sha256": sha(learned_path),
            "seed": seed,
            "matrix_sha256": {k: sha(v) for k, v in matrices.items()},
            "soft_selection_data": "unchanged validation only; one choice per model",
            "perturbed_data": "both queries and targets; independent per-structure noise seed",
            "rigid_axis_angle_translation": [[1, 2, 3], 0.713, [13.125, -8.25, 3.5]],
            "noise_seed": "first 8 little-endian bytes SHA256('R2-noise:'+record_id)",
            "same_normal_draw_across_noise_scales": True,
            "state_rate_denominator": "joint safe mask; same-partner subset explicitly separate",
            "first_condition_and_selection_fixed_before_perturbed_search": True,
        },
    )
    save(directory / "jit.json", warmup())
    base_features = {}
    for row in manifest:
        sid = row["record_id"]
        with np.load(OUT / f"encoded/{sid}.npz") as a:
            base_features[sid] = {k: a[k] for k in ["x", "mask", "partners"]}
    baseline = {name: encoded_arrays(*model, base_features) for name, model in models.items()}
    # Cross-check batched official inference against the saved full per-residue implementation.
    for sid, a in baseline["official"].items():
        with np.load(OUT / f"encoded/{sid}.npz") as original:
            assert np.array_equal(a["states"][a["mask"]], original["official_states"][a["mask"]])
    selection, outputs, started = {}, [], perf_counter()
    for condition in protocol["coordinate_conditions"]:
        began = perf_counter()
        features = {}
        if condition == "unchanged":
            features = base_features
        else:
            cache = directory / condition / "features"
            cache.mkdir(parents=True)
            for row in manifest:
                sid = row["record_id"]
                chain = read_backbone(OUT / row["path"], row["chain"])
                encoded = encode_chain(replace(chain, xyz=perturb(chain.xyz, sid, condition)), sid)
                a = {
                    "x": np.array(
                        [r["features"] if r["valid"] else [0.0] * 10 for r in encoded["rows"]]
                    ),
                    "mask": np.array(encoded["valid_seed_mask"]),
                    "partners": np.array(
                        [r["partner"] if r["partner"] is not None else -1 for r in encoded["rows"]]
                    ),
                }
                np.savez_compressed(cache / f"{sid}.npz", **a)
                features[sid] = a
        encode_seconds = perf_counter() - began
        for name, model in models.items():
            arrays = (
                baseline[name] if condition == "unchanged" else encoded_arrays(*model, features)
            )
            for split in ["validation", "test"]:
                qr = [r for r in manifest if r["split"] == split and r["role"] == "query"]
                tr = [r for r in manifest if r["split"] == split and r["role"] == "target"]
                dest = directory / condition / name / split
                scores, summary, queries, targets = exhaustive(qr, tr, arrays, matrices[name], dest)
                independent_recount(scores, qr, tr, summary["metrics"])
                counts = Counter()
                per_structure = []
                for row in qr + tr:
                    sid = row["record_id"]
                    changes = stability_counts(baseline[name][sid], arrays[sid])
                    counts.update(changes)
                    per_structure.append({"record_id": sid, "role": row["role"], **changes})
                if condition == "unchanged" and split == "validation":
                    trials = []
                    for threshold in protocol["soft_seed_validation"]["thresholds"]:
                        filtered, stats = filter_scores(
                            scores, queries, targets, arrays, threshold=threshold
                        )
                        stats["MAP"] = evaluate(filtered, qr, tr)["all"]["summary"]["MAP"]
                        trials.append(stats)
                    acceptable = [r for r in trials if r["retain_exact_at_10"] >= 0.98]
                    chosen = (
                        min(acceptable, key=lambda r: (r["candidate_pairs"], r["threshold"]))
                        if acceptable
                        else None
                    )
                    selection[name] = {
                        "trials": trials,
                        "threshold": chosen["threshold"] if chosen else None,
                        "passed": chosen is not None,
                    }
                    save(directory / "soft-selection.json", selection)
                configs = [("double", None)]
                if selection[name]["passed"]:
                    configs.append(("soft", selection[name]["threshold"]))
                filters = {}
                for label, threshold in configs:
                    filtered, stats = filter_scores(
                        scores, queries, targets, arrays, threshold=threshold
                    )
                    m = evaluate(filtered, qr, tr)
                    independent_recount(filtered, qr, tr, m)
                    stats["metrics"] = m
                    filters[label] = stats
                    write_scores(dest / f"scores-{label}.tsv", filtered)
                output = {
                    "condition": condition,
                    "model": name,
                    "split": split,
                    "counts": dict(counts),
                    "exhaustive": summary,
                    "filters": filters,
                    "condition_feature_seconds": encode_seconds,
                }
                save(dest / "stability.json", {**output, "per_structure": per_structure})
                outputs.append(output)
                print(
                    condition,
                    name,
                    split,
                    "MAP",
                    summary["metrics"]["all"]["summary"]["MAP"],
                    "state changes",
                    counts["state_changes"],
                    "soft",
                    selection[name]["passed"],
                    flush=True,
                )
        if perf_counter() - started > protocol["limits"]["cpu_seconds"]:
            raise RuntimeError("Stability run exceeded preregistered time ceiling")
    save(
        out / "stability-results.json",
        {
            "runs": outputs,
            "selection": selection,
            "independent_recounts_passed": True,
            "wall_seconds": perf_counter() - started,
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


if __name__ == "__main__":
    main()
