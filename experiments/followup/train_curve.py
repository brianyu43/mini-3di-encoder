"""Execute the preregistered 3 data fractions x 5 seeds at an equal step budget."""

import copy
import resource
from pathlib import Path
from time import perf_counter

from experiments.common import ROOT, bounded_cpu, read, save, sha

from .common import ORIGINAL, OUT
from .sampling import nested_pair_subsets


def main():
    gate = read(OUT / "continuation-gates.json")
    if not (gate["R2_MAP_gate_triggered"] or gate["R2_failure_examples"]):
        raise ValueError("R2 continuation gate not established")
    out = OUT / "R2"
    out.mkdir(exist_ok=False)
    bounded_cpu(out)
    import numpy as np
    import torch

    from experiments.matrices import estimate_matrix, write_matrix
    from experiments.model import PairedVQVAE, export_model
    from mini3di_encoder.learned import embed_batch, load_learned_model, predict_states

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol_path = Path(__file__).with_name("R2_PREREGISTRATION.json")
    protocol = read(protocol_path)
    save(out / "protocol.json", protocol)
    pairs = [
        r
        for r in read(ORIGINAL / "alignment_audit.json")
        if r["split"] == "train" and r["accepted"]
    ]
    subsets = nested_pair_subsets(pairs, protocol["structure_pair_counts"])
    cache = {}
    for row in read(ORIGINAL / "manifest.json"):
        if row["split"] == "train":
            with np.load(ORIGINAL / f"features/{row['record_id']}.npz") as arrays:
                cache[row["record_id"]] = arrays["x"].astype(np.float32)
    with np.load(ORIGINAL / "pairs-validation.npz") as arrays:
        validation = tuple(torch.from_numpy(arrays[k]) for k in ["x", "y"])
    datasets = {}
    dataset_summaries = {}
    for count, selected in subsets.items():
        a = np.concatenate([cache[p["a"]][np.array(p["indices"])[:, 0]] for p in selected])
        b = np.concatenate([cache[p["b"]][np.array(p["indices"])[:, 1]] for p in selected])
        x, y = np.concatenate([a, b]), np.concatenate([b, a])
        np.savez_compressed(out / f"pairs-{count}.npz", x=x, y=y)
        datasets[count] = tuple(torch.from_numpy(arr) for arr in [x, y])
        dataset_summaries[str(count)] = {
            "structure_pairs": count,
            "directed_residue_pairs": len(x),
            "superfamilies": len({p["superfamily"] for p in selected}),
            "pair_ids": [[p["a"], p["b"]] for p in selected],
            "sha256": sha(out / f"pairs-{count}.npz"),
        }
    save(
        out / "data-freeze.json",
        {
            "datasets": dataset_summaries,
            "source_alignment_audit_sha256": sha(ORIGINAL / "alignment_audit.json"),
            "original_validation_pairs_sha256": sha(ORIGINAL / "pairs-validation.npz"),
            "protocol_sha256": sha(protocol_path),
            "trainer_sha256": sha(__file__),
            "sampling_sha256": sha(Path(__file__).with_name("sampling.py")),
            "search_manifest_sha256": sha(OUT / "manifest-v2.json"),
            "test_used_for_selection": False,
        },
    )
    started = perf_counter()
    runs = []

    def evaluate(model, arrays):
        model.eval()
        x, y = arrays
        totals = np.zeros(3)
        counts = np.zeros(20, dtype=np.int64)
        with torch.no_grad():
            for begin in range(0, len(x), 4096):
                a, b = x[begin : begin + 4096], y[begin : begin + 4096]
                loss, nll, vq, ids = model.objective(a, b)
                totals += len(a) * np.array([loss.item(), nll.item(), vq.item()])
                counts += np.bincount(ids.numpy(), minlength=20)
        probabilities = counts[counts > 0] / counts.sum()
        return {
            "loss": float(totals[0] / len(x)),
            "nll": float(totals[1] / len(x)),
            "vq_loss": float(totals[2] / len(x)),
            "state_counts": counts.tolist(),
            "states_used": int(np.count_nonzero(counts)),
            "perplexity": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        }

    for count, train in datasets.items():
        for seed in protocol["seeds"]:
            if perf_counter() - started > protocol["limits"]["cpu_seconds"]:
                raise RuntimeError("R2 CPU budget exceeded")
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = PairedVQVAE()
            optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
            best, checkpoint, best_step = float("inf"), None, None
            history = []
            began = perf_counter()
            for step in range(protocol["optimizer_steps"] + 1):
                if step:
                    model.train()
                    indices = rng.integers(0, len(train[0]), size=protocol["batch_size"])
                    optimizer.zero_grad(set_to_none=True)
                    loss, _, _, _ = model.objective(train[0][indices], train[1][indices])
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite training loss: {count}, {seed}, {step}")
                    loss.backward()
                    optimizer.step()
                if step % protocol["checkpoint_every_steps"] == 0:
                    train_metrics, validation_metrics = (
                        evaluate(model, train),
                        evaluate(model, validation),
                    )
                    history.append(
                        {"step": step, "train": train_metrics, "validation": validation_metrics}
                    )
                    if step and validation_metrics["loss"] < best:
                        best, best_step = validation_metrics["loss"], step
                        checkpoint = copy.deepcopy(model.state_dict())
            model.load_state_dict(checkpoint)
            model.eval()
            directory = out / f"pairs-{count}-seed-{seed}"
            directory.mkdir()
            torch.save(checkpoint, directory / "checkpoint.pt")
            save(directory / "history.json", history)
            model_path = directory / "encoder.json"
            save(
                model_path,
                export_model(
                    model,
                    {
                        "geometry_upstream_commit": "941cd33ff0771cd2e3f144e3293e22a2b87e9fda",
                        "training_source_commit": "654100b11242e581f9e6d43798b07a778903862e",
                        "experiment": "R2 fixed-step nested-pair learning curve",
                        "structure_pairs": count,
                        "seed": seed,
                        "selected_step": best_step,
                        "selection": "paired validation total loss",
                        "training_pairs_sha256": sha(out / f"pairs-{count}.npz"),
                        "protocol_sha256": sha(protocol_path),
                        "model_code_sha256": sha(ROOT / "experiments/model.py"),
                    },
                ),
            )
            layers, spec = load_learned_model(model_path)
            checks = {}
            for split, arrays in [("train", train), ("validation", validation)]:
                with torch.no_grad():
                    z = model.encoder(arrays[0]).numpy()
                deployed = embed_batch(layers, arrays[0].numpy())
                centers = np.array(spec["centroids"])
                expected = np.argmin(
                    np.sum((z[:, None].astype(float) - centers) ** 2, axis=2), axis=1
                )
                mismatches = int(
                    np.count_nonzero(expected != predict_states(layers, spec, arrays[0].numpy()))
                )
                error = float(np.max(np.abs(z - deployed)))
                assert error < 1e-4
                checks[split] = {
                    "rows": len(z),
                    "max_embedding_error": error,
                    "state_mismatches": mismatches,
                }
            # Both directions are already present; do not double the symmetric counts.
            counts = np.zeros((20, 20), dtype=np.int64)
            left, right = [predict_states(layers, spec, a.numpy()) for a in train]
            np.add.at(counts, (left, right), 1)
            matrix, matrix_summary = estimate_matrix(counts)
            write_matrix(directory / "substitution.mat", matrix)
            np.save(directory / "matrix-counts.npy", counts)
            summary = {
                "structure_pairs": count,
                "fraction": count / 152,
                "seed": seed,
                "selected_step": best_step,
                "validation_loss": best,
                "train": evaluate(model, train),
                "validation": evaluate(model, validation),
                "export_checks": checks,
                "matrix_summary": matrix_summary,
                "seconds": perf_counter() - began,
                "encoder_sha256": sha(model_path),
                "matrix_sha256": sha(directory / "substitution.mat"),
            }
            save(directory / "summary.json", summary)
            runs.append(summary)
            print(
                "trained",
                count,
                seed,
                "best step",
                best_step,
                "loss",
                round(best, 5),
                "export",
                checks,
                flush=True,
            )
    full = [r for r in runs if r["structure_pairs"] == 152]
    selected = min(full, key=lambda r: (r["validation_loss"], r["seed"]))
    save(
        out / "training-results.json",
        {
            "runs": runs,
            "primary_selected_seed": selected["seed"],
            "primary_selected_structure_pairs": 152,
            "test_used": False,
            "wall_seconds": perf_counter() - started,
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "script_sha256": sha(__file__),
        },
    )
    print("R2 training complete; selected", selected["seed"], flush=True)


if __name__ == "__main__":
    main()
