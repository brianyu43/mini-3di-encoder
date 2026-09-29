"""Train three fixed CPU runs; choose checkpoint/seed using paired validation loss only."""

import argparse
import copy
import resource
from pathlib import Path
from time import perf_counter

from .common import ARCHIVE_ROOT, DEFAULT_OUT, bounded_cpu, read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    bounded_cpu(out)
    import numpy as np
    import torch

    from mini3di_encoder.learned import embed_batch, load_learned_model, predict_states

    from .model import PairedVQVAE, export_model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = read(out / "experiment_config.json")
    freeze = read(out / "data-freeze.json")
    assert sha(out / "experiment_config.json") == freeze["experiment_config_sha256"]
    arrays = {s: np.load(out / f"pairs-{s}.npz") for s in ["train", "validation"]}
    tensors = {s: (torch.from_numpy(a["x"]), torch.from_numpy(a["y"])) for s, a in arrays.items()}
    models = out / "models"
    models.mkdir(exist_ok=False)
    runs = []

    def evaluate(model, x, y):
        model.eval()
        totals = np.zeros(3)
        counts = np.zeros(20, dtype=np.int64)
        with torch.no_grad():
            for begin in range(0, len(x), 4096):
                a, b = x[begin : begin + 4096], y[begin : begin + 4096]
                total, nll, vq, ids = model.objective(a, b)
                totals += len(a) * np.array([total.item(), nll.item(), vq.item()])
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

    for seed in config["seeds"]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = PairedVQVAE()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        history = []
        best, checkpoint, best_epoch = float("inf"), None, None
        started = perf_counter()
        for epoch in range(config["epochs"] + 1):
            if epoch:
                model.train()
                order = rng.permutation(len(tensors["train"][0]))
                batches = [
                    order[i : i + config["batch_size"]]
                    for i in range(0, len(order), config["batch_size"])
                ]
                if len(batches) > 1 and len(batches[-1]) == 1:
                    batches[-2] = np.concatenate([batches[-2], batches.pop()])
                for indices in batches:
                    x, y = [t[indices] for t in tensors["train"]]
                    optimizer.zero_grad(set_to_none=True)
                    loss, _, _, _ = model.objective(x, y)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite loss at seed {seed}, epoch {epoch}")
                    loss.backward()
                    optimizer.step()
            metrics = {s: evaluate(model, *tensors[s]) for s in tensors}
            history.append({"epoch": epoch, **metrics, "seconds": perf_counter() - started})
            if epoch > 0 and metrics["validation"]["loss"] < best:
                best, best_epoch = metrics["validation"]["loss"], epoch
                checkpoint = copy.deepcopy(model.state_dict())
            if epoch % 10 == 0:
                print(
                    seed,
                    epoch,
                    "val",
                    round(metrics["validation"]["loss"], 5),
                    "train states",
                    metrics["train"]["states_used"],
                    flush=True,
                )
        model.load_state_dict(checkpoint)
        model.eval()
        directory = models / f"seed-{seed}"
        directory.mkdir()
        torch.save(checkpoint, directory / "checkpoint.pt")
        save(directory / "history.json", history)
        provenance = {
            "seed": seed,
            "selected_epoch": best_epoch,
            "selection": "minimum validation total loss",
            "geometry_upstream_commit": "941cd33ff0771cd2e3f144e3293e22a2b87e9fda",
            "training_source_commit": "654100b11242e581f9e6d43798b07a778903862e",
            "manifest_sha256": sha(out / "manifest.json"),
            "train_pairs_sha256": sha(out / "pairs-train.npz"),
            "validation_pairs_sha256": sha(out / "pairs-validation.npz"),
            "experiment_config_sha256": sha(out / "experiment_config.json"),
            "model_code_sha256": sha(Path(__file__).with_name("model.py")),
            "upstream_training_sha256": sha(
                ARCHIVE_ROOT / "references/upstream/foldseek-analysis/training/train_vqvae.py"
            ),
        }
        save(directory / "encoder.json", export_model(model, provenance))
        layers, spec = load_learned_model(directory / "encoder.json")
        checks = {}
        for split, (x, _) in tensors.items():
            with torch.no_grad():
                z = model.encoder(x).numpy()
            deployed = embed_batch(layers, x.numpy())
            centers = np.asarray(spec["centroids"])
            reference_states = np.argmin(
                np.sum((z[:, None].astype(float) - centers) ** 2, axis=2), 1
            )
            states = predict_states(layers, spec, x.numpy())
            max_error = float(np.max(np.abs(z - deployed)))
            mismatches = int(np.count_nonzero(states != reference_states))
            if max_error > 1e-4 or mismatches:
                raise ValueError(f"Export mismatch {split}: {max_error}, {mismatches}")
            checks[split] = {
                "rows": len(x),
                "max_embedding_error": max_error,
                "state_mismatches": mismatches,
            }
        summary = {
            "seed": seed,
            "selected_epoch": best_epoch,
            "validation_loss": best,
            "train": evaluate(model, *tensors["train"]),
            "validation": evaluate(model, *tensors["validation"]),
            "seconds": perf_counter() - started,
            "export_checks": checks,
            "encoder_sha256": sha(directory / "encoder.json"),
        }
        save(directory / "summary.json", summary)
        runs.append(summary)
    chosen = min(runs, key=lambda r: (r["validation_loss"], r["seed"]))
    save(
        models / "selection.json",
        {
            "runs": runs,
            "selected_seed": chosen["seed"],
            "test_used": False,
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "script_sha256": sha(Path(__file__)),
        },
    )
    print("selected", chosen["seed"], chosen["selected_epoch"], flush=True)


if __name__ == "__main__":
    main()
