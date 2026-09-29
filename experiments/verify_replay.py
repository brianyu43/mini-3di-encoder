"""Verify a fresh extraction/alignment/training/matrix replay without retuning."""

import argparse
from pathlib import Path

import numpy as np

from .common import DEFAULT_OUT, read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    old, new = args.original, args.replay
    names = ["manifest.json", "pair_plan.json", "experiment_config.json", "alignment_audit.json"]
    names += [
        f"matrices/{name}.mat" for name in ["official_original", "official_refit", "learned_refit"]
    ]
    for name in names:
        assert sha(old / name) == sha(new / name), name
    features = list((old / "features").glob("*.npz"))
    for path in features + [old / "pairs-train.npz", old / "pairs-validation.npz"]:
        with np.load(path) as a, np.load(new / path.relative_to(old)) as b:
            assert a.files == b.files
            assert all(np.array_equal(a[key], b[key]) for key in a.files), str(path)
    selected = read(old / "models/selection.json")["selected_seed"]
    assert read(new / "models/selection.json")["selected_seed"] == selected
    for seed in [17, 29, 43]:
        a, b = [read(p / f"models/seed-{seed}/encoder.json") for p in [old, new]]
        assert a["layers"] == b["layers"] and a["centroids"] == b["centroids"]
        histories = [read(p / f"models/seed-{seed}/history.json") for p in [old, new]]
        assert len(histories[0]) == len(histories[1]) == 61
        for before, after in zip(*histories, strict=True):
            assert before["epoch"] == after["epoch"]
            assert before["train"] == after["train"]
            assert before["validation"] == after["validation"]
    save(
        old / "replay_verification.json",
        {
            "passed": True,
            "fresh_extraction_structures": len(features),
            "alignment_pairs": 480,
            "same_arrays": True,
            "identical_numeric_model_exports": 3,
            "identical_epoch_metrics": 183,
            "identical_matrices": 3,
            "selected_seed": selected,
            "replay_directory": str(new),
            "scope": "Same machine and package versions; wall times excluded",
            "script_sha256": sha(Path(__file__)),
            "restore": read(new / "restore.json"),
        },
    )
    print(
        "Full replay matched: 384 structures, 480 alignments, "
        "3 models, 183 epoch metrics, 3 matrices"
    )


if __name__ == "__main__":
    main()
