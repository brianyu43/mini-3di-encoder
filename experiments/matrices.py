"""Estimate half-bit log-odds from training-only aligned state pairs."""

import argparse
import shutil
from pathlib import Path

import numpy as np

from mini3di_encoder.learned import load_learned_model, predict_states

from .common import ALPHABET, ARCHIVE_ROOT, DEFAULT_OUT, read, save, sha


def estimate_matrix(counts, pseudocount=0.5, scale=2.0):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (20, 20) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Counts must be finite nonnegative 20x20")
    if not np.array_equal(counts, counts.T) or counts.sum() <= 0 or pseudocount <= 0:
        raise ValueError("Nonempty symmetric counts and positive smoothing required")
    joint = (counts + pseudocount) / (counts.sum() + counts.size * pseudocount)
    background = joint.sum(axis=1)
    log_odds = np.log2(joint / np.outer(background, background))
    scores = np.zeros((21, 21), dtype=np.int64)
    scores[:20, :20] = np.rint(scale * log_odds).astype(np.int64)
    return scores, {
        "joint": joint.tolist(),
        "background": background.tolist(),
        "mutual_information_bits_smoothed": float(np.sum(joint * log_odds)),
        "random_expected_score": float(np.sum(np.outer(background, background) * scores[:20, :20])),
        "directed_counts": int(counts.sum()),
        "pseudocount_per_cell": pseudocount,
        "scale": scale,
        "rounding": "numpy.rint ties-to-even",
        "X_score": 0,
    }


def write_matrix(path, matrix):
    letters = ALPHABET + "X"
    lines = ["  " + " ".join(letters)]
    lines.extend(
        letter + " " + " ".join(map(str, row)) for letter, row in zip(letters, matrix, strict=True)
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--official-matrix",
        type=Path,
        default=ARCHIVE_ROOT / "references/upstream/foldseek/data/mat3di.out",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    config = read(out / "experiment_config.json")
    selected = read(out / "models/selection.json")["selected_seed"]
    encoder = out / f"models/seed-{selected}/encoder.json"
    learned, spec = load_learned_model(encoder)
    rows = [r for r in read(out / "manifest.json") if r["split"] == "train"]
    state_maps = {"official_refit": {}, "learned_refit": {}}
    masks = {}
    for row in rows:
        sid = row["record_id"]
        with np.load(out / f"features/{sid}.npz") as arrays:
            state_maps["official_refit"][sid] = arrays["states"]
            state_maps["learned_refit"][sid] = predict_states(learned, spec, arrays["x"])
            masks[sid] = arrays["mask"]
    directory = out / "matrices"
    directory.mkdir(exist_ok=False)
    source = args.official_matrix
    shutil.copyfile(source, directory / "official_original.mat")
    summaries = {}
    audit = read(out / "alignment_audit.json")
    for variant, states in state_maps.items():
        counts = np.zeros((20, 20), dtype=np.int64)
        for pair in audit:
            if pair["split"] != "train" or not pair["accepted"]:
                continue
            a, b = pair["a"], pair["b"]
            ij = np.array(pair["indices"])
            assert masks[a][ij[:, 0]].all() and masks[b][ij[:, 1]].all()
            sa, sb = states[a][ij[:, 0]], states[b][ij[:, 1]]
            np.add.at(counts, (sa, sb), 1)
            np.add.at(counts, (sb, sa), 1)
        matrix, summary = estimate_matrix(
            counts, config["matrix_pseudocount"], config["matrix_scale"]
        )
        write_matrix(directory / f"{variant}.mat", matrix)
        np.save(directory / f"{variant}-counts.npy", counts)
        summary.update(
            {
                "matrix_sha256": sha(directory / f"{variant}.mat"),
                "counts_sha256": sha(directory / f"{variant}-counts.npy"),
                "states_with_aligned_counts": int(np.count_nonzero(counts.sum(axis=1))),
                "minimum": int(matrix.min()),
                "maximum": int(matrix.max()),
            }
        )
        summaries[variant] = summary
    save(
        directory / "provenance.json",
        {
            "estimates": summaries,
            "train_only": True,
            "official_matrix_sha256": sha(source),
            "encoder_sha256": sha(encoder),
            "train_pairs_sha256": sha(out / "pairs-train.npz"),
            "alignment_audit_sha256": sha(out / "alignment_audit.json"),
            "script_sha256": sha(Path(__file__)),
            "difference_from_upstream": (
                "Smoothing 0.5/cell, invalids excluded rather than merged; X neutral"
            ),
        },
    )
    print(
        {
            k: {
                n: v[n]
                for n in ["directed_counts", "states_with_aligned_counts", "minimum", "maximum"]
            }
            for k, v in summaries.items()
        }
    )


if __name__ == "__main__":
    main()
