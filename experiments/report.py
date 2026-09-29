"""Export measured result tables, model bundle and standalone scientific figures."""

import csv
import shutil

from .common import DEFAULT_OUT, ROOT, bounded_cpu, read, save, sha


def main():
    out = DEFAULT_OUT
    bounded_cpu(out)
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = ROOT / "reports/sessions-11-15"
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    search = read(out / "search-test/summary.json")
    training = read(out / "models/selection.json")
    for source, destination in [
        ("search-test/summary.json", "search-results.json"),
        ("models/selection.json", "training-results.json"),
        ("feature_summary.json", "feature-summary.json"),
        ("encoding_benchmark.json", "encoding-benchmark.json"),
        ("independent_audit.json", "independent-audit.json"),
        ("replay_verification.json", "replay-verification.json"),
        ("package/verification.json", "package-verification.json"),
        ("cli-verification.json", "cli-verification.json"),
    ]:
        shutil.copyfile(out / source, report / destination)
    keys = [
        "variant",
        "mode",
        "MAP",
        "tie_expected_MAP",
        "top1_rate",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "retain_exact_at_10",
        "aligned_pairs",
        "dp_cells",
        "search_seconds_median",
    ]
    with (report / "search-results.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(search["rows"])
    align_seconds = sum(read(p)["seconds"] for p in (out / "alignments").glob("*.json"))
    resource = {
        "alignment_subprocess_seconds_sum": align_seconds,
        "three_training_runs_seconds_sum": sum(r["seconds"] for r in training["runs"]),
        "experiment_files_logical_bytes": sum(
            p.stat().st_size
            for p in out.rglob("*")
            if p.is_file() and "package" not in p.relative_to(out).parts
        ),
        "package_verification_logical_bytes": sum(
            p.stat().st_size for p in (out / "package").rglob("*") if p.is_file()
        ),
        "research_venv_logical_bytes": sum(
            p.stat().st_size for p in (ROOT / ".venv-research").rglob("*") if p.is_file()
        ),
        "original_archive_bytes": 306064157,
        "scope": (
            "Logical file sizes; separate existing input archive, research environment, "
            "one experiment excluding packaging, and package verification. "
            "Not cumulative RSS or whole-session wall time."
        ),
    }
    save(report / "resources.json", resource)
    bundle = ROOT / "models/paired-vqvae-small"
    bundle.mkdir(parents=True, exist_ok=True)
    selected = training["selected_seed"]
    shutil.copyfile(out / f"models/seed-{selected}/encoder.json", bundle / "encoder.json")
    shutil.copyfile(out / "matrices/learned_refit.mat", bundle / "substitution.mat")
    save(
        bundle / "bundle.json",
        {
            "name": "paired-vqvae-small-v1",
            "encoder_sha256": sha(bundle / "encoder.json"),
            "matrix_sha256": sha(bundle / "substitution.mat"),
            "alphabet_size": 20,
            "gap_open": 14,
            "gap_extend": 2,
            "seed": selected,
            "epoch": 39,
            "selection": "Paired validation loss for model; validation MAP for gaps",
            "warning": (
                "Different alphabet from official 3Di. "
                "Use this matching matrix and reencode queries and targets."
            ),
            "test_freeze_sha256": sha(out / "test-freeze.json"),
        },
    )
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    colors = ["#345995", "#c77b18", "#26866b"]

    def finish(fig, name):
        fig.savefig(figures / f"{name}.png")
        fig.savefig(figures / f"{name}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for run, color in zip(training["runs"], colors, strict=True):
        history = read(out / f"models/seed-{run['seed']}/history.json")
        axes[0].plot(
            [r["epoch"] for r in history],
            [r["validation"]["loss"] for r in history],
            label=f"Seed {run['seed']}; best epoch {run['selected_epoch']}",
            color=color,
        )
        axes[0].scatter([run["selected_epoch"]], [run["validation_loss"]], color=color, s=25)
    axes[0].set(
        xlabel="Epoch (0 = before training)",
        ylabel="Paired validation total loss (log scale)",
        yscale="log",
        title="Three fixed training runs",
    )
    axes[0].legend(fontsize=8)
    chosen = next(r for r in training["runs"] if r["seed"] == selected)
    axes[1].bar(np.arange(20), chosen["train"]["state_counts"], color=colors[2])
    axes[1].set(
        xlabel="New alphabet state ID",
        ylabel="Directed training feature count",
        title="Selected seed 29: all 20 states used",
        xticks=range(0, 20, 2),
    )
    finish(fig, "training")

    exact = [r for r in search["rows"] if r["mode"] == "exhaustive"]
    labels = [
        "Official + original matrix",
        "Official + refitted matrix",
        "Learned + refitted matrix",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    for i, row in enumerate(exact):
        low, high = row["MAP_bootstrap"]["ci95"]
        axes[0].errorbar(
            row["MAP"],
            i,
            xerr=[[row["MAP"] - low], [high - row["MAP"]]],
            fmt="o",
            color=colors[i],
            capsize=5,
        )
        axes[0].text(0.805, i - 0.16, f"MAP {row['MAP']:.4f}", color=colors[i], fontsize=9)
        axes[1].barh(i, row["recall_at_10"], color=colors[i], height=0.5)
        axes[1].text(0.05, i, f"{row['recall_at_10']:.4f}", va="center", color="white")
    axes[0].set(
        xlim=(0.8, 1.015),
        ylim=(-0.6, 2.6),
        yticks=range(3),
        yticklabels=labels,
        xlabel="MAP; exploratory 95% SF bootstrap interval",
        title="Exhaustive search quality",
    )
    axes[1].set(
        xlim=(0, 1.02),
        ylim=(-0.6, 2.6),
        yticks=[],
        xlabel="Recall@10 (six relevant targets/query)",
        title="16 queries / 48 targets / 8 superfamilies",
    )
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    finish(fig, "search-quality")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    modes = ["exhaustive", "single", "double", "double-ungapped"]
    for variant, label, color in zip([r["variant"] for r in exact], labels, colors, strict=True):
        rows = [
            next(r for r in search["rows"] if r["variant"] == variant and r["mode"] == m)
            for m in modes
        ]
        axes[0].plot(
            range(4),
            [r["dp_cells"] / rows[0]["dp_cells"] for r in rows],
            "o-",
            color=color,
            label=label,
        )
        axes[1].plot(range(4), [r["retain_exact_at_10"] for r in rows], "o-", color=color)
        axes[2].plot(range(4), [r["search_seconds_median"] for r in rows], "o-", color=color)
    for ax in axes:
        ax.set_xticks(range(4), ["All", "1 seed", "2 seeds", "+ Ungapped"], rotation=22)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(
        ylabel="DP cells / exhaustive DP cells", ylim=(0, 1.06), title="Less alignment work"
    )
    axes[1].set(
        ylabel="Own exhaustive top-10 retained", ylim=(0, 1.06), title="Some useful candidates lost"
    )
    axes[2].set(
        ylabel="Warm search seconds (median of 3)",
        ylim=(0, 0.8),
        title="Filters do not speed up this small test",
    )
    axes[0].legend(fontsize=7, loc="lower left")
    finish(fig, "filters-and-time")
    save(
        report / "artifact-hashes.json",
        {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted(
                [
                    *report.rglob("*.json"),
                    *figures.glob("*"),
                    *bundle.glob("*.json"),
                    bundle / "substitution.mat",
                ]
            )
            if p.name != "artifact-hashes.json"
        },
    )
    print("Wrote result tables, three figures, and matching model/matrix bundle", flush=True)


if __name__ == "__main__":
    main()
