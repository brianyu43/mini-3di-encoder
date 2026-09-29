"""Render source-backed R1 result tables, PR plots and continuation evidence."""

import csv
import shutil

from experiments.common import ROOT, bounded_cpu, read, save, sha

from .common import OUT


def main():
    out = OUT
    bounded_cpu(out)
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = ROOT / "reports/followup-v2"
    destination.mkdir(parents=True, exist_ok=True)
    results = read(out / "R1-results.json")
    protocol = read(out / "protocol-v2.json")
    rows = results["rows"]
    assert read(out / "R1-independent-audit.json")["passed"]
    for name in [
        "R1-results.json",
        "R1-independent-audit.json",
        "data-freeze.json",
        "test-freeze.json",
        "protocol-v2.json",
        "manifest-v2.json",
        "denominators-v2.json",
        "selection-audit.json",
        "similarity-summary.json",
        "cost-preflight.json",
        "validation-selection.json",
        "encoding.json",
    ]:
        shutil.copyfile(out / name, destination / name)
    flat = []
    for row in rows:
        for setting in row["settings"]:
            for view, metrics in row["metrics"].items():
                flat.append(
                    {
                        "setting": setting,
                        "variant": row["variant"],
                        "gap_open": row["gap"][0],
                        "gap_extend": row["gap"][1],
                        "view": view,
                        **metrics["summary"],
                        "search_seconds": row["search_seconds_median"],
                    }
                )
    with (destination / "R1-results.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(flat)
    variants = protocol["variants"]
    labels = [
        "Official / original matrix",
        "Official / refitted matrix",
        "Learned / refitted matrix",
    ]
    colors = ["#315c91", "#bb7b22", "#28836c"]
    plt.rcParams.update(
        {"font.size": 10, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, setting in zip(axes, ["common", "selected"], strict=True):
        for index, (variant, color, label) in enumerate(zip(variants, colors, labels, strict=True)):
            row = next(r for r in rows if r["variant"] == variant and setting in r["settings"])
            x = np.arange(4) + (index - 1) * 0.18
            points = [row["metrics"][view]["summary"]["MAP"] for view in protocol["views"]]
            ci = [row["metrics"][view]["MAP_bootstrap"]["ci95"] for view in protocol["views"]]
            ax.errorbar(
                x,
                points,
                yerr=[
                    [p - interval[0] for p, interval in zip(points, ci, strict=True)],
                    [interval[1] - p for p, interval in zip(points, ci, strict=True)],
                ],
                fmt="o",
                color=color,
                label=label,
                capsize=3,
            )
        ax.set(
            xticks=range(4),
            xticklabels=["All", "Cross-family", "Same fold", "Other folds"],
            ylim=(0, 1.03),
            ylabel="MAP; 95% fold-cluster interval",
            title="Common gap (10, 1)" if setting == "common" else "Validation-selected gaps",
        )
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8, loc="lower left")
    for ext in ["png", "svg"]:
        fig.savefig(destination / f"R1-quality.{ext}", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, view in zip(axes, ["all", "cross_family"], strict=True):
        for variant, color, label in zip(variants, colors, labels, strict=True):
            m = next(r for r in rows if r["variant"] == variant and "common" in r["settings"])[
                "metrics"
            ][view]
            ax.plot(m["pr_recall"], m["pr_precision_macro_interpolated"], color=color, label=label)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1.02),
            xlabel="Recall",
            ylabel="Macro interpolated precision",
            title="All targets" if view == "all" else "Same-family targets excluded",
        )
    axes[0].legend(fontsize=8)
    for ext in ["png", "svg"]:
        fig.savefig(destination / f"R1-precision-recall.{ext}", bbox_inches="tight")
    plt.close(fig)
    common = {r["variant"]: r for r in rows if "common" in r["settings"]}
    learned = common["learned_refit"]
    difference = (
        learned["metrics"]["all"]["summary"]["MAP"]
        - common["official_refit"]["metrics"]["all"]["summary"]["MAP"]
    )
    gate = {
        "R1_complete": True,
        "R2_MAP_gate_triggered": difference <= -0.02,
        "learned_minus_refit_common_gap_MAP": difference,
        "R2_failure_examples": len(
            [
                r
                for r in read(out / "R1-independent-audit.json")["poor_query_examples"]
                if r["variant"] == "learned_refit" and r["gap"] == [10, 1]
            ]
        ),
        "learned_score_time_fraction": learned["stage_seconds_median"]["sw_score"]
        / learned["search_seconds_median"],
        "R3_status": (
            "No filter-throughput conclusion from exhaustive R1 alone; "
            "preserve controlled-system plan"
        ),
        "script_sha256": sha(__file__),
    }
    save(out / "continuation-gates.json", gate)
    save(destination / "continuation-gates.json", gate)
    print(gate, flush=True)


if __name__ == "__main__":
    main()
