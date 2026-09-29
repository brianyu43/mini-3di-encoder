"""Compare both full encoders on the same 64 structures, separate from search."""

import argparse
import resource
import statistics
import subprocess
from pathlib import Path
from time import perf_counter

from .common import DEFAULT_OUT, bounded_cpu, read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    bounded_cpu(out)
    import numpy as np

    from mini3di_encoder.atoms import read_backbone
    from mini3di_encoder.encode import encode_chain
    from mini3di_encoder.learned import load_learned_model, predict_states

    rows = [r for r in read(out / "manifest.json") if r["split"] == "test"]
    model = out / f"models/seed-{read(out / 'models/selection.json')['selected_seed']}/encoder.json"
    layers, spec = load_learned_model(model)
    hardware = {}
    for key in ["machdep.cpu.brand_string", "hw.memsize", "hw.logicalcpu"]:
        result = subprocess.run(["sysctl", "-n", key], text=True, capture_output=True)
        hardware[key] = (
            result.stdout.strip() if result.returncode == 0 else "unavailable in sandbox"
        )
    began = perf_counter()
    chains = {r["record_id"]: read_backbone(out / r["path"], r["chain"]) for r in rows}
    parse_seconds = perf_counter() - began
    # Warm each API once. No output is used to select models or settings.
    first = rows[0]["record_id"]
    for path in [None, model]:
        encode_chain(chains[first], first, encoder_model=path)
    repeats, state_counts = [], {}
    for repeat in range(3):
        for variant in ["official", "learned"] if repeat % 2 == 0 else ["learned", "official"]:
            seconds = 0.0
            counts = np.zeros(20, dtype=int)
            valid_count = 0
            for row in rows:
                sid = row["record_id"]
                began = perf_counter()
                result = encode_chain(
                    chains[sid], sid, encoder_model=model if variant == "learned" else None
                )
                seconds += perf_counter() - began
                with np.load(out / f"features/{sid}.npz") as arrays:
                    mask = arrays["mask"]
                    expected = (
                        arrays["states"]
                        if variant == "official"
                        else predict_states(layers, spec, arrays["x"])
                    )
                    actual = np.array([r["state"] for r in result["rows"]])
                    assert np.array_equal(result["valid_seed_mask"], mask)
                    assert np.array_equal(actual[mask], expected[mask])
                    counts += np.bincount(actual[mask], minlength=20)
                    valid_count += int(mask.sum())
            repeats.append({"repeat": repeat, "variant": variant, "seconds": seconds})
            if variant in state_counts:
                assert counts.tolist() == state_counts[variant]
            state_counts[variant] = counts.tolist()
            print("encoder benchmark", repeat, variant, seconds, flush=True)
    save(
        out / "encoding_benchmark.json",
        {
            "structures": len(rows),
            "residues": sum(r["length"] for r in rows),
            "valid_search_positions": valid_count,
            "scope": (
                "Full encode_chain including geometry, per-residue tracing and model load; "
                "excludes parsing and JSON serialization"
            ),
            "cpu_threads": 1,
            "repeats": repeats,
            "parse_once_seconds": parse_seconds,
            "seconds_median": {
                v: statistics.median(r["seconds"] for r in repeats if r["variant"] == v)
                for v in state_counts
            },
            "state_counts": state_counts,
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "state_mismatches": 0,
            "script_sha256": sha(Path(__file__)),
            "hardware": hardware,
            "model_sha256": sha(model),
        },
    )


if __name__ == "__main__":
    main()
