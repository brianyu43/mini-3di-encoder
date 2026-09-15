"""Produce days 02–05 artifacts and compare selected residues to native Foldseek."""

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from mini3di_encoder.atoms import ATOM_NAMES, read_backbone
from mini3di_encoder.day01 import EXPECTED_VERSION, one_fasta, sha256
from mini3di_encoder.network import forward, official_model
from mini3di_encoder.trace import trace_residue

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build, native_output, logs = [out / name for name in ["_build", "_official", "logs"]]
    for path in [build, native_output, logs]:
        path.mkdir()
    commands = []

    def run(name, argv):
        argv = [str(item) for item in argv]
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=120)
        (logs / (name + ".stdout.txt")).write_text(result.stdout)
        (logs / (name + ".stderr.txt")).write_text(result.stderr)
        commands.append({"name": name, "argv": argv, "returncode": result.returncode})
        save(out / "commands.json", commands)
        result.check_returncode()
        return result.stdout

    config = json.loads((ROOT / "configs/day02-05.json").read_text())
    save(out / "config.json", config)
    manifest = json.loads((ROOT / "references/source-manifest.json").read_text())
    for row in manifest["files"]:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Original source changed: " + row["path"])
    upstream = ROOT / "references/upstream/foldseek"
    binary = ROOT / "tools/foldseek/bin/foldseek"
    if run("foldseek_version", [binary, "version"]).strip() != EXPECTED_VERSION:
        raise ValueError("Foldseek version mismatch")
    compiler_version = run("compiler_version", ["/usr/bin/clang++", "--version"])
    model, spec = official_model()
    if sha256(upstream / "data/encoder_weights_3di.kerasify") != spec["weights_sha256"]:
        raise ValueError("Bundled and upstream weights disagree")
    header = (upstream / "lib/3di/structureto3di.h").read_text()
    block = header.split("const double centroids", 1)[1].split("};", 1)[0]
    native_centroids = [
        [float(a), float(b)] for a, b in re.findall(r"\{\s*([-\d.]+),\s*([-\d.]+)\s*\}", block)
    ]
    if native_centroids != spec["centroids"]:
        raise ValueError("Bundled and upstream centers disagree")
    blob = (upstream / "data/encoder_weights_3di.kerasify").read_bytes()
    generated = build / "encoder_weights_3di.kerasify.h"
    generated.write_text(
        "const unsigned char encoder_weights_3di_kerasify[] = {"
        + ",".join(str(v) for v in blob)
        + "};\n"
        + f"const unsigned int encoder_weights_3di_kerasify_len = {len(blob)};\n"
    )
    executable = build / "oracle"
    run(
        "compile_oracle",
        [
            "/usr/bin/clang++",
            "-std=c++17",
            "-O0",
            "-ffp-contract=off",
            "-I",
            upstream / "lib",
            "-I",
            upstream / "lib/3di",
            "-I",
            upstream / "lib/mmseqs/lib/simde",
            "-I",
            build,
            ROOT / "validation/cpp_oracle.cpp",
            upstream / "lib/3di/structureto3di.cpp",
            upstream / "lib/kerasify/keras_model.cpp",
            "-o",
            executable,
        ],
    )
    structure = ROOT / config["structure"]
    chain = read_backbone(structure, config["chain"])
    aa_id, aa = one_fasta(ROOT / "results/day01/official_aa.fasta")
    _, original_three_di = one_fasta(ROOT / "results/day01/official_3di.fasta")
    if chain.aa != aa or aa_id != "1UBQ_A":
        raise ValueError("Atom reader disagrees with official sequence")
    save(out / "day02_atoms.json", [chain.row(i) for i in range(len(chain.residues))])
    with (out / "day02_atoms.tsv").open("w", newline="") as handle:
        fields = ["index_0", "chain", "residue_number", "insertion_code", "aa", "cb_status"]
        fields += [f"{atom}_{field}" for atom in ATOM_NAMES for field in ["present", "x", "y", "z"]]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for i in range(len(chain.residues)):
            row = chain.row(i)
            flat = {
                "index_0": i,
                "chain": chain.chain_id,
                "residue_number": row["pdb_residue_number"],
                "insertion_code": row["insertion_code"],
                "aa": row["aa"],
                "cb_status": row["cb_status"],
            }
            for atom in ATOM_NAMES:
                record = row["atoms"][atom]
                flat[f"{atom}_present"] = record["present"]
                for k, axis in enumerate("xyz"):
                    flat[f"{atom}_{axis}"] = (
                        "" if record["xyz"] is None else f"{record['xyz'][k]:.3f}"
                    )
            writer.writerow(flat)
    coordinate_input = out / "native_coordinate_input.txt"
    coordinate_input.write_text(
        str(len(chain.residues))
        + "\n"
        + "\n".join(" ".join(repr(float(v)) for v in row.ravel()) for row in chain.xyz)
        + "\n"
    )
    selected = config["selected_indices_0"]
    endpoints = config["endpoint_indices_0"]
    native = json.loads(
        run(
            "native_coordinates",
            [
                executable,
                "coordinates",
                upstream / "data/encoder_weights_3di.kerasify",
                coordinate_input,
                *selected,
                *endpoints,
            ],
        )
    )
    save(out / "native_trace.json", native)
    native_by_index = {row["index"]: row for row in native}
    run(
        "official_descriptor",
        [
            binary,
            "structureto3didescriptor",
            structure,
            native_output / "descriptor",
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
            "--mask-bfactor-threshold",
            "0",
        ],
    )
    descriptor_blob = (native_output / "descriptor").read_bytes()
    descriptor = descriptor_blob.rstrip(b"\0\n").decode()
    (out / "official_descriptor.tsv").write_text(descriptor + "\n")
    descriptor_header, descriptor_aa, descriptor_di, numbers = descriptor.split("\t")
    if descriptor_aa != aa or descriptor_di != original_three_di:
        raise ValueError("Official descriptor and official day-01 output disagree")
    official_features = np.array([float(x) for x in numbers.split(",")]).reshape(-1, 10)
    comparisons = []
    for i in selected:
        trace = trace_residue(chain, i)
        save(out / f"trace_i{i:03d}.json", trace)
        if not trace["valid"]:
            raise ValueError(f"Preselected inspection position {i} is invalid")
        reference = native_by_index[i]
        values = np.array([f["value"] for f in trace["features"]])
        features_error = float(np.max(np.abs(values - reference["features"])))
        embedding_error = float(
            np.max(np.abs(np.array(trace["network"]["embedding"]) - reference["embedding"]))
        )
        geometry_error = max(
            float(np.max(np.abs(np.array(trace[k]) - reference[k])))
            for k in ["virtual_center", "partner_virtual_center", "cb_used"]
        )
        row = {
            "index_0": i,
            "pdb_residue_number": chain.residues[i][0],
            "partner_index_0": trace["partner"]["sequence_index_0"],
            "partner_pdb_residue_number": trace["partner"]["pdb_residue_number"],
            "native_geometry_max_abs_error": geometry_error,
            "native_feature_max_abs_error": features_error,
            "native_embedding_max_abs_error": embedding_error,
            "partner_equal": trace["partner"]["sequence_index_0"] == reference["partner"],
            "native_state_equal": trace["state"] == reference["state"],
            "official_binary_letter_equal": trace["letter"] == original_three_di[i],
            "letter": trace["letter"],
            "state": trace["state"],
            "descriptor_print_equal": bool(
                np.array_equal([float(f"{x:.3E}") for x in values], official_features[i])
            ),
        }
        comparisons.append(row)
        save(out / "comparisons.json", comparisons)
        if (
            geometry_error > config["native_geometry_atol"]
            or features_error > config["native_feature_atol"]
            or embedding_error > config["native_embedding_atol"]
            or not all(
                row[k]
                for k in [
                    "partner_equal",
                    "native_state_equal",
                    "official_binary_letter_equal",
                    "descriptor_print_equal",
                ]
            )
        ):
            raise ValueError(f"Official mismatch at residue {i}: {row}")
    endpoint_results = []
    for i in endpoints:
        trace = trace_residue(chain, i)
        save(out / f"trace_i{i:03d}.json", trace)
        if (
            trace["valid"]
            or trace["state"] != native_by_index[i]["state"]
            or trace["letter"] != original_three_di[i]
        ):
            raise ValueError("Endpoint handling mismatch")
        endpoint_results.append(
            {
                "index_0": i,
                "valid": False,
                "state": trace["state"],
                "letter": trace["letter"],
                "reason": trace["invalid_reason"],
            }
        )
    rng = np.random.default_rng(config["network_probe_seed"])
    probes = np.vstack(
        [
            np.zeros(10),
            np.ones(10),
            np.eye(10),
            rng.normal(size=(config["network_random_probes"], 10)),
        ]
    ).astype(np.float32)
    probe_file = out / "network_probe_input.txt"
    probe_file.write_text(
        str(len(probes))
        + "\n"
        + "\n".join(" ".join(repr(float(x)) for x in row) for row in probes)
        + "\n"
    )
    native_embeddings = json.loads(
        run(
            "native_network",
            [executable, "features", upstream / "data/encoder_weights_3di.kerasify", probe_file],
        )
    )
    own_embeddings = [forward(model, row)["embedding"] for row in probes]
    probe_error = float(np.max(np.abs(np.array(native_embeddings) - own_embeddings)))
    save(
        out / "network_probes.json",
        {
            "inputs": probes.tolist(),
            "native_outputs": native_embeddings,
            "own_outputs": own_embeddings,
            "max_abs_error": probe_error,
        },
    )
    if probe_error > config["native_embedding_atol"]:
        raise ValueError("Native neural-network probe mismatch")
    report = {
        "completed_sessions": [2, 3, 4, 5],
        "g1_implementation_gate": "passed",
        "input_sha256": sha256(structure),
        "residue_count": len(chain.residues),
        "observed_atom_counts": dict(
            zip(ATOM_NAMES, chain.present.sum(axis=0).tolist(), strict=True)
        ),
        "selected_valid_residues": len(selected),
        "selected_indices_0": selected,
        "endpoints": endpoint_results,
        "comparisons": comparisons,
        "network_probe_count": len(probes),
        "network_probe_max_abs_error": probe_error,
        "model_layers": [
            {"shape": list(x.weights.shape), "activation": x.activation} for x in model
        ],
        "model_parameter_count": sum(x.weights.size + x.bias.size for x in model),
        "model_weights_sha256": spec["weights_sha256"],
        "native_oracle_sha256": sha256(executable),
        "compiler": compiler_version,
        "source_files_verified": len(manifest["files"]),
        "platform": platform.platform(),
        "python": sys.version,
        "scope": (
            "Four preselected residues in one development chain; "
            "no whole-chain validation, retraining or search benchmark."
        ),
        "descriptor_export": "Official DB bytes with only trailing null/newline removed for TSV.",
    }
    save(out / "summary.json", report)
    print(
        json.dumps(
            {
                "gate": report["g1_implementation_gate"],
                "comparisons": comparisons,
                "network_probe_max_abs_error": probe_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
