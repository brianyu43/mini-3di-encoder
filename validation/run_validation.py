"""Run frozen development structures against Foldseek and rigid transformations."""

import argparse
import json
import platform
import resource
from dataclasses import replace
from pathlib import Path

import numpy as np
from reference import VERSION, compare_structure, run, save, sha

from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.encode import encode_chain, search_record

ROOT = Path(__file__).resolve().parents[1]


def compile_oracle(reference_root, out):
    upstream = reference_root / "references/upstream/foldseek"
    manifest = json.loads((reference_root / "references/source-manifest.json").read_text())
    verified = {}
    for entry in manifest["files"]:
        path = reference_root / entry["path"]
        if path.exists() and (
            "foldseek/lib/" in entry["path"] or "/foldseek/bin/" in entry["path"]
        ):
            assert sha(path) == entry["sha256"], path
            verified[entry["path"]] = sha(path)
    save(out / "reference_hashes.json", verified)
    weights = ROOT / "src/mini3di_encoder/data/encoder.kerasify"
    data = weights.read_bytes()
    (out / "encoder_weights_3di.kerasify.h").write_text(
        "unsigned char encoder_weights_3di_kerasify[] = {" + ",".join(map(str, data)) + "};\n"
        f"unsigned int encoder_weights_3di_kerasify_len = {len(data)};\n"
    )
    oracle = out / "oracle"
    run(
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
            out,
            ROOT / "validation/native_oracle.cpp",
            upstream / "lib/3di/structureto3di.cpp",
            upstream / "lib/kerasify/keras_model.cpp",
            "-o",
            oracle,
        ],
        out,
        "compile",
    )
    return oracle, weights


def change_rows(before, after):
    differences = []
    for a, b in zip(before["rows"], after["rows"], strict=True):
        if (a["partner"], a["state"], a["seed_valid"]) != (
            b["partner"],
            b["state"],
            b["seed_valid"],
        ):
            differences.append(
                {
                    "index": a["index"],
                    "partner_before": a["partner"],
                    "partner_after": b["partner"],
                    "state_before": a["state"],
                    "state_after": b["state"],
                    "seed_before": a["seed_valid"],
                    "seed_after": b["seed_valid"],
                    "partner_margin_before": a["partner_margin"],
                    "state_margin_before": a["state_margin"],
                    "embedding_before": a["embedding"],
                    "embedding_after": b["embedding"],
                }
            )
    return differences


def write_transformed_pdb(original, output, rotation, translation):
    lines = []
    for line in original.read_text().splitlines():
        if line.startswith("ATOM  "):
            xyz = np.array([float(line[a:b]) for a, b in [(30, 38), (38, 46), (46, 54)]])
            xyz = xyz @ rotation.T + translation
            line = line[:30] + "".join(f"{v:8.3f}" for v in xyz) + line[54:]
        lines.append(line)
    output.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "validation/dataset.json").read_text())
    save(out / "config.json", config)
    oracle, weights = compile_oracle(args.reference_root, out)
    foldseek = args.reference_root / "tools/foldseek/bin/foldseek"
    assert run([foldseek, "version"], out, "version").strip() == VERSION
    rng = np.random.default_rng(config["rotation_seed"])
    transforms = [(np.eye(3), np.array([11.25, -7.5, 3.125]))]
    for _ in range(config["rigid_transform_count"] - 1):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        transforms.append((rotation, rng.uniform(-20, 20, 3)))
    summaries, rotations, rounded, export = [], [], [], {}
    for record in config["records"]:
        path = ROOT / record["path"]
        assert sha(path) == record["sha256"]
        record_id = record["record_id"]
        chain = read_backbone(path, record["chain"])
        assert len(chain.residues) == record["length"]
        summary = compare_structure(
            path, record["chain"], record_id, foldseek, oracle, weights, out / record_id
        )
        summaries.append(summary)
        own = json.loads((out / record_id / "encoded.json").read_text())
        export[record_id] = search_record(own)
        for k, (rotation, translation) in enumerate(transforms):
            changed = replace(chain, xyz=chain.xyz @ rotation.T + translation)
            encoded = encode_chain(changed, record_id)
            diffs = change_rows(own, encoded)
            feature_error = max(
                (
                    max(abs(a - b) for a, b in zip(x["features"], y["features"], strict=True))
                    for x, y in zip(own["rows"], encoded["rows"], strict=True)
                    if x["valid"] and y["valid"]
                ),
                default=0.0,
            )
            rotations.append(
                {
                    "record_id": record_id,
                    "transform": k,
                    "differences": diffs,
                    "max_feature_error": feature_error,
                    "rotation": rotation.tolist(),
                    "translation": translation.tolist(),
                }
            )
            transformed_path = out / record_id / f"rounded-{k}.pdb"
            write_transformed_pdb(path, transformed_path, rotation, translation)
            quantized = encode_chain(read_backbone(transformed_path, record["chain"]), record_id)
            rounded_diffs = change_rows(own, quantized)
            # Quantization and algorithm disagreements are separate: re-check rounded inputs
            # against the native sources AND the official binary if any assignment changes.
            reference = None
            if rounded_diffs:
                reference = compare_structure(
                    transformed_path,
                    record["chain"],
                    record_id,
                    foldseek,
                    oracle,
                    weights,
                    out / record_id / f"rounded-{k}-reference",
                )
            rounded.append(
                {
                    "record_id": record_id,
                    "transform": k,
                    "differences": rounded_diffs,
                    "rounded_input_reference": reference,
                }
            )
        print(json.dumps(summary), flush=True)
    for role in ["query", "target"]:
        with (out / f"{role}.jsonl").open("w") as stream:
            for row in config["records"]:
                if row["role"] == role:
                    stream.write(json.dumps(export[row["record_id"]], allow_nan=False) + "\n")
    save(
        out / "export_provenance.json",
        {
            "dataset_sha256": sha(ROOT / "validation/dataset.json"),
            "weights_sha256": sha(weights),
            "upstream_commit": VERSION,
            "inputs": [
                {
                    "record_id": row["record_id"],
                    "input_sha256": row["sha256"],
                    "encoded_sha256": sha(out / row["record_id"] / "encoded.json"),
                }
                for row in config["records"]
            ],
            "query_sha256": sha(out / "query.jsonl"),
            "target_sha256": sha(out / "target.jsonl"),
        },
    )
    save(out / "rotations.json", rotations)
    save(out / "rounded_coordinates.json", rounded)
    save(out / "structures.json", summaries)
    summary = {
        "structure_count": len(summaries),
        "residues": sum(r["length"] for r in summaries),
        "all_matches": sum(r["all_matches"] for r in summaries),
        "common_valid": sum(r["common_valid_count"] for r in summaries),
        "common_valid_matches": sum(r["common_valid_matches"] for r in summaries),
        "mismatch_rows": sum(r["mismatch_rows"] for r in summaries),
        "rigid_transform_cases": len(rotations),
        "rigid_transform_changed_rows": sum(len(r["differences"]) for r in rotations),
        "rounded_transform_changed_rows": sum(len(r["differences"]) for r in rounded),
        "rounded_reference_mismatches": sum(
            r["rounded_input_reference"]["mismatch_rows"]
            for r in rounded
            if r["rounded_input_reference"]
        ),
        "encode_seconds_total": sum(r["encode_seconds"] for r in summaries),
        "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for folder in ["src", "validation"]
            for p in (ROOT / folder).rglob("*")
            if p.suffix in [".py", ".cpp", ".json"]
        },
    }
    save(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
