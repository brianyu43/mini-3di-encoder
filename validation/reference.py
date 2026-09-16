"""Validation-only adapters to pinned, independently executed native tools."""

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.encode import encode_chain

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
VERSION = "941cd33ff0771cd2e3f144e3293e22a2b87e9fda"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(argv, directory, name):
    started = perf_counter()
    result = subprocess.run([str(a) for a in argv], capture_output=True, text=True)
    save(
        directory / f"{name}.command.json",
        {
            "argv": [str(a) for a in argv],
            "returncode": result.returncode,
            "wall_seconds": perf_counter() - started,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    result.check_returncode()
    return result.stdout


def read_db(path):
    blob = path.read_bytes()
    result = {}
    for line in Path(str(path) + ".index").read_text().splitlines():
        key, start, length = map(int, line.split())
        result[key] = blob[start : start + length].rstrip(b"\x00\n").decode()
    return result


def native_trace(chain, oracle, weights, directory):
    coords = directory / "coordinates.txt"
    coords.write_text(
        str(len(chain.residues))
        + "\n"
        + "\n".join(" ".join(format(float(x), ".17g") for x in row.ravel()) for row in chain.xyz)
        + "\n"
    )
    return json.loads(run([oracle, "coordinates", weights, coords], directory, "native"))


def compare_structure(path, chain_id, record_id, foldseek, oracle, weights, directory):
    directory.mkdir(parents=True, exist_ok=False)
    chain = read_backbone(path, chain_id)
    started = perf_counter()
    own = encode_chain(chain, record_id)
    encode_seconds = perf_counter() - started
    save(directory / "encoded.json", own)
    native = native_trace(chain, oracle, weights, directory)
    save(directory / "native.json", native)
    db = directory / "official"
    run(
        [
            foldseek,
            "createdb",
            path,
            db,
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
            "--db-extraction-mode",
            "0",
            "--mask-bfactor-threshold",
            "0",
            "--coord-store-mode",
            "1",
            "-v",
            "0",
        ],
        directory,
        "official",
    )
    aa, states, headers = read_db(db), read_db(Path(str(db) + "_ss")), read_db(Path(str(db) + "_h"))
    assert aa.keys() == states.keys() == headers.keys()
    assert len(aa) == 1, "Validation input must have one chain/model; do not silently pick a record"
    key = next(iter(aa))
    assert aa[key] == chain.aa, "Official/parser residue correspondence differs"
    assert len(native) == len(states[key]) == len(own["aa"])
    (directory / "official_3di.fasta").write_text(f">{headers[key]}\n{states[key]}\n")
    mismatches, rows = [], []
    max_feature_error = max_embedding_error = 0.0
    for row, ref, letter in zip(own["rows"], native, states[key], strict=True):
        i = row["index"]
        issues = []
        native_letter = ALPHABET[ref["state"]]
        if row["letter"] != letter:
            issues.append("official_binary_state")
        if native_letter != letter:
            issues.append("native_vs_binary")
        if row["valid"] != ref["valid"]:
            issues.append("feature_mask")
        if row["valid"] and ref["valid"]:
            if row["partner"] != ref["partner"]:
                issues.append("partner")
            fe = float(np.max(np.abs(np.array(row["features"]) - ref["features"])))
            ze = float(np.max(np.abs(np.array(row["embedding"]) - ref["embedding"])))
            max_feature_error, max_embedding_error = (
                max(max_feature_error, fe),
                max(max_embedding_error, ze),
            )
            if fe > 1e-10:
                issues.append("features")
            if ze > 1e-6:
                issues.append("embedding")
            if row["state"] != ref["state"]:
                issues.append("native_state")
        detail = {
            "index": i,
            "pdb_number": row["pdb_number"],
            "insertion_code": row["insertion_code"],
            "own_valid": row["valid"],
            "native_valid": ref["valid"],
            "own_letter": row["letter"],
            "official_letter": letter,
            "partner": row["partner"],
            "native_partner": ref["partner"],
            "issues": ";".join(issues),
        }
        rows.append(detail)
        if issues:
            mismatches.append(detail)
    for name, data in [("positions.tsv", rows), ("mismatches.tsv", mismatches)]:
        with (directory / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(data)
    common = [i for i, row in enumerate(own["rows"]) if row["valid"] and native[i]["valid"]]
    summary = {
        "record_id": record_id,
        "input_sha256": sha(path),
        "length": len(rows),
        "all_matches": sum(row["own_letter"] == row["official_letter"] for row in rows),
        "common_valid_count": len(common),
        "common_valid_matches": sum(own["raw_three_di"][i] == states[key][i] for i in common),
        "mask_mismatches": sum(row["own_valid"] != row["native_valid"] for row in rows),
        "mismatch_rows": len(mismatches),
        "max_feature_error": max_feature_error,
        "max_embedding_error": max_embedding_error,
        "encode_seconds": encode_seconds,
        "valid_seed_count": sum(own["valid_seed_mask"]),
    }
    save(directory / "summary.json", summary)
    return summary
