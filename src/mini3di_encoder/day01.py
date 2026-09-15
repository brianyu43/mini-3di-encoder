"""Create a reproducible official reference and a CA table; not an encoder."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from Bio import SeqIO

from .structures import read_ca_rows

EXPECTED_VERSION = "941cd33ff0771cd2e3f144e3293e22a2b87e9fda"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def one_fasta(path: Path) -> tuple[str, str]:
    records = list(SeqIO.parse(path, "fasta"))
    if len(records) != 1:
        raise ValueError(f"Day 01 expects exactly one official chain, got {len(records)}")
    return records[0].id, str(records[0].seq)


def attach_3di_headers(database: Path) -> None:
    """Release 10 shares AA headers; verify keys before sharing that identity."""

    def keys(suffix: str) -> list[int]:
        lines = Path(str(database) + suffix + ".index").read_text().splitlines()
        result = [int(line.split("\t")[0]) for line in lines]
        if not result or len(result) != len(set(result)):
            raise ValueError("Empty or duplicate database keys")
        return sorted(result)

    if keys("") != keys("_ss") or keys("") != keys("_h"):
        raise ValueError("AA, 3Di and header database keys differ")
    for suffix in ["", ".index", ".dbtype"]:
        target = Path(str(database) + "_ss_h" + suffix)
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(Path(str(database) + "_h" + suffix), target)


def run_day01(structure: Path, chain: str, foldseek: Path, out: Path) -> dict:
    structure, foldseek = structure.resolve(), foldseek.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    rows = read_ca_rows(structure, chain)
    out.mkdir(parents=True, exist_ok=False)
    work = out / "_db"
    work.mkdir()
    logs = out / "logs"
    logs.mkdir()
    commands = []
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def command(name: str, arguments: list[str]) -> str:
        argv = [str(foldseek), *arguments]
        start = perf_counter()
        process = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env)
        (logs / f"{name}.stdout.txt").write_text(process.stdout)
        (logs / f"{name}.stderr.txt").write_text(process.stderr)
        commands.append(
            {
                "name": name,
                "argv": argv,
                "cwd": str(Path.cwd()),
                "returncode": process.returncode,
                "wall_seconds": perf_counter() - start,
            }
        )
        write_json(
            out / "commands.json",
            {
                "commands": commands,
                "thread_environment": {
                    k: env[k]
                    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
                },
            },
        )
        process.check_returncode()
        return process.stdout.strip()

    version = command("version", ["version"])
    if version != EXPECTED_VERSION:
        raise ValueError(f"Official reference version mismatch: {version}")
    command(
        "createdb",
        [
            "createdb",
            str(structure),
            str(work / "db"),
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
        ],
    )
    attach_3di_headers(work / "db")
    command("aa_fasta", ["convert2fasta", str(work / "db"), str(out / "official_aa.fasta")])
    command(
        "three_di_fasta", ["convert2fasta", str(work / "db_ss"), str(out / "official_3di.fasta")]
    )
    aa_id, aa = one_fasta(out / "official_aa.fasta")
    di_id, three_di = one_fasta(out / "official_3di.fasta")
    if aa_id != di_id or not aa_id.endswith("_" + chain):
        raise ValueError(f"Official chain identity mismatch: {aa_id}, {di_id}, {chain}")
    if aa != "".join(row["aa"] for row in rows):
        raise ValueError("Official amino acid sequence does not match CA table")
    if len(three_di) != len(rows):
        raise ValueError("Official 3Di length does not match CA table")
    if set(three_di) - set("ACDEFGHIKLMNPQRSTVWYX"):
        raise ValueError("Unexpected official raw alphabet; do not silently normalize")
    with (out / "residues.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0], "official_3di_char"], delimiter="\t")
        writer.writeheader()
        for row, state in zip(rows, three_di, strict=True):
            display = {k: (f"{v:.3f}" if k.startswith("ca_") else v) for k, v in row.items()}
            writer.writerow({**display, "official_3di_char": state})
    units = "bytes" if sys.platform == "darwin" else "KiB"
    environment = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "compute_threads_requested": 1,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["biopython", "numpy", "pytest", "ruff", "setuptools"]
        },
        "foldseek_version": version,
        "foldseek_sha256": sha256(foldseek),
        "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "children_peak_rss_reported": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "rss_units": units,
        "rss_note": "OS high-water values; self/children are separate, not simultaneous total.",
        "scope": "One-chain reference preparation, not an encoder speed benchmark.",
    }
    write_json(out / "environment.json", environment)
    summary = {
        "session": 1,
        "structure": str(structure),
        "structure_sha256": sha256(structure),
        "chain": chain,
        "official_record_id": aa_id,
        "residue_count": len(rows),
        "official_aa_length": len(aa),
        "official_3di_length": len(three_di),
        "aa_exact_match": True,
        "ca_coordinates_finite": True,
        "own_encoder_implemented": False,
        "three_di_validity_mask_implemented": False,
        "state_agreement_with_own_encoder": None,
        "header_policy": "Verified matching AA/3Di/header DB keys, then shared AA headers.",
        "limitation": "CA presence is not a 3Di validity mask. Raw D can be valid or invalid.",
        "output_sha256": {
            name: sha256(out / name)
            for name in ["official_aa.fasta", "official_3di.fasta", "residues.tsv"]
        },
    }
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--foldseek", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_day01(args.structure, args.chain, args.foldseek, args.out),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
