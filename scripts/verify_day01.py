"""Verify preserved sources, raw PDB coordinates, tests and an official rerun."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / "references/source-manifest.json").read_text())
    for record in manifest["files"]:
        if digest(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Source hash mismatch: " + record["path"])
    # Independent fixed-column inspection: do not reuse Bio.PDB or read_ca_rows.
    raw_ca = []
    for line in (ROOT / "data/raw/1UBQ.pdb").read_text().splitlines():
        if line[:6] == "ATOM  " and line[21] == "A" and line[12:16].strip() == "CA":
            raw_ca.append(
                (
                    int(line[22:26]),
                    line[26].strip(),
                    line[17:20],
                    *(f"{float(line[a:b]):.3f}" for a, b in [(30, 38), (38, 46), (46, 54)]),
                )
            )
    with (ROOT / "results/day01/residues.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    table_ca = [
        (
            int(r["pdb_residue_number"]),
            r["insertion_code"],
            r["residue_name"],
            r["ca_x"],
            r["ca_y"],
            r["ca_z"],
        )
        for r in rows
    ]
    if raw_ca != table_ca or len(rows) != 76:
        raise ValueError("Raw PDB and CA table disagree")
    checks = []
    commands = [
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
        ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
        ("format", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("dependencies", [sys.executable, "-m", "pip", "check"]),
        (
            "repeat",
            [
                sys.executable,
                "-m",
                "mini3di_encoder.day01",
                "--structure",
                "data/raw/1UBQ.pdb",
                "--chain",
                "A",
                "--foldseek",
                "tools/foldseek/bin/foldseek",
                "--out",
                str(out / "_repeat"),
            ],
        ),
    ]
    for name, argv in commands:
        process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=120)
        (out / f"{name}.stdout.txt").write_text(process.stdout)
        (out / f"{name}.stderr.txt").write_text(process.stderr)
        checks.append({"name": name, "argv": argv, "returncode": process.returncode})
        if process.returncode:
            raise RuntimeError(f"{name} failed; see {out}")
    first = json.loads((ROOT / "results/day01/summary.json").read_text())
    second = json.loads((out / "_repeat/summary.json").read_text())
    if first["output_sha256"] != second["output_sha256"]:
        raise ValueError("Official rerun is not byte-identical")
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        ROOT / "pyproject.toml",
        ROOT / "requirements-dev.lock.txt",
    ]
    report = {
        "source_hashes_verified": len(manifest["files"]),
        "raw_ca_rows_verified": len(rows),
        "repeated_outputs_byte_identical": True,
        "scope": "Day 01 preparation only; no independent encoder exists yet.",
        "checks": checks,
        "validated_files": {str(p.relative_to(ROOT)): digest(p) for p in paths},
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
