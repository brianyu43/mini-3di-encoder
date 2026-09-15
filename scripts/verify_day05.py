"""Record final tests and an isolated wheel installation with bundled model data."""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from mini3di_encoder.day01 import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--wheelhouse", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    package = out / "_package"
    package.mkdir()
    checks = []

    def run(name, argv, cwd=ROOT):
        argv = [str(x) for x in argv]
        process = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=120)
        (out / f"{name}.stdout.txt").write_text(process.stdout)
        (out / f"{name}.stderr.txt").write_text(process.stderr)
        checks.append(
            {"name": name, "argv": argv, "cwd": str(cwd), "exit_code": process.returncode}
        )
        (out / "commands.json").write_text(json.dumps(checks, indent=2) + "\n")
        process.check_returncode()
        return process.stdout

    for name, tail in [
        ("pytest", ["pytest", "-q"]),
        ("ruff", ["ruff", "check", "."]),
        ("format", ["ruff", "format", "--check", "."]),
        ("dependencies", ["pip", "check"]),
    ]:
        run(name, [sys.executable, "-m", *tail])
    source_manifest = json.loads((ROOT / "references/source-manifest.json").read_text())
    for row in source_manifest["files"]:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Upstream hash changed: " + row["path"])
    dependencies = []
    for row in json.loads((ROOT / "references/wheel-manifest.json").read_text())["wheels"]:
        if row["name"] not in ("numpy", "biopython"):
            continue
        path = args.wheelhouse.resolve() / row["filename"]
        if sha256(path) != row["sha256"]:
            raise ValueError("Wheel hash changed")
        dependencies.append(path)
    if len(dependencies) != 2:
        raise ValueError("Missing offline runtime dependencies")
    run(
        "wheel",
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--no-index",
            "--wheel-dir",
            package,
            ROOT,
        ],
    )
    (wheel,) = package.glob("mini_3di_encoder*.whl")
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.namelist()
    for needed in ["encoder.kerasify", "official.json", "FOLDSEEK-LICENSE.txt"]:
        if not any(name.endswith("/data/" + needed) for name in entries):
            raise ValueError("Model resource absent from wheel: " + needed)
    run("venv", [sys.executable, "-m", "venv", package / "venv"])
    python = package / "venv/bin/python"
    run(
        "install", [python, "-m", "pip", "install", "--no-index", "--no-deps", *dependencies, wheel]
    )
    location = run(
        "installed_location",
        [python, "-I", "-c", "import mini3di_encoder; print(mini3di_encoder.__file__)"],
        cwd=package,
    ).strip()
    if "/site-packages/" not in location:
        raise ValueError("Fresh install imported source checkout")
    installed_trace = out / "installed_trace.json"
    run(
        "installed_trace",
        [
            python,
            "-I",
            "-m",
            "mini3di_encoder.trace",
            "--structure",
            ROOT / "data/raw/1UBQ.pdb",
            "--index",
            "1",
            "--out",
            installed_trace,
        ],
        cwd=package,
    )
    original = ROOT / "results/day02-05/trace_i001.json"
    if sha256(installed_trace) != sha256(original):
        raise ValueError("Installed package trace differs from checked source trace")
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "tests").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "src/mini3di_encoder/data").glob("*")),
        ROOT / "validation/cpp_oracle.cpp",
        ROOT / "configs/day02-05.json",
        ROOT / "pyproject.toml",
    ]
    report = {
        "all_checks_passed": True,
        "checks": checks,
        "installed_location": location,
        "installed_trace_byte_identical": True,
        "wheel_sha256": sha256(wheel),
        "source_files_verified": len(source_manifest["files"]),
        "tested_file_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"checks_passed": len(checks), "installed_trace_byte_identical": True}, indent=2)
    )


if __name__ == "__main__":
    main()
