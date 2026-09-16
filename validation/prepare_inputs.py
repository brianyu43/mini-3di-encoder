"""Restore only frozen inputs from an already downloaded SCOPe pilot, without network."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "validation/dataset.json").read_text())
    manifest = {
        row["record_id"]: row
        for line in (args.pilot_root / "manifest.jsonl").read_text().splitlines()
        if (row := json.loads(line))
    }
    for record in config["records"]:
        destination = ROOT / record["path"]
        if destination.exists():
            data = destination.read_bytes()
        else:
            entry = manifest[record["record_id"]]
            data = (args.pilot_root / entry["structure_relpath"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {record['record_id']}")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(data)
        print(record["record_id"], "verified")


if __name__ == "__main__":
    main()
