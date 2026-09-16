"""Encode complete PDB chains with separate raw states and safe search masks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .atoms import Chain, read_backbone
from .geometry import prepare_geometry
from .network import official_model
from .trace import trace_prepared


def encode_chain(chain: Chain, record_id: str) -> dict:
    if not record_id or any(c.isspace() or ord(c) < 32 for c in record_id):
        raise ValueError("record_id must be nonempty without whitespace/control characters")
    if not chain.residues:
        raise ValueError("Empty chains cannot be encoded")
    geometry = prepare_geometry(chain)
    model, spec = official_model()
    links = chain.peptide_links()
    rows = []
    for i in range(len(chain.residues)):
        trace = trace_prepared(chain, i, geometry, model, spec)
        candidates = trace.get("candidates", [])
        j = trace["partner"]["sequence_index_0"] if trace["partner"] else None
        seed_valid = bool(trace["valid"] and all(links[k] for k in [i - 1, i, j - 1, j]))
        rows.append(
            {
                "index": i,
                "pdb_number": chain.residues[i][0],
                "insertion_code": chain.residues[i][1],
                "aa": chain.aa[i],
                "segment": chain.segment(i),
                "partner": j,
                "partner_margin": (
                    candidates[1]["distance"] - candidates[0]["distance"]
                    if len(candidates) > 1
                    else None
                ),
                "valid": trace["valid"],
                "invalid_reason": trace["invalid_reason"],
                "seed_valid": seed_valid,
                "seed_invalid_reason": (
                    None if seed_valid else trace["invalid_reason"] or "chain_discontinuity"
                ),
                "state": trace["state"],
                "letter": trace["letter"],
                "features": [f["value"] for f in trace["features"]] if trace["valid"] else None,
                "embedding": trace["network"]["embedding"] if trace["valid"] else None,
                "state_margin": trace.get("nearest_margin"),
            }
        )
    valid = [row["valid"] for row in rows]
    seed_mask = [row["seed_valid"] for row in rows]
    raw = "".join(row["letter"] for row in rows)
    return {
        "record_id": record_id,
        "chain": chain.chain_id,
        "model": chain.model_id,
        "selected_altloc": chain.altloc,
        "aa": chain.aa,
        "raw_three_di": raw,
        "three_di": "".join(c if ok else "X" for c, ok in zip(raw, seed_mask, strict=True)),
        "residue_valid_mask": geometry.valid.tolist(),
        "feature_valid_mask": valid,
        "valid_seed_mask": seed_mask,
        "peptide_links": list(links),
        "seed_policy": "feature-valid and four C-N links within (0,2.0] A in same TER segments",
        "rows": rows,
        "synthetic": False,
        "weights_sha256": spec["weights_sha256"],
        "upstream_commit": spec["upstream_commit"],
    }


def search_record(encoded: dict) -> dict:
    """Exact mini-3di-search schema; provenance belongs in the companion JSON."""
    keys = ("record_id", "aa", "three_di", "valid_seed_mask", "synthetic")
    record = {key: encoded[key] for key in keys}
    n = len(record["aa"])
    if not n or len(record["three_di"]) != n or len(record["valid_seed_mask"]) != n:
        raise ValueError("AA/3Di/mask length mismatch")
    if any(type(ok) is not bool for ok in record["valid_seed_mask"]):
        raise ValueError("Seed mask must contain booleans")
    if any(
        (c == "X") == ok
        for c, ok in zip(record["three_di"], record["valid_seed_mask"], strict=True)
    ):
        raise ValueError("Every invalid search position must be X and every X invalid")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--record-id")
    parser.add_argument("--model", type=int)
    parser.add_argument(
        "--altloc", help="Explicit alternate-location label; blank atoms are shared"
    )
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = encode_chain(
        read_backbone(args.structure, args.chain, model=args.model, altloc=args.altloc),
        args.record_id or f"{args.structure.stem}:{args.chain}",
    )
    result["input"] = {
        "path": str(args.structure.resolve()),
        "sha256": hashlib.sha256(args.structure.read_bytes()).hexdigest(),
    }
    record = search_record(result)
    args.out.mkdir(parents=True)
    (args.out / "chain.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.out / "records.jsonl").write_text(json.dumps(record, allow_nan=False) + "\n")
    (args.out / "raw_3di.fasta").write_text(f">{result['record_id']}\n{result['raw_three_di']}\n")
    print(
        json.dumps(
            {
                "record_id": result["record_id"],
                "length": len(result["aa"]),
                "valid": int(np.count_nonzero(result["feature_valid_mask"])),
                "out": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
