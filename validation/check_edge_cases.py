"""Compare short/missing-atom coordinate inputs to the pinned native oracle.

This bypasses both file parsers: a missing CA remains an explicit position here,
whereas the official Foldseek file parser can discard that residue.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from reference import native_trace, save, sha

from mini3di_encoder.atoms import Chain, read_backbone
from mini3di_encoder.encode import encode_chain

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original = read_backbone(ROOT / "examples/1UBQ.pdb", "A")
    cases = []
    for n in (1, 2, 3):
        cases.append((f"short-{n}", Chain("A", original.residues[:n], original.xyz[:n])))
    for k, name in enumerate(("N", "CA", "C", "CB")):
        xyz = original.xyz.copy()
        xyz[19, k] = np.nan
        cases.append((f"missing-{name}", replace(original, xyz=xyz)))
    weights = ROOT / "src/mini3di_encoder/data/encoder.kerasify"
    summaries = []
    for name, chain in cases:
        directory = args.out / name
        directory.mkdir()
        own = encode_chain(chain, name)
        ref = native_trace(chain, args.oracle, weights, directory)
        assert len(own["rows"]) == len(ref)
        for a, b in zip(own["rows"], ref, strict=True):
            assert a["state"] == b["state"] and a["valid"] == b["valid"]
            if a["valid"]:
                assert a["partner"] == b["partner"]
                np.testing.assert_allclose(a["features"], b["features"], rtol=0, atol=1e-10)
                np.testing.assert_allclose(a["embedding"], b["embedding"], rtol=0, atol=1e-6)
        save(directory / "own.json", own)
        save(directory / "native.json", ref)
        summaries.append({"case": name, "positions": len(ref), "mismatches": 0})
    save(
        args.out / "summary.json",
        {
            "scope": "Coordinate-level native comparison, not official file-parser equivalence",
            "oracle_sha256": sha(args.oracle),
            "weights_sha256": sha(weights),
            "script_sha256": sha(Path(__file__)),
            "cases": summaries,
        },
    )
    print(summaries)


if __name__ == "__main__":
    main()
