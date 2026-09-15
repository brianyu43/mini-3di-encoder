"""Inspect one residue from PDB atoms to an official-weight 3Di state."""

import argparse
import json
from pathlib import Path

from .atoms import Chain, read_backbone
from .geometry import FEATURES, interaction_features, partner_candidates, prepare_geometry
from .network import forward, nearest_state, official_model


def trace_residue(chain: Chain, i: int) -> dict:
    if not 0 <= i < len(chain.residues):
        raise ValueError("Residue index out of range")
    layers, spec = official_model()
    geometry = prepare_geometry(chain)
    result = {
        "residue": chain.row(i),
        "valid": False,
        "invalid_reason": None,
        "state": spec["invalid_state"],
        "letter": spec["alphabet"][spec["invalid_state"]],
        "features": None,
        "network": None,
        "partner": None,
        "official_weights_sha256": spec["weights_sha256"],
    }
    if i in (0, len(chain.residues) - 1):
        result["invalid_reason"] = "chain_endpoint"
        return result
    if not geometry.valid[i]:
        result["invalid_reason"] = geometry.reasons[i]
        return result
    candidates = partner_candidates(geometry, i)
    if not candidates:
        result["invalid_reason"] = "no_valid_interior_partner"
        return result
    j = candidates[0]["index"]
    result.update(
        partner=chain.row(j),
        candidates=candidates,
        cb_used=geometry.cb_used[i].tolist(),
        cb_was_approximated=not bool(chain.present[i, 3]),
        virtual_center=geometry.virtual_centers[i].tolist(),
        partner_virtual_center=geometry.virtual_centers[j].tolist(),
    )
    try:
        features = interaction_features(chain, geometry, i, j)
    except ValueError as error:
        result["invalid_reason"] = str(error)
        return result
    result.update(
        valid=True,
        vectors=features["vectors"],
        neighbors=features["neighbors"],
        features=[
            {"index": k, "name": name, "formula": formula, "unit": unit, "value": float(value)}
            for k, ((name, formula, unit), value) in enumerate(
                zip(FEATURES, features["values"], strict=True)
            )
        ],
    )
    result["network"] = forward(layers, features["values"])
    result.update(nearest_state(result["network"]["embedding"], spec))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--index", type=int, required=True, help="0-based sequential residue index")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = trace_residue(read_backbone(args.structure, args.chain), args.index)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "state": result["state"],
                "letter": result["letter"],
                "output": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
