"""Encode a supported PDB chain and compare it with an external 3Di sequence."""

from dataclasses import dataclass

from .atoms import Chain
from .trace import trace_residue


@dataclass(frozen=True)
class ChainEncoding:
    letters: str
    valid: tuple[bool, ...]
    invalid_reasons: tuple[str | None, ...]


@dataclass(frozen=True)
class SequenceComparison:
    equal: bool
    observed_length: int
    reference_length: int
    mismatches: tuple[dict, ...]


def encode_chain(chain: Chain) -> ChainEncoding:
    """Encode every preserved residue row; invalid rows retain the official D sentinel."""
    rows = tuple(trace_residue(chain, i) for i in range(len(chain.residues)))
    return ChainEncoding(
        letters="".join(row["letter"] for row in rows),
        valid=tuple(row["valid"] for row in rows),
        invalid_reasons=tuple(row["invalid_reason"] for row in rows),
    )


def compare_sequence(observed: str, reference: str) -> SequenceComparison:
    """Compare by position and expose length differences instead of truncating with zip."""
    common = min(len(observed), len(reference))
    mismatches = [
        {"index": i, "observed": observed[i], "reference": reference[i]}
        for i in range(common)
        if observed[i] != reference[i]
    ]
    mismatches.extend(
        {
            "index": i,
            "observed": observed[i] if i < len(observed) else None,
            "reference": reference[i] if i < len(reference) else None,
        }
        for i in range(common, max(len(observed), len(reference)))
    )
    return SequenceComparison(
        equal=not mismatches,
        observed_length=len(observed),
        reference_length=len(reference),
        mismatches=tuple(mismatches),
    )
