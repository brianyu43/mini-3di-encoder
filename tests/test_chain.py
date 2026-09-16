from pathlib import Path

from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.chain import compare_sequence, encode_chain

ROOT = Path(__file__).parents[1]
REFERENCE = "DWEWEAEPVRDIDIDDDDQQAFPLVVLVVVCVPPNDDSVFWFKADPRDTGDGGHGNVVVPQDPHYYIYIYTHDDDD"
MISSING_CA10_REFERENCE = (
    "DWEWEAEPVDIDIDDDDQQAFPLVVLVVVCVPPNDDSVFWFKADPRDTGDGGHGNVVVPQDPHYYIYIYTHDDDD"
)


def test_1ubq_full_chain_matches_foldseek_10_941cd33():
    encoded = encode_chain(read_backbone(ROOT / "examples/1UBQ.pdb", "A"))

    comparison = compare_sequence(encoded.letters, REFERENCE)

    assert comparison.equal
    assert comparison.mismatches == ()
    assert sum(encoded.valid) == 74
    assert encoded.invalid_reasons[0] == "chain_endpoint"
    assert encoded.invalid_reasons[-1] == "chain_endpoint"


def test_comparison_reports_substitution_and_length_mismatch():
    comparison = compare_sequence("ACD", "AX")

    assert not comparison.equal
    assert comparison.observed_length == 3
    assert comparison.reference_length == 2
    assert comparison.mismatches == (
        {"index": 1, "observed": "C", "reference": "X"},
        {"index": 2, "observed": "D", "reference": None},
    )


def test_missing_backbone_exposes_unsupported_row_alignment(tmp_path):
    lines = (ROOT / "examples/1UBQ.pdb").read_text().splitlines(keepends=True)
    missing_ca = tmp_path / "1UBQ_missing_CA10.pdb"
    missing_ca.write_text(
        "".join(
            line
            for line in lines
            if not (
                line.startswith("ATOM  ")
                and line[21] == "A"
                and int(line[22:26]) == 10
                and line[12:16].strip() == "CA"
            )
        )
    )

    encoded = encode_chain(read_backbone(missing_ca, "A"))
    comparison = compare_sequence(encoded.letters, MISSING_CA10_REFERENCE)

    assert not comparison.equal
    assert comparison.observed_length == 76
    assert comparison.reference_length == 75
    assert encoded.invalid_reasons[8:11] == (
        "A six-residue neighborhood is invalid",
        "missing_backbone_atom",
        "A six-residue neighborhood is invalid",
    )
