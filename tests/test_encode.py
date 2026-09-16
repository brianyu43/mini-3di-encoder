import json
from pathlib import Path

import numpy as np
import pytest

from mini3di_encoder.atoms import Chain, read_backbone
from mini3di_encoder.encode import encode_chain, search_record

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "examples/1UBQ.pdb"


def test_whole_chain_against_independent_native_states_masks_and_features():
    expected = json.loads((ROOT / "tests/fixtures/1ubq_chain_native.json").read_text())
    result = encode_chain(read_backbone(PDB, "A"), "1UBQ:A")
    assert len(result["aa"]) == len(expected) == 76
    assert sum(result["feature_valid_mask"]) == sum(result["valid_seed_mask"]) == 74
    official = (ROOT / "tests/fixtures/1ubq_official_3di.fasta").read_text().splitlines()[1]
    assert result["raw_three_di"] == official
    for row, ref in zip(result["rows"], expected, strict=True):
        assert row["state"] == ref["state"]
        assert row["valid"] == ref["valid"]
        if row["valid"]:
            assert row["partner"] == ref["partner"]
            np.testing.assert_allclose(row["features"], ref["features"], rtol=0, atol=1e-10)
            np.testing.assert_allclose(row["embedding"], ref["embedding"], rtol=0, atol=1e-6)
    # Valid D residues remain D; only invalid positions become X.
    record = search_record(result)
    assert record["three_di"][0] == record["three_di"][-1] == "X"
    assert any(
        c == "D" and m for c, m in zip(record["three_di"], record["valid_seed_mask"], strict=True)
    )
    assert set(record) == {"record_id", "aa", "three_di", "valid_seed_mask", "synthetic"}
    assert record["synthetic"] is False


@pytest.mark.parametrize("length", [1, 2, 3])
def test_short_chains_have_explicit_invalid_states_without_crashing(length):
    original = read_backbone(PDB, "A")
    chain = Chain("A", original.residues[:length], original.xyz[:length])
    encoded = encode_chain(chain, "short")
    assert encoded["raw_three_di"] == "D" * length
    assert encoded["three_di"] == "X" * length
    assert not any(encoded["feature_valid_mask"])


@pytest.mark.parametrize("atom", ["N", "CA", "C", "CB"])
def test_missing_atoms_preserve_length_id_and_observed_coordinates(tmp_path, atom):
    text = (
        "\n".join(
            line
            for line in PDB.read_text().splitlines()
            if not (
                line.startswith("ATOM  ") and int(line[22:26]) == 20 and line[12:16].strip() == atom
            )
        )
        + "\n"
    )
    path = tmp_path / "missing.pdb"
    path.write_text(text)
    chain = read_backbone(path, "A")
    saved = chain.xyz.copy()
    result = encode_chain(chain, "missing")
    assert len(result["aa"]) == 76 and chain.residues[19][0] == 20
    np.testing.assert_array_equal(chain.xyz, saved)
    if atom == "CB":
        assert result["feature_valid_mask"][19]
    else:
        assert not result["residue_valid_mask"][19]
        assert not any(result["valid_seed_mask"][18:21])
        assert result["three_di"][18:21] == "XXX"
    json.dumps(result, allow_nan=False)


def test_spatial_chain_break_masks_both_backbone_neighbors():
    chain = read_backbone(PDB, "A")
    moved = chain.xyz.copy()
    moved[38:] += [100.0, 0, 0]
    broken = Chain("A", chain.residues, moved)
    encoded = encode_chain(broken, "broken")
    assert not encoded["peptide_links"][37]
    assert not any(encoded["valid_seed_mask"][37:39])
    assert encoded["three_di"][37:39] == "XX"
    for row in encoded["rows"]:
        if row["seed_valid"]:
            i, j = row["index"], row["partner"]
            assert not ({i - 1, i, j - 1, j} & {37})


def test_export_rejects_mask_token_and_length_disagreement():
    result = encode_chain(read_backbone(PDB, "A"), "sample")
    with pytest.raises(ValueError, match="length"):
        search_record({**result, "aa": "A"})
    with pytest.raises(ValueError, match="invalid"):
        search_record({**result, "three_di": result["raw_three_di"]})
    with pytest.raises(ValueError, match="booleans"):
        search_record({**result, "valid_seed_mask": [0] * 76})
