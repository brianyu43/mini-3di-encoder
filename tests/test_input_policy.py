from pathlib import Path

import numpy as np
import pytest

from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.encode import encode_chain

PDB = Path(__file__).resolve().parents[1] / "examples/1UBQ.pdb"


def write(tmp_path, lines):
    path = tmp_path / "input.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_explicit_conformer_selection_never_mixes_a_and_b(tmp_path):
    lines = []
    for line in PDB.read_text().splitlines():
        if line.startswith("ATOM  ") and int(line[22:26]) == 2 and line[12:16].strip() == "CB":
            lines.extend(
                [
                    line[:16] + "A" + line[17:],
                    line[:16] + "B" + line[17:30] + f"{99.0:8.3f}" + line[38:],
                ]
            )
        else:
            lines.append(line)
    path = write(tmp_path, lines)
    with pytest.raises(ValueError, match="Alternate"):
        read_backbone(path, "A")
    original = read_backbone(PDB, "A")
    selected = read_backbone(path, "A", altloc="A")
    np.testing.assert_array_equal(selected.xyz, original.xyz)
    assert selected.altloc == "A"
    assert read_backbone(path, "A", altloc="B").xyz[1, 3, 0] == 99.0
    # If a requested conformer is absent, do not borrow another conformer's atom.
    assert not read_backbone(path, "A", altloc="C").present[1, 3]


def test_explicit_model_and_chain_selection(tmp_path):
    atoms = [line for line in PDB.read_text().splitlines() if line.startswith("ATOM  ")]
    second = [
        line[:21] + "B" + line[22:30] + f"{float(line[30:38]) + 10:8.3f}" + line[38:]
        for line in atoms
    ]
    path = write(
        tmp_path, ["MODEL        1", *atoms, "ENDMDL", "MODEL        2", *second, "ENDMDL"]
    )
    with pytest.raises(ValueError, match="one model"):
        read_backbone(path, "A")
    a, b = read_backbone(path, "A", model=1), read_backbone(path, "B", model=2)
    assert a.aa == b.aa and b.model_id == 2
    np.testing.assert_array_equal(a.present, b.present)
    np.testing.assert_allclose(
        (b.xyz - a.xyz)[a.present],
        np.broadcast_to([10, 0, 0], a.xyz[a.present].shape),
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="absent"):
        read_backbone(path, "A", model=3)
    with pytest.raises(ValueError, match="No standard"):
        read_backbone(path, "B", model=1)


def test_ter_creates_a_boundary_even_when_coordinates_are_close(tmp_path):
    lines = []
    inserted = False
    for line in PDB.read_text().splitlines():
        if not inserted and line.startswith("ATOM  ") and int(line[22:26]) == 39:
            lines.append("TER                  A")
            inserted = True
        lines.append(line)
    chain = read_backbone(write(tmp_path, lines), "A")
    assert len(chain.residues) == 76
    assert chain.segment(37) == 0 and chain.segment(38) == 1
    encoded = encode_chain(chain, "ter")
    assert not encoded["peptide_links"][37]
    assert encoded["three_di"][37:39] == "XX"


def test_insertion_codes_and_numbering_gaps_do_not_change_sequence_distance(tmp_path):
    lines = []
    for line in PDB.read_text().splitlines():
        if line.startswith("ATOM  "):
            number = int(line[22:26])
            line = line[:22] + f"{number * 3:4d}" + ("A" if number == 2 else " ") + line[27:]
        lines.append(line)
    chain = read_backbone(write(tmp_path, lines), "A")
    assert chain.residues[1][:2] == (6, "A")
    original = encode_chain(read_backbone(PDB, "A"), "original")
    changed = encode_chain(chain, "renumbered")
    assert changed["raw_three_di"] == original["raw_three_di"]
    assert changed["valid_seed_mask"] == original["valid_seed_mask"]


def test_mmcif_is_explicitly_unsupported(tmp_path):
    with pytest.raises(ValueError, match="mmCIF"):
        read_backbone(write(tmp_path, ["data_test", "loop_"]), "A")


def test_unknown_amino_acid_does_not_discard_observed_structure(tmp_path):
    lines = []
    for line in PDB.read_text().splitlines():
        if line.startswith("ATOM  ") and int(line[22:26]) == 2:
            line = line[:17] + "UNK" + line[20:]
        lines.append(line)
    encoded = encode_chain(read_backbone(write(tmp_path, lines), "A"), "unknown")
    original = encode_chain(read_backbone(PDB, "A"), "original")
    assert encoded["aa"][1] == "X"
    assert encoded["raw_three_di"] == original["raw_three_di"]
    assert encoded["valid_seed_mask"] == original["valid_seed_mask"]
