from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from mini3di_encoder.atoms import ATOM_NAMES, read_backbone

PDB = Path(__file__).resolve().parents[1] / "data/raw/1UBQ.pdb"


def atom(serial, name, residue="ALA", number=7, insertion=" ", alt=" "):
    return (
        f"ATOM  {serial:5d} {name:^4}{alt}{residue:3} A{number:4d}{insertion}   "
        f"{1.123:8.3f}{2.456:8.3f}{3.789:8.3f}{1.0:6.2f}{20.0:6.2f}          C  \n"
    )


def fixture(tmp_path, text):
    path = tmp_path / "atoms.pdb"
    path.write_text(text)
    return path


def test_missing_atoms_and_glycine_are_distinct(tmp_path):
    text = atom(1, "CA", "GLY") + atom(2, "CA", "ALA", insertion="A")
    chain = read_backbone(fixture(tmp_path, text), "A")
    assert chain.residues == ((7, "", "GLY"), (7, "A", "ALA"))
    assert chain.aa == "GA"
    assert chain.row(0)["cb_status"] == "glycine_no_cb"
    assert chain.row(1)["cb_status"] == "missing_cb"
    assert chain.row(0)["atoms"]["N"] == {"present": False, "xyz": None}
    assert not chain.backbone_valid.any()
    assert chain.xyz.dtype == np.float64
    assert chain.xyz[0, 1, 0] == 1.123  # no intermediate float32 rounding
    assert not chain.xyz.flags.writeable


@pytest.mark.parametrize(
    "text,match",
    [
        (atom(1, "N") + atom(2, "N"), "Duplicate"),
        (atom(1, "N", alt="A"), "Alternate"),
        (atom(1, "N", residue="UNK"), "Nonstandard"),
        ("MODEL        1\n" + atom(1, "N") + "ENDMDL\nMODEL        2\n", "one model"),
        (atom(1, "N") + "TER                  A\n" + atom(2, "CA"), "after TER"),
    ],
)
def test_ambiguous_inputs_are_rejected(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        read_backbone(fixture(tmp_path, text), "A")


def test_real_atom_inventory_against_biopython():
    chain = read_backbone(PDB, "A")
    other = [r for r in PDBParser(QUIET=True).get_structure("x", PDB)[0]["A"] if r.id[0] == " "]
    assert len(chain.residues) == len(other) == 76
    assert chain.present.sum(axis=0).tolist() == [76, 76, 76, 70]
    for i, residue in enumerate(other):
        assert chain.residues[i] == (residue.id[1], residue.id[2].strip(), residue.resname)
        for k, atom_name in enumerate(ATOM_NAMES):
            assert bool(chain.present[i, k]) == (atom_name in residue)
            if atom_name in residue:
                np.testing.assert_allclose(
                    chain.xyz[i, k], residue[atom_name].coord, atol=2e-6, rtol=0
                )
