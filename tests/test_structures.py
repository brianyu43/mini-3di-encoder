"""Handwritten PDB fixtures protect residue mapping and explicit failure rules."""

import pytest

from mini3di_encoder.structures import read_ca_rows


def atom(serial=1, name="CA", residue="ALA", number=7, insertion=" ", alt=" ", x=1.0):
    return (
        f"ATOM  {serial:5d} {name:^4}{alt}{residue:3} A{number:4d}{insertion}   "
        f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{20.0:6.2f}          C  \n"
    )


def pdb(tmp_path, text):
    path = tmp_path / "fixture.pdb"
    path.write_text(text)
    return path


def test_number_insertion_and_sequence_index_are_distinct(tmp_path):
    path = pdb(tmp_path, atom() + atom(serial=2, residue="GLY", insertion="A", x=4))
    rows = read_ca_rows(path, "A")
    assert [
        (r["sequence_index_0"], r["pdb_residue_number"], r["insertion_code"], r["aa"]) for r in rows
    ] == [(0, 7, "", "A"), (1, 7, "A", "G")]
    assert (rows[0]["ca_x"], rows[0]["ca_y"], rows[0]["ca_z"]) == (1, 2, 3)


def test_missing_ca_is_not_silently_dropped(tmp_path):
    with pytest.raises(ValueError, match="Missing CA"):
        read_ca_rows(pdb(tmp_path, atom(name="N")), "A")


def test_altloc_requires_later_policy(tmp_path):
    with pytest.raises(ValueError, match="Alternate"):
        read_ca_rows(pdb(tmp_path, atom(alt="A")), "A")


def test_nonstandard_residue_is_not_silently_mapped(tmp_path):
    with pytest.raises(ValueError, match="Nonstandard"):
        read_ca_rows(pdb(tmp_path, atom(residue="UNK")), "A")


def test_missing_chain(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        read_ca_rows(pdb(tmp_path, atom()), "B")


def test_nonfinite_ca(tmp_path):
    with pytest.raises(ValueError, match="Nonfinite"):
        read_ca_rows(pdb(tmp_path, atom(x=float("nan"))), "A")


def test_multiple_models_are_not_silently_collapsed(tmp_path):
    text = "MODEL        1\n" + atom() + "ENDMDL\nMODEL        2\n" + atom() + "ENDMDL\n"
    with pytest.raises(ValueError, match="exactly one"):
        read_ca_rows(pdb(tmp_path, text), "A")
