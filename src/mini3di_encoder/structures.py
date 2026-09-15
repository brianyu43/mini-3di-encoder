"""Day 01: inspect a clean PDB chain without inventing missing coordinates."""

from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1


def read_ca_rows(path: Path, chain_id: str) -> list[dict]:
    """Keep PDB numbering and insertion codes distinct from sequence indices.

    This first-day reader accepts one model and standard ATOM residues with
    unambiguous CA coordinates. It is not yet a general Foldseek input parser.
    Water and other HETATM records are excluded explicitly.
    """
    structure = PDBParser(PERMISSIVE=False, QUIET=False).get_structure(path.stem, path)
    models = list(structure)
    if len(models) != 1:
        raise ValueError("Day 01 requires exactly one PDB model")
    if chain_id not in models[0]:
        raise ValueError(f"Chain {chain_id!r} not found")
    rows = []
    for residue in models[0][chain_id]:
        if residue.id[0] != " ":
            continue
        if residue.is_disordered():
            raise ValueError(f"Alternate/disordered residue is outside day 01: {residue.id}")
        if residue.resname not in protein_letters_3to1:
            raise ValueError(f"Nonstandard residue: {residue.resname}")
        if "CA" not in residue:
            raise ValueError(f"Missing CA: {residue.id}")
        xyz = residue["CA"].coord
        if not np.isfinite(xyz).all():
            raise ValueError(f"Nonfinite CA: {residue.id}")
        rows.append(
            {
                "sequence_index_0": len(rows),
                "chain": chain_id,
                "pdb_residue_number": residue.id[1],
                "insertion_code": residue.id[2].strip(),
                "residue_name": residue.resname,
                "aa": protein_letters_3to1[residue.resname],
                "ca_x": float(xyz[0]),
                "ca_y": float(xyz[1]),
                "ca_z": float(xyz[2]),
            }
        )
    if not rows:
        raise ValueError("No standard residues in selected chain")
    return rows
