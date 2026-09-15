"""Day 02: explicit PDB residue identities and unmodified float64 backbone atoms."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ATOM_NAMES = ("N", "CA", "C", "CB")
AMINO_ACIDS = dict(
    zip(
        "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
        "ACDEFGHIKLMNPQRSTVWY",
        strict=True,
    )
)


@dataclass(frozen=True)
class Chain:
    chain_id: str
    residues: tuple[tuple[int, str, str], ...]
    xyz: np.ndarray  # [residue, N/CA/C/CB, x/y/z]; missing atoms are NaN, never zero.

    @property
    def present(self) -> np.ndarray:
        return np.isfinite(self.xyz).all(axis=2)

    @property
    def backbone_valid(self) -> np.ndarray:
        return self.present[:, :3].all(axis=1)

    @property
    def aa(self) -> str:
        return "".join(AMINO_ACIDS[name] for _, _, name in self.residues)

    def row(self, i: int) -> dict:
        number, insertion, name = self.residues[i]
        present = self.present[i]
        cb_status = (
            "observed" if present[3] else ("glycine_no_cb" if name == "GLY" else "missing_cb")
        )
        return {
            "sequence_index_0": i,
            "chain": self.chain_id,
            "pdb_residue_number": number,
            "insertion_code": insertion,
            "residue_name": name,
            "aa": self.aa[i],
            "cb_status": cb_status,
            "backbone_valid": bool(self.backbone_valid[i]),
            "atoms": {
                atom: {
                    "present": bool(present[k]),
                    "xyz": self.xyz[i, k].tolist() if present[k] else None,
                }
                for k, atom in enumerate(ATOM_NAMES)
            },
        }


def read_backbone(path: Path, chain_id: str) -> Chain:
    """Clean single-model PDB subset. Preserve missing atoms and reject ambiguity.

    Parse decimal coordinates directly as float64, like pinned Gemmi/Vec3.
    The day-01 Bio.PDB reader remains separate (Bio.PDB stores float32).
    Alternate locations, modified residues, split chains and mmCIF are deferred.
    """
    if len(chain_id) != 1:
        raise ValueError("PDB chain ID must have one character")
    lines = path.read_text().splitlines()
    if sum(line.startswith("MODEL ") for line in lines) > 1:
        raise ValueError("Only one model is supported")
    residues, coordinates, seen_atoms = [], [], set()
    index = {}
    closed = False
    for line_no, line in enumerate(lines, 1):
        if line.startswith("TER") and len(line) > 21 and line[21] == chain_id:
            closed = True
        if not line.startswith("ATOM  ") or len(line) <= 21 or line[21] != chain_id:
            continue
        if closed:
            raise ValueError("Selected chain continues after TER; policy deferred to day 08")
        if len(line) < 54:
            raise ValueError(f"Truncated ATOM line {line_no}")
        if line[16] != " ":
            raise ValueError(f"Alternate location on line {line_no}; policy deferred")
        name = line[17:20]
        if name not in AMINO_ACIDS:
            raise ValueError(f"Nonstandard ATOM residue {name}")
        identity = (int(line[22:26]), line[26].strip())
        if identity not in index:
            index[identity] = len(residues)
            residues.append((*identity, name))
            coordinates.append(np.full((4, 3), np.nan, dtype=np.float64))
        i = index[identity]
        if i != len(residues) - 1 or residues[i][2] != name:
            raise ValueError("Noncontiguous or ambiguous residue identity")
        atom = line[12:16].strip()
        if (i, atom) in seen_atoms:
            raise ValueError(f"Duplicate atom {identity} {atom}")
        seen_atoms.add((i, atom))
        if atom not in ATOM_NAMES:
            continue
        xyz = np.array([float(line[a:b]) for a, b in [(30, 38), (38, 46), (46, 54)]])
        if not np.isfinite(xyz).all():
            raise ValueError(f"Nonfinite observed atom on line {line_no}")
        coordinates[i][ATOM_NAMES.index(atom)] = xyz
    if not residues:
        raise ValueError(f"No standard ATOM residues in chain {chain_id!r}")
    xyz = np.stack(coordinates)
    xyz.setflags(write=False)
    return Chain(chain_id, tuple(residues), xyz)
