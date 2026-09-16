"""Explicit PDB residue identities and unmodified float64 backbone atoms."""

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
# UNK preserves an unknown amino acid while still allowing observed geometry.
# Pinned GemmiWrapper::threeToOneAA also maps UNK to X.
AMINO_ACIDS["UNK"] = "X"


@dataclass(frozen=True)
class Chain:
    chain_id: str
    residues: tuple[tuple[int, str, str], ...]
    xyz: np.ndarray  # [residue, N/CA/C/CB, x/y/z]; missing atoms are NaN, never zero.
    segment_ids: tuple[int, ...] = ()
    model_id: int = 1
    altloc: str | None = None

    def __post_init__(self):
        if self.xyz.shape != (len(self.residues), 4, 3):
            raise ValueError("Coordinates must have shape [residues,4,3]")
        if self.segment_ids and len(self.segment_ids) != len(self.residues):
            raise ValueError("Segment/residue length mismatch")

    def segment(self, i: int) -> int:
        return self.segment_ids[i] if self.segment_ids else 0

    def peptide_links(self, maximum_distance: float = 2.0) -> tuple[bool, ...]:
        """Conservative search-safety policy, not a Foldseek feature or learned parameter.

        A link requires the same TER segment and a finite C(i)-N(i+1) distance in
        (0, 2.0] angstrom. Numbering gaps alone do not imply a broken peptide bond.
        """
        links = []
        for i in range(len(self.residues) - 1):
            distance = np.linalg.norm(self.xyz[i, 2] - self.xyz[i + 1, 0])
            links.append(
                bool(
                    self.segment(i) == self.segment(i + 1)
                    and np.isfinite(distance)
                    and 0 < distance <= maximum_distance
                )
            )
        return tuple(links)

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
            "model": self.model_id,
            "segment": self.segment(i),
            "selected_altloc": self.altloc,
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


def read_backbone(
    path: Path,
    chain_id: str,
    *,
    model: int | None = None,
    altloc: str | None = None,
) -> Chain:
    """PDB ATOM reader with explicit model/conformer selection and TER segments.

    Missing atoms remain NaN. Blank altloc atoms are shared; nonblank alternatives
    require an explicit label, and different conformers are never mixed. mmCIF and
    modified ATOM residues are unsupported and must not be silently normalized.
    """
    if len(chain_id) != 1:
        raise ValueError("PDB chain ID must have one character")
    if altloc is not None and (len(altloc) != 1 or not altloc.isalnum()):
        raise ValueError("altloc must be one alphanumeric PDB label")
    lines = path.read_text().splitlines()
    if any(line.startswith("data_") for line in lines):
        raise ValueError("mmCIF is not supported; provide a PDB file")
    models = [int(line[10:14]) for line in lines if line.startswith("MODEL ")]
    if len(set(models)) != len(models):
        raise ValueError("Duplicate MODEL identifiers")
    if model is None and len(models) > 1:
        raise ValueError("Only one model is supported")
    chosen_model = model if model is not None else (models[0] if models else 1)
    if chosen_model not in (models or [1]):
        raise ValueError(f"Model {chosen_model} is absent")
    current_model = None if models else 1
    residues, coordinates, seen_atoms = [], [], set()
    segments = []
    index = {}
    pending_segment = False
    segment = 0
    for line_no, line in enumerate(lines, 1):
        if line.startswith("MODEL "):
            current_model = int(line[10:14])
            continue
        if line.startswith("ENDMDL"):
            current_model = None
            continue
        if current_model != chosen_model:
            continue
        if line.startswith("TER") and len(line) > 21 and line[21] == chain_id:
            pending_segment = True
        if not line.startswith("ATOM  ") or len(line) <= 21 or line[21] != chain_id:
            continue
        if pending_segment:
            segment += 1
            pending_segment = False
        if len(line) < 54:
            raise ValueError(f"Truncated ATOM line {line_no}")
        label = line[16].strip()
        if label and altloc is None:
            raise ValueError(f"Alternate location on line {line_no}; select altloc explicitly")
        name = line[17:20]
        if name not in AMINO_ACIDS:
            raise ValueError(f"Nonstandard ATOM residue {name}")
        identity = (segment, int(line[22:26]), line[26].strip())
        if identity not in index:
            index[identity] = len(residues)
            residues.append((*identity[1:], name))
            segments.append(segment)
            coordinates.append(np.full((4, 3), np.nan, dtype=np.float64))
        i = index[identity]
        if i != len(residues) - 1 or residues[i][2] != name:
            raise ValueError("Noncontiguous or ambiguous residue identity")
        atom = line[12:16].strip()
        if (i, atom, label) in seen_atoms:
            raise ValueError(f"Duplicate atom {identity} {atom} conformer {label!r}")
        seen_atoms.add((i, atom, label))
        if label and label != altloc:
            continue
        if atom not in ATOM_NAMES:
            continue
        if np.isfinite(coordinates[i][ATOM_NAMES.index(atom)]).any():
            raise ValueError(f"Duplicate shared/selected atom {identity} {atom}")
        xyz = np.array([float(line[a:b]) for a, b in [(30, 38), (38, 46), (46, 54)]])
        if not np.isfinite(xyz).all():
            raise ValueError(f"Nonfinite observed atom on line {line_no}")
        coordinates[i][ATOM_NAMES.index(atom)] = xyz
    if not residues:
        raise ValueError(f"No standard ATOM residues in chain {chain_id!r}")
    xyz = np.stack(coordinates)
    xyz.setflags(write=False)
    return Chain(chain_id, tuple(residues), xyz, tuple(segments), chosen_model, altloc)
