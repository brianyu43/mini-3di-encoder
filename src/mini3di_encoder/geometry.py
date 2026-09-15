"""Days 03–04: transparent float64 geometry following Foldseek 10-941cd33.

See THIRD_PARTY.md for the pinned upstream algorithm and constants.
Degenerate geometry is rejected explicitly, rather than assigned a normal state.
"""

import math
from dataclasses import dataclass

import numpy as np

from .atoms import Chain

PI = 3.14159265359  # Preserve the upstream constant instead of silently using math.pi.
CA_CB_DISTANCE = 1.5336
FEATURES = (
    ("angle_i", "dot(u1,u2)", "unitless"),
    ("angle_j", "dot(u3,u4)", "unitless"),
    ("i_in_to_partner", "dot(u1,u5)", "unitless"),
    ("j_in_to_partner", "dot(u3,u5)", "unitless"),
    ("i_in_j_out", "dot(u1,u4)", "unitless"),
    ("i_out_j_in", "dot(u2,u3)", "unitless"),
    ("i_in_j_in", "dot(u1,u3)", "unitless"),
    ("ca_distance", "norm(CA[j]-CA[i])", "angstrom"),
    ("signed_sequence_distance_clipped", "clip(j-i,-4,4)", "residue index"),
    ("signed_log_sequence_distance", "sign(j-i)*ln(abs(j-i)+1)", "unitless"),
)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def length(a: np.ndarray) -> float:
    return math.sqrt(dot(a, a))


def unit(a: np.ndarray) -> np.ndarray:
    norm = length(a)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Zero or nonfinite vector cannot be normalized")
    return a / norm


def approximate_cb(ca: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    v1, v2 = unit(c - ca), unit(n - ca)
    b1 = v2 + v1 / 3.0
    u1, u2 = unit(b1), unit(np.cross(v1, b1))
    direction = -v1 / 3.0 + (-u1 / 2.0 - u2 * (math.sqrt(3) / 2.0)) * (math.sqrt(8) / 3.0)
    return ca + direction * CA_CB_DISTANCE


def rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    k = unit(axis)
    return (
        v * math.cos(angle)
        + np.cross(k, v) * math.sin(angle)
        + (k * dot(k, v) * (1 - math.cos(angle)))
    )


def virtual_center(ca: np.ndarray, cb: np.ndarray, n: np.ndarray) -> np.ndarray:
    v = cb - ca
    first = rotate(v, np.cross(v, n - ca), (270.0 / 180.0) * PI)
    second = rotate(first, n - ca, 0.0)
    return ca + second * 2.0


@dataclass(frozen=True)
class Geometry:
    cb_used: np.ndarray
    virtual_centers: np.ndarray
    valid: np.ndarray
    reasons: tuple[str, ...]


def prepare_geometry(chain: Chain) -> Geometry:
    n_residues = len(chain.residues)
    cb_used = np.full((n_residues, 3), np.nan)
    centers = np.full_like(cb_used, np.nan)
    valid = chain.backbone_valid.copy()
    reasons = []
    for i in range(n_residues):
        if not valid[i]:
            reasons.append("missing_backbone_atom")
            continue
        n, ca, c, cb = chain.xyz[i]
        try:
            cb_used[i] = cb if chain.present[i, 3] else approximate_cb(ca, n, c)
            centers[i] = virtual_center(ca, cb_used[i], n)
            reasons.append("valid")
        except ValueError:
            valid[i] = False
            reasons.append("degenerate_geometry")
    return Geometry(cb_used, centers, valid, tuple(reasons))


def partner_candidates(geometry: Geometry, i: int) -> list[dict]:
    size = len(geometry.valid)
    if not 0 <= i < size:
        raise ValueError("Residue index out of range")
    if i in (0, size - 1) or not geometry.valid[i]:
        return []
    candidates = [
        {"index": j, "distance": length(geometry.virtual_centers[i] - geometry.virtual_centers[j])}
        for j in range(1, size - 1)
        if j != i and geometry.valid[j]
    ]
    # Upstream scans j in ascending order with '<': exact ties choose the lower j.
    return sorted(candidates, key=lambda row: (row["distance"], row["index"]))


def interaction_features(chain: Chain, geometry: Geometry, i: int, j: int) -> dict:
    if i == j or min(i, j) < 1 or max(i, j) >= len(chain.residues) - 1:
        raise ValueError("Both residues must be distinct interior positions")
    neighbors = [i - 1, i, i + 1, j - 1, j, j + 1]
    if not geometry.valid[neighbors].all():
        raise ValueError("A six-residue neighborhood is invalid")
    ca = chain.xyz[:, 1]
    u1, u2 = unit(ca[i] - ca[i - 1]), unit(ca[i + 1] - ca[i])
    u3, u4 = unit(ca[j] - ca[j - 1]), unit(ca[j + 1] - ca[j])
    u5 = unit(ca[j] - ca[i])
    delta = j - i
    values = np.array(
        [
            dot(u1, u2),
            dot(u3, u4),
            dot(u1, u5),
            dot(u3, u5),
            dot(u1, u4),
            dot(u2, u3),
            dot(u1, u3),
            length(ca[i] - ca[j]),
            max(-4, min(4, delta)),
            math.copysign(math.log(abs(delta) + 1), delta),
        ],
        dtype=np.float64,
    )
    return {
        "values": values,
        "neighbors": neighbors,
        "vectors": {f"u{k}": vec.tolist() for k, vec in enumerate([u1, u2, u3, u4, u5], 1)},
    }
