"""Boundary cases for coordinate perturbations and positional soft-seed semantics."""

import numpy as np

from experiments.followup.evaluation import (
    IndexConfig,
    ProteinRecord,
    build_index,
    collect_hits,
    soft_hits,
)
from experiments.followup.stability import perturb, stability_counts


def record(sid, sequence, mask=None):
    return ProteinRecord(
        sid,
        "A" * len(sequence),
        sequence,
        tuple([True] * len(sequence) if mask is None else mask),
        False,
    )


def test_soft_seed_one_change_and_invalid_windows():
    query = record("q", "ACDXACD", [True, True, True, False, True, True, True])
    targets = [record("a", "ACE"), record("b", "AED"), record("c", "AEE")]
    index = build_index(targets, IndexConfig(3), allow_real=True)
    second = np.array([1, 3, 3, 0, 1, 3, 3])  # E alternatives at positions 1/2.
    hits = soft_hits(query, index, second, np.zeros(7), 0.01)
    observed = {(index.targets[h.target_numeric_id].record_id, h.query_start) for h in hits}
    assert observed == {("a", 0), ("a", 4), ("b", 0), ("b", 4)}
    assert len(hits) == len(set(hits))
    assert set(collect_hits(query, index)).issubset(hits)
    assert soft_hits(query, index, second, np.ones(7), 0.01) == collect_hits(query, index)


def test_rigid_transform_and_seeded_noise_preserve_missing_atoms():
    xyz = np.arange(36, dtype=float).reshape(3, 4, 3)
    xyz[0, 3] = np.nan
    rigid = perturb(xyz, "q", "rigid_float64")
    np.testing.assert_allclose(
        np.linalg.norm(xyz[0, 0] - xyz[2, 1]), np.linalg.norm(rigid[0, 0] - rigid[2, 1]), atol=1e-12
    )
    small = perturb(xyz, "q", "gaussian_0.001_A")
    large = perturb(xyz, "q", "gaussian_0.01_A")
    np.testing.assert_allclose((small - xyz) * 10, large - xyz, atol=1e-12)
    np.testing.assert_array_equal(np.isnan(small), np.isnan(xyz))
    np.testing.assert_array_equal(small, perturb(xyz, "q", "gaussian_0.001_A"))


def test_state_change_is_separated_from_partner_and_mask_changes():
    before = {
        "mask": np.ones(5, dtype=bool),
        "partners": np.arange(5),
        "states": np.zeros(5, dtype=int),
        "margin": np.full(5, 0.02),
    }
    after = {k: v.copy() for k, v in before.items()}
    after["mask"][4] = False
    after["states"][1:3] = 1
    after["partners"][1] = 4
    result = stability_counts(before, after)
    assert result["common_valid"] == 4
    assert result["partner_changes"] == 1
    assert result["state_changes_same_partner"] == 1
    assert result["mask_changes"] == 1
    assert result["valid_3mers"] == result["broken_3mers"] == 3
