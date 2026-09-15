import math

import numpy as np
import pytest

from mini3di_encoder.atoms import Chain
from mini3di_encoder.geometry import (
    CA_CB_DISTANCE,
    Geometry,
    approximate_cb,
    interaction_features,
    partner_candidates,
    prepare_geometry,
    rotate,
    unit,
    virtual_center,
)


def line_chain():
    xyz = np.zeros((8, 4, 3))
    for i in range(8):
        xyz[i] = [(i, 1, 0), (i, 0, 0), (i + 1, 0, 0), (i, 0, 1)]
    return Chain("A", tuple((10 + i * 3, "", "ALA") for i in range(8)), xyz)


def test_vector_operations_have_hand_computable_answers():
    np.testing.assert_allclose(unit(np.array([3.0, 4.0, 0.0])), [0.6, 0.8, 0])
    np.testing.assert_allclose(
        rotate(np.array([1.0, 0, 0]), np.array([0.0, 0, 1]), math.pi / 2), [0, 1, 0], atol=1e-15
    )
    for bad in [np.zeros(3), np.array([np.nan, 0, 0])]:
        with pytest.raises(ValueError, match="normalized"):
            unit(bad)


def test_tetrahedral_cb_and_virtual_distance():
    ca = np.zeros(3)
    n, c = np.array([-1 / 3, math.sqrt(8) / 3, 0]), np.array([1.0, 0, 0])
    cb = approximate_cb(ca, n, c)
    assert np.linalg.norm(cb - ca) == pytest.approx(CA_CB_DISTANCE)
    center = virtual_center(ca, cb, n)
    assert np.linalg.norm(center - ca) == pytest.approx(2 * CA_CB_DISTANCE)


def test_partner_exclusions_and_exact_distance_tie():
    # Ends are closest but excluded; residue 2 and 4 tie about query 3.
    points = np.array(
        [[0.0, 0, 0], [8, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [9, 0, 0], [0, 0, 0]]
    )
    mask = np.ones(7, dtype=bool)
    geometry = Geometry(points.copy(), points, mask, tuple("valid" for _ in mask))
    candidates = partner_candidates(geometry, 3)
    assert [row["index"] for row in candidates] == [2, 4, 1, 5]
    mask[2] = False
    assert partner_candidates(geometry, 3)[0]["index"] == 4
    assert partner_candidates(geometry, 0) == []


def test_ten_features_and_signed_index_distance():
    chain = line_chain()
    geom = prepare_geometry(chain)
    expected = [1, 1, 1, 1, 1, 1, 1, 2, 2, math.log(3)]
    np.testing.assert_allclose(interaction_features(chain, geom, 2, 4)["values"], expected)
    reverse = interaction_features(chain, geom, 4, 2)["values"]
    np.testing.assert_allclose(reverse, [1, 1, -1, -1, 1, 1, 1, 2, -2, -math.log(3)])
    # PDB numbers differ by six, but sequence positions differ by two.
    assert chain.residues[4][0] - chain.residues[2][0] == 6


def test_missing_neighbor_blocks_features_and_raw_atoms_stay_missing():
    chain = line_chain()
    chain.xyz[5, 0] = np.nan
    chain.xyz[2, 3] = np.nan
    original = chain.xyz.copy()
    geometry = prepare_geometry(chain)
    assert geometry.valid[2]
    assert not geometry.valid[5]
    assert np.isfinite(geometry.cb_used[2]).all()
    np.testing.assert_array_equal(chain.xyz, original)
    with pytest.raises(ValueError, match="neighborhood"):
        interaction_features(chain, geometry, 2, 4)


def test_degenerate_geometry_is_explicitly_invalid():
    chain = line_chain()
    chain.xyz[2, 3] = chain.xyz[2, 1]
    geom = prepare_geometry(chain)
    assert not geom.valid[2]
    assert geom.reasons[2] == "degenerate_geometry"
