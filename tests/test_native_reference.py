"""Regression against stored outputs of unmodified native sources, not fresh C++ runs.

The run_day02_05 script rebuilds and refreshes native comparisons in a NEW directory.
These tests use frozen results and cannot silently regenerate their own expected values.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.network import forward, official_model
from mini3di_encoder.trace import trace_residue

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/day02-05"


@pytest.mark.parametrize("index", [1, 9, 22, 67])
def test_selected_residue_against_native_reference(index):
    native = {row["index"]: row for row in json.loads((RESULT / "native_trace.json").read_text())}
    original = native[index]
    trace = trace_residue(read_backbone(ROOT / "data/raw/1UBQ.pdb", "A"), index)
    assert trace["valid"]
    assert trace["partner"]["sequence_index_0"] == original["partner"]
    assert trace["state"] == original["state"]
    for key in ["virtual_center", "partner_virtual_center", "cb_used"]:
        np.testing.assert_allclose(trace[key], original[key], atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        [f["value"] for f in trace["features"]], original["features"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        trace["network"]["embedding"], original["embedding"], atol=1e-6, rtol=0
    )


def test_network_against_native_zero_basis_and_random_probes():
    probes = json.loads((RESULT / "network_probes.json").read_text())
    model, _ = official_model()
    predictions = [forward(model, row)["embedding"] for row in probes["inputs"]]
    assert len(predictions) == 44
    np.testing.assert_allclose(predictions, probes["native_outputs"], atol=1e-6, rtol=0)


@pytest.mark.parametrize("index", [0, 75])
def test_raw_endpoint_d_is_not_a_valid_encoded_residue(index):
    trace = trace_residue(read_backbone(ROOT / "data/raw/1UBQ.pdb", "A"), index)
    assert trace["letter"] == "D" and trace["state"] == 2
    assert not trace["valid"]
    assert trace["invalid_reason"] == "chain_endpoint"
    assert trace["features"] is None and trace["network"] is None
