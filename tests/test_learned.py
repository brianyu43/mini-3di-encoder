import json
from pathlib import Path

import numpy as np
import pytest

from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.encode import encode_chain, search_record
from mini3di_encoder.learned import embed_batch, load_learned_model
from mini3di_encoder.network import forward, official_model


def constant_model():
    centers = [[100.0 + i, 100.0] for i in range(20)]
    centers[5] = [0.0, 0.0]
    return {
        "format": "mini3di-learned-v1",
        "alphabet": "ACDEFGHIKLMNPQRSTVWY",
        "layers": [
            {"weights": np.zeros((a, b)).tolist(), "bias": [0.0] * b, "activation": act}
            for a, b, act in [(10, 10, 2), (10, 10, 2), (10, 2, 1)]
        ],
        "centroids": centers,
        "provenance": {"geometry_upstream_commit": "synthetic-test-only"},
    }


def test_learned_runtime_changes_states_but_preserves_masks_and_residue_mapping(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(json.dumps(constant_model()))
    chain = read_backbone(Path(__file__).resolve().parents[1] / "examples/1UBQ.pdb", "A")
    official = encode_chain(chain, "test")
    learned = encode_chain(chain, "test", encoder_model=model)
    assert learned["encoder_kind"] == "learned-vqvae"
    assert learned["valid_seed_mask"] == official["valid_seed_mask"]
    assert learned["aa"] == official["aa"]
    assert learned["three_di"] == "X" + "G" * 74 + "X"
    assert len(search_record(learned)["three_di"]) == 76


def test_batched_numpy_has_same_accumulation_as_scalar_reference():
    layers, _ = official_model()
    features = np.random.default_rng(6).normal(size=(50, 10)).astype(np.float32)
    expected = np.array([forward(layers, x)["embedding"] for x in features], dtype=np.float32)
    np.testing.assert_array_equal(embed_batch(layers, features), expected)


@pytest.mark.parametrize("defect", ["shape", "centers", "alphabet"])
def test_malformed_models_are_rejected(tmp_path, defect):
    payload = constant_model()
    if defect == "shape":
        payload["layers"][0]["weights"] = [[0.0]]
    elif defect == "centers":
        payload["centroids"][0][0] = float("nan")
    else:
        payload["alphabet"] = "AB"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_learned_model(path)
