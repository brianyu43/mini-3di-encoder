import struct

import numpy as np
import pytest

from mini3di_encoder.network import forward, nearest_state, official_model, parse_dense_model


def tiny_blob(activation=1):
    # Input [2,3] @ [[1,2],[3,4]] + [5,6] = [16,22].
    return struct.pack("<5I", 1, 1, 2, 2, 2) + struct.pack("<6fI", 1, 2, 3, 4, 5, 6, activation)


def test_weight_orientation_bias_and_relu():
    assert forward(parse_dense_model(tiny_blob()), np.array([2, 3]))["embedding"] == [16, 22]
    trace = forward(parse_dense_model(tiny_blob(2)), np.array([-2, -3]))
    assert trace["layers"][0]["pre_activation"] == [-6, -10]
    assert trace["embedding"] == [0, 0]


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        tiny_blob()[:-1],
        tiny_blob() + b"x",
        tiny_blob(9),
        struct.pack("<I", 0),
        struct.pack("<5I", 1, 1, 999999, 2, 2),
    ],
)
def test_invalid_model_bytes_fail_closed(blob):
    with pytest.raises(ValueError):
        parse_dense_model(blob)


@pytest.mark.parametrize("values", [[1], [1, float("nan")], [1e300, 2]])
def test_wrong_or_nonfinite_network_input(values):
    with pytest.raises(ValueError):
        forward(parse_dense_model(tiny_blob()), np.array(values))


def test_official_shape_and_centroid_ties():
    model, spec = official_model()
    assert [x.weights.shape for x in model] == [(10, 10), (10, 10), (10, 2)]
    assert sum(x.weights.size + x.bias.size for x in model) == 242
    assert nearest_state(spec["centroids"][2], spec)["letter"] == "D"
    tied = {**spec, "centroids": [[0, 0], [0, 0]] + [[100, 100]] * 18}
    assert nearest_state([0, 0], tied)["state"] == 0
    assert nearest_state([0, 0], tied)["nearest_margin"] == 0
