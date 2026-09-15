"""Day 05: bounded kerasify reader and float32 dense inference without torch."""

import hashlib
import json
import struct
from dataclasses import dataclass
from importlib.resources import files

import numpy as np


@dataclass(frozen=True)
class DenseLayer:
    weights: np.ndarray  # [input, output], stored row-major by kerasify
    bias: np.ndarray
    activation: int  # 1 linear, 2 ReLU


def parse_dense_model(blob: bytes) -> tuple[DenseLayer, ...]:
    offset = 0

    def take(count):
        nonlocal offset
        if count < 0 or offset + count > len(blob):
            raise ValueError("Truncated kerasify file")
        result = blob[offset : offset + count]
        offset += count
        return result

    def integer():
        return struct.unpack("<I", take(4))[0]

    count = integer()
    if not 1 <= count <= 16:
        raise ValueError("Unsupported layer count")
    layers = []
    for _ in range(count):
        kind, inputs, outputs, biases = [integer() for _ in range(4)]
        if kind != 1 or not 1 <= inputs <= 1024 or not 1 <= outputs <= 1024:
            raise ValueError("Unsupported dense layer or oversized dimensions")
        if biases != outputs or (layers and inputs != len(layers[-1].bias)):
            raise ValueError("Inconsistent dense dimensions")
        weights = np.frombuffer(take(4 * inputs * outputs), dtype="<f4").reshape(inputs, outputs)
        bias = np.frombuffer(take(4 * biases), dtype="<f4")
        activation = integer()
        if activation not in (1, 2):
            raise ValueError("Only linear and ReLU activations are supported")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError("Nonfinite network parameters")
        layers.append(DenseLayer(weights, bias, activation))
    if offset != len(blob):
        raise ValueError("Trailing bytes in kerasify file")
    return tuple(layers)


def forward(layers: tuple[DenseLayer, ...], features: np.ndarray) -> dict:
    if not layers:
        raise ValueError("Empty network")
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.asarray(features, dtype=np.float32)
    if x.shape != (layers[0].weights.shape[0],) or not np.isfinite(x).all():
        raise ValueError("Wrong feature shape or nonfinite float32 input")
    trace = []
    for layer in layers:
        values = np.zeros(len(layer.bias), dtype=np.float32)
        # Match Kerasify's accumulation order; do not change it to a BLAS reduction/FMA.
        with np.errstate(over="ignore", invalid="ignore"):
            for k in range(len(x)):
                values = values + x[k] * layer.weights[k]
            values = values + layer.bias
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite network activation")
        output = np.maximum(values, np.float32(0)) if layer.activation == 2 else values.copy()
        trace.append(
            {
                "input": x.tolist(),
                "pre_activation": values.tolist(),
                "activation": "relu" if layer.activation == 2 else "linear",
                "output": output.tolist(),
            }
        )
        x = output
    return {"embedding": x.astype(np.float64).tolist(), "layers": trace}


def official_model() -> tuple[tuple[DenseLayer, ...], dict]:
    data = files("mini3di_encoder").joinpath("data")
    spec = json.loads(data.joinpath("official.json").read_text())
    blob = data.joinpath("encoder.kerasify").read_bytes()
    if hashlib.sha256(blob).hexdigest() != spec["weights_sha256"]:
        raise ValueError("Official weights hash mismatch")
    layers = parse_dense_model(blob)
    if [(x.weights.shape, x.activation) for x in layers] != [
        ((10, 10), 2),
        ((10, 10), 2),
        ((10, 2), 1),
    ]:
        raise ValueError("Unexpected official architecture")
    return layers, spec


def nearest_state(embedding: list[float], spec: dict) -> dict:
    z = np.asarray(embedding, dtype=np.float64)
    centers = np.asarray(spec["centroids"], dtype=np.float64)
    if z.shape != (2,) or centers.shape != (20, 2) or not np.isfinite(z).all():
        raise ValueError("Invalid embedding or centers")
    distances = [(float(z[0] - c[0])) ** 2 + (float(z[1] - c[1])) ** 2 for c in centers]
    order = sorted(range(20), key=lambda index: (distances[index], index))
    state = order[0]
    return {
        "state": state,
        "letter": spec["alphabet"][state],
        "squared_distances": distances,
        "nearest_margin": distances[order[1]] - distances[state],
    }
