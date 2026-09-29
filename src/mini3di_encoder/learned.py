"""Load an exported learned encoder as bounded JSON; inference needs only NumPy."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .network import DenseLayer


def load_learned_model(path: Path):
    if path.stat().st_size > 200_000:
        raise ValueError("Oversized learned encoder")
    blob = path.read_bytes()
    data = json.loads(blob)
    if data.get("format") != "mini3di-learned-v1" or data.get("alphabet") != "ACDEFGHIKLMNPQRSTVWY":
        raise ValueError("Unsupported learned encoder format/alphabet")
    layers = []
    expected = [((10, 10), 2), ((10, 10), 2), ((10, 2), 1)]
    if len(data["layers"]) != len(expected):
        raise ValueError("Unexpected learned layer count")
    for row, (shape, activation) in zip(data["layers"], expected, strict=True):
        weights = np.asarray(row["weights"], dtype=np.float32)
        bias = np.asarray(row["bias"], dtype=np.float32)
        if weights.shape != shape or bias.shape != (shape[1],) or row["activation"] != activation:
            raise ValueError("Unexpected learned layer shape/activation")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError("Nonfinite learned parameters")
        layers.append(DenseLayer(weights, bias, activation))
    centers = np.asarray(data["centroids"], dtype=np.float64)
    if centers.shape != (20, 2) or not np.isfinite(centers).all():
        raise ValueError("Invalid learned centers")
    spec = {
        "alphabet": data["alphabet"],
        "centroids": centers.tolist(),
        "invalid_state": 2,
        "weights_sha256": hashlib.sha256(blob).hexdigest(),
        "upstream_commit": data["provenance"]["geometry_upstream_commit"],
        "model_kind": "learned-vqvae",
        "provenance": data["provenance"],
    }
    return tuple(layers), spec


def embed_batch(layers, features):
    """Same accumulation order as network.forward, batched across residues only."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 10 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite N x 10 array")
    for layer in layers:
        values = np.zeros((len(x), len(layer.bias)), dtype=np.float32)
        for k in range(x.shape[1]):
            values = values + x[:, k, None] * layer.weights[k]
        values = values + layer.bias
        x = np.maximum(values, np.float32(0)) if layer.activation == 2 else values
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite learned embedding")
    return x


def predict_states(layers, spec, features):
    z = embed_batch(layers, features).astype(np.float64)
    distances = np.sum((z[:, None] - np.asarray(spec["centroids"])[None]) ** 2, axis=-1)
    return np.argmin(distances, axis=1)
