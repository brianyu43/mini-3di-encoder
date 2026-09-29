"""Independent pinned-source forward/gradient checks for the optional trainer."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.model import PairedVQVAE, export_model  # noqa: E402
from mini3di_encoder.learned import embed_batch, load_learned_model  # noqa: E402

REFERENCE = Path(__file__).resolve().parents[2] / (
    "mini-3di-encoder-archive/session-01-05-60de6e8/references/upstream/"
    "foldseek-analysis/training/train_vqvae.py"
)


def test_forward_and_parameter_gradients_match_pinned_training_source():
    if not REFERENCE.exists():
        pytest.skip("Pinned training source not installed on this machine")
    spec = importlib.util.spec_from_file_location("pinned_training", REFERENCE)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.manual_seed(123)
    own = PairedVQVAE()
    ref = upstream.create_vqvae(123, 10, 10, 2, 20)
    ref.load_state_dict(own.state_dict())
    rng = torch.Generator().manual_seed(91)
    x = torch.randn(64, 10, generator=rng)
    y = torch.randn(64, 10, generator=rng) + 1.0
    total, nll, vq, indices = own.objective(x, y)
    ref_vq, (mu, variance), _, encodings = ref(x)
    ref_nll = torch.nn.GaussianNLLLoss()(y, mu, variance)
    torch.testing.assert_close(vq, ref_vq)
    torch.testing.assert_close(nll, ref_nll)
    assert torch.equal(indices, encodings.argmax(1))
    total.backward()
    (ref_vq + ref_nll).backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        own.named_parameters(), ref.named_parameters(), strict=True
    ):
        assert name == ref_name
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=1e-6, rtol=1e-5)


def test_fused_json_inference_matches_unfused_encoder(tmp_path):
    import json

    torch.manual_seed(7)
    model = PairedVQVAE()
    # Update BatchNorm running statistics so this tests nontrivial fusion.
    for _ in range(3):
        model.encoder(torch.randn(32, 10) * 2 + 3)
    model.eval()
    path = tmp_path / "encoder.json"
    path.write_text(json.dumps(export_model(model, {"geometry_upstream_commit": "test"})))
    layers, _ = load_learned_model(path)
    x = np.random.default_rng(5).normal(size=(100, 10)).astype(np.float32)
    with torch.no_grad():
        expected = model.encoder(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(embed_batch(layers, x), expected, atol=2e-6, rtol=1e-5)


def test_codebook_and_encoder_receive_gradients():
    torch.manual_seed(9)
    model = PairedVQVAE()
    loss, *_ = model.objective(torch.randn(64, 10), torch.randn(64, 10))
    loss.backward()
    assert model.vq.embedding.weight.grad.abs().sum() > 0
    assert model.encoder[0].weight.grad.abs().sum() > 0
