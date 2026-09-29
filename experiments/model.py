"""Small paired-feature VQ-VAE matching the pinned public training architecture."""

import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10), nn.BatchNorm1d(10), nn.ReLU(), nn.Linear(10, 10)
        )
        self.mu = nn.Linear(10, 10)
        self.logvar = nn.Linear(10, 10)

    def forward(self, z):
        hidden = self.layers(z)
        # Retain the pinned source's factor 0.5; GaussianNLLLoss receives this as variance.
        return self.mu(hidden), torch.exp(0.5 * self.logvar(hidden))


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 2)
        self.embedding.weight.data.uniform_(-1 / 20, 1 / 20)

    def forward(self, inputs):
        centers = self.embedding.weight
        distances = (
            inputs.square().sum(1, keepdim=True) + centers.square().sum(1) - 2 * inputs @ centers.T
        )
        indices = distances.argmin(1)
        quantized = self.embedding(indices)
        commitment = F.mse_loss(quantized.detach(), inputs)
        codebook = F.mse_loss(quantized, inputs.detach())
        loss = codebook + 0.25 * commitment
        straight_through = inputs + (quantized - inputs).detach()
        return loss, straight_through, indices


class PairedVQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )
        self.decoder = Decoder()
        self.vq = Quantizer()

    def forward(self, x):
        vq_loss, quantized, indices = self.vq(self.encoder(x))
        mean, variance = self.decoder(quantized)
        return vq_loss, mean, variance, indices

    def objective(self, x, y):
        vq, mean, variance, indices = self(x)
        nll = F.gaussian_nll_loss(mean, y, variance)
        return nll + vq, nll, vq, indices


def export_model(model, provenance):
    model.eval()
    layers = [
        nn.utils.fuse_linear_bn_eval(model.encoder[0], model.encoder[1]),
        nn.utils.fuse_linear_bn_eval(model.encoder[3], model.encoder[4]),
        model.encoder[6],
    ]
    return {
        "format": "mini3di-learned-v1",
        "alphabet": "ACDEFGHIKLMNPQRSTVWY",
        "layers": [
            {
                "weights": layer.weight.detach().T.tolist(),
                "bias": layer.bias.detach().tolist(),
                "activation": activation,
            }
            for layer, activation in zip(layers, [2, 2, 1], strict=True)
        ],
        "centroids": model.vq.embedding.weight.detach().tolist(),
        "provenance": provenance,
    }
