"""Four-stage convolutional variational autoencoder."""
from __future__ import annotations

import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        # [B,3,64,64] -> 32x32 -> 16x16 -> 8x8 -> [B,256,4,4].
        layers = []
        channels = [3, 32, 64, 128, 256]
        for a, b in zip(channels[:-1], channels[1:]):
            layers.extend([nn.Conv2d(a, b, 4, 2, 1), nn.ReLU()])
        self.encoder = nn.Sequential(*layers, nn.Flatten())
        self.mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.project = nn.Linear(latent_dim, 256 * 4 * 4)
        # [B,256,4,4] -> 8x8 -> 16x16 -> 32x32 -> [B,3,64,64].
        layers = []
        channels = [256, 128, 64, 32, 3]
        for index, (a, b) in enumerate(zip(channels[:-1], channels[1:])):
            layers.extend([nn.ConvTranspose2d(a, b, 4, 2, 1), nn.Sigmoid() if index == 3 else nn.ReLU()])
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.mu(encoded), self.logvar(encoded).clamp(-20, 10)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.project(z).reshape(-1, 256, 4, 4))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


def vae_loss(reconstruction, target, mu, logvar, beta: float = 1.0):
    """Sum squared error over pixels; sum KL over latent dims; mean over B."""
    reconstruction_loss = (reconstruction - target).square().flatten(1).sum(1).mean()
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
    return reconstruction_loss + beta * kl, reconstruction_loss, kl


def freeze_vae(model: ConvVAE) -> ConvVAE:
    model.eval()
    model.requires_grad_(False)
    return model
