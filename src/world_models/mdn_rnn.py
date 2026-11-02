"""LSTM with per-dimension independent Gaussian mixtures (no shared index)."""
from __future__ import annotations

import math
import torch
from torch import nn


class MDNRNN(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256, num_mixtures: int = 5):
        super().__init__()
        self.latent_dim, self.hidden_dim, self.num_mixtures = latent_dim, hidden_dim, num_mixtures
        self.rnn = nn.LSTM(latent_dim + 3, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, latent_dim * num_mixtures * 3)

    def forward(self, z: torch.Tensor, actions: torch.Tensor, state=None):
        # z [B,T,D], actions [B,T,3]; output parameters each [B,T,D,K].
        hidden, state = self.rnn(torch.cat((z, actions), dim=-1), state)
        values = self.head(hidden).reshape(*hidden.shape[:2], self.latent_dim, self.num_mixtures, 3)
        logits, means, log_stds = values.unbind(-1)
        return (logits, means, log_stds.clamp(-7, 5)), state


def log_prob(params, target: torch.Tensor) -> torch.Tensor:
    """Sum independent dimension log probabilities; output [B,T]."""
    logits, means, log_stds = params
    log_stds = log_stds.clamp(-7, 5)
    component = -0.5 * ((target.unsqueeze(-1) - means) * torch.exp(-log_stds)).square()
    component = component - log_stds - 0.5 * math.log(2 * math.pi)
    return torch.logsumexp(torch.log_softmax(logits, dim=-1) + component, dim=-1).sum(-1)


def mdn_nll(params, target: torch.Tensor, mask: torch.Tensor | None = None):
    """Mean NLL per valid transition, summed over D (nats/transition)."""
    losses = -log_prob(params, target)
    if mask is None:
        return losses.mean()
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def sample_latent(params, deterministic: bool = False) -> torch.Tensor:
    logits, means, log_stds = params
    weights = torch.softmax(logits, dim=-1)
    if deterministic:
        # Mixture expectation is a diagnostic, not a stochastic sample.
        return (weights * means).sum(-1)
    # Independently draw a component for each latent dimension. Sampling on CPU
    # avoids backend-specific multinomial limitations; results return to device.
    indices = torch.multinomial(weights.detach().cpu().reshape(-1, weights.shape[-1]), 1)
    indices = indices.reshape(*weights.shape[:-1], 1).to(means.device)
    selected_mean = means.gather(-1, indices).squeeze(-1)
    selected_std = log_stds.clamp(-7, 5).gather(-1, indices).squeeze(-1).exp()
    return selected_mean + selected_std * torch.randn_like(selected_mean)


def posterior_latents(mu, logvar, mode: str):
    if mode == "mu":
        return mu
    if mode != "sample":
        raise ValueError("latent mode must be mu or sample")
    return mu + (0.5 * logvar).exp() * torch.randn_like(mu)
