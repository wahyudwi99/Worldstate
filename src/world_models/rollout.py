"""Free-running dynamics; this function has no future-observation argument."""
from __future__ import annotations

import torch

from .mdn_rnn import sample_latent


@torch.no_grad()
def warm_state(model, history_z: torch.Tensor, history_actions: torch.Tensor):
    """Consume only transitions preceding the start state."""
    if history_z.shape[1] == 0:
        return None
    _, state = model(history_z, history_actions)
    return state


@torch.no_grad()
def imagine(model, initial_z: torch.Tensor, actions: torch.Tensor, state=None,
            deterministic: bool = False) -> torch.Tensor:
    """initial_z [B,D], recorded actions [B,H,3] -> generated [B,H,D]."""
    z = initial_z
    generated = []
    for index in range(actions.shape[1]):
        params, state = model(z[:, None], actions[:, index:index + 1], state)
        z = sample_latent(params, deterministic=deterministic)[:, 0]
        generated.append(z)
    if not generated:
        return initial_z.new_empty((len(initial_z), 0, initial_z.shape[-1]))
    return torch.stack(generated, dim=1)
