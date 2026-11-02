import math

import torch

from world_models.mdn_rnn import MDNRNN, log_prob, mdn_nll, sample_latent
from world_models.vae import ConvVAE, freeze_vae


def test_nll_against_reference():
    torch.manual_seed(7)
    target = torch.randn(2, 3, 4)
    logits, means, log_stds = [torch.randn(2, 3, 4, 2) for _ in range(3)]
    params = logits, means, log_stds
    weights = torch.softmax(logits, -1)
    component = torch.distributions.Normal(means, log_stds.exp()).log_prob(target[..., None]).exp()
    reference = ((weights * component).sum(-1)).log().sum(-1)
    torch.testing.assert_close(log_prob(params, target), reference)
    zeros = torch.zeros(2, 3, 4, 1)
    expected = 0.5 * (target.square() + math.log(2 * math.pi)).sum(-1)
    torch.testing.assert_close(-log_prob((zeros, zeros, zeros), target), expected)


def test_shapes_gradients_sampling_and_mask():
    model = MDNRNN(4, 8, 2)
    params, _ = model(torch.randn(2, 3, 4), torch.rand(2, 3, 3))
    assert all(p.shape == (2, 3, 4, 2) for p in params)
    target = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    loss = mdn_nll(params, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sample_latent(params).shape == target.shape
    assert sample_latent(params, deterministic=True).shape == target.shape
    changed = target.clone()
    changed[mask == 0] += 100
    torch.testing.assert_close(loss, mdn_nll(params, changed, mask))


def test_frozen_vae_is_not_updated():
    vae = freeze_vae(ConvVAE(4))
    before = {key: value.clone() for key, value in vae.state_dict().items()}
    model = MDNRNN(4, 8, 2)
    optimizer = torch.optim.Adam(model.parameters())
    with torch.no_grad():
        mu, _ = vae.encode(torch.rand(2, 3, 64, 64))
    params, _ = model(mu[:, None], torch.zeros(2, 1, 3))
    loss = mdn_nll(params, mu[:, None])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert all(not p.requires_grad and p.grad is None for p in vae.parameters())
    assert all(torch.equal(before[key], value) for key, value in vae.state_dict().items())
