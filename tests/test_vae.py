import torch

from world_models.vae import ConvVAE, vae_loss


def test_shapes_loss_and_gradients():
    torch.manual_seed(42)
    model = ConvVAE(32)
    x = torch.rand(1, 3, 64, 64)
    reconstruction, mu, logvar = model(x)
    assert mu.shape == logvar.shape == (1, 32)
    assert reconstruction.shape == x.shape
    losses = vae_loss(reconstruction, x, mu, logvar)
    assert all(torch.isfinite(loss) for loss in losses)
    losses[0].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpoint_reload(tmp_path):
    from world_models.utils import load_checkpoint, save_checkpoint
    model = ConvVAE(2)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "vae.pt"
    save_checkpoint(path, model, optimizer, {"seed": 42}, 1, 2, 1.0, "vae")
    loaded = load_checkpoint(path)
    restored = ConvVAE(2)
    restored.load_state_dict(loaded["model"])
    assert loaded["step"] == 2
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
