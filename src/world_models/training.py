"""Training and evaluation routines shared by the independent CLI scripts."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import FrameDataset, SequenceDataset, episode_split
from .mdn_rnn import MDNRNN, mdn_nll, posterior_latents
from .utils import (fingerprint, finite_or_raise, load_checkpoint, save_checkpoint,
                    save_history, seed_all, select_device, write_json)
from .vae import ConvVAE, freeze_vae, vae_loss
from .visualization import image_grid, to_images


def load_vae(path, device, frozen=True):
    checkpoint = load_checkpoint(path)
    if checkpoint["kind"] != "vae":
        raise ValueError("Expected a VAE checkpoint")
    model = ConvVAE(checkpoint["config"]["model"]["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return (freeze_vae(model) if frozen else model), checkpoint


def load_mdn(path, device):
    checkpoint = load_checkpoint(path)
    if checkpoint["kind"] != "mdn":
        raise ValueError("Expected an MDN-RNN checkpoint")
    settings = checkpoint["config"]["model"]
    model = MDNRNN(**settings).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def make_loaders(config: dict, kind: str, vae_sha: str | None = None):
    root = Path(config["output_dir"])
    split = episode_split(root / "data", config["seed"], config["data"]["val_fraction"])
    loaders = {}
    for part in ("train", "val"):
        paths = [root / ("data" if kind == "vae" else "latents") / name for name in split[part]]
        dataset = FrameDataset(paths) if kind == "vae" else SequenceDataset(paths, config["training"]["seq_len"], vae_sha)
        if not len(dataset):
            raise ValueError(f"Empty {part} dataset")
        generator = torch.Generator().manual_seed(config["seed"])
        loaders[part] = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                                   shuffle=part == "train", num_workers=0, generator=generator)
    return loaders


def vae_pass(model, loader, device, beta, optimizer=None, grad_clip=5.0):
    model.train(optimizer is not None)
    totals = dict(total=0.0, reconstruction=0.0, kl=0.0)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for frames in tqdm(loader, desc="VAE train" if optimizer else "VAE validation", leave=False):
            frames = frames.to(device)
            if optimizer:
                reconstruction, mu, logvar = model(frames)
            else:
                mu, logvar = model.encode(frames)
                reconstruction = model.decode(mu)
            total, rec, kl = vae_loss(reconstruction, frames, mu, logvar, beta)
            finite_or_raise(total, "VAE loss")
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            n = len(frames)
            for key, value in zip(totals, (total, rec, kl)):
                totals[key] += value.item() * n
            count += n
    return {key: value / count for key, value in totals.items()}


def train_vae(config: dict):
    seed_all(config["seed"])
    device = select_device(config["device"])
    root = Path(config["output_dir"])
    write_json(root / "vae_config.json", config)
    loaders = make_loaders(config, "vae")
    model = ConvVAE(config["model"]["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    history, best, step = [], float("inf"), 0
    for epoch in range(1, config["training"]["vae_epochs"] + 1):
        train = vae_pass(model, loaders["train"], device, config["training"]["beta"], optimizer,
                         config["training"]["grad_clip"])
        val = vae_pass(model, loaders["val"], device, config["training"]["beta"])
        row = dict(epoch=epoch, **{f"train_{k}": v for k, v in train.items()}, **{f"val_{k}": v for k, v in val.items()})
        history.append(row)
        step += len(loaders["train"])
        print(row, flush=True)
        if val["total"] < best:
            best = val["total"]
            save_checkpoint(root / "checkpoints/vae_best.pt", model, optimizer, config, epoch, step, best, "vae")
        save_checkpoint(root / "checkpoints/vae_last.pt", model, optimizer, config, epoch, step, best, "vae")
        save_history(root / "metrics", history, "vae_training")
    return root / "checkpoints/vae_best.pt"


def evaluate_vae(checkpoint_path, device_name=None):
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    seed_all(config["seed"])
    device = select_device(device_name or config["device"])
    model, _ = load_vae(checkpoint_path, device)
    loader = make_loaders(config, "vae")["val"]
    metrics = vae_pass(model, loader, device, config["training"]["beta"])
    frames = next(iter(loader))[:8].to(device)
    with torch.no_grad():
        mu, _ = model.encode(frames)
        reconstruction = model.decode(mu)
    root = Path(config["output_dir"])
    image_grid([to_images(frames), to_images(reconstruction)], ["Original", "VAE decode(mu)"], root / "plots/vae_reconstruction.png")
    write_json(root / "metrics/vae_evaluation.json", dict(**metrics, reconstruction_mode="decode_posterior_mean"))
    print(metrics)
    return metrics


def mdn_pass(model, loader, device, latent_mode, optimizer=None, grad_clip=5.0):
    model.train(optimizer is not None)
    total, count = 0.0, 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for mu, logvar, actions, mask in tqdm(loader, desc="MDN train" if optimizer else "MDN validation", leave=False):
            mu, logvar, actions, mask = [x.to(device) for x in (mu, logvar, actions, mask)]
            z = posterior_latents(mu, logvar, latent_mode)
            params, _ = model(z[:, :-1], actions)  # Fresh hidden state for each independent window.
            loss = mdn_nll(params, z[:, 1:], mask)
            finite_or_raise(loss, "MDN NLL")
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            n = mask.sum().item()
            total += loss.item() * n
            count += n
    return total / count


def train_mdn(config, vae_checkpoint):
    seed_all(config["seed"])
    device = select_device(config["device"])
    vae, _ = load_vae(vae_checkpoint, device)
    if vae.latent_dim != config["model"]["latent_dim"]:
        raise ValueError("VAE and MDN latent dimensions differ")
    vae_sha = fingerprint(vae_checkpoint)
    loaders = make_loaders(config, "mdn", vae_sha)
    model = MDNRNN(**config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    root = Path(config["output_dir"])
    write_json(root / "mdn_config.json", config)
    history, best, step = [], float("inf"), 0
    for epoch in range(1, config["training"]["mdn_epochs"] + 1):
        train = mdn_pass(model, loaders["train"], device, config["training"]["latent_mode"], optimizer,
                         config["training"]["grad_clip"])
        # Fixed seed makes the sampled validation posterior repeatable each epoch.
        seed_all(config["seed"] + 10000)
        val = mdn_pass(model, loaders["val"], device, config["training"]["latent_mode"])
        seed_all(config["seed"] + epoch)
        row = dict(epoch=epoch, train_nll=train, val_nll=val)
        history.append(row)
        step += len(loaders["train"])
        print(row, flush=True)
        if val < best:
            best = val
            save_checkpoint(root / "checkpoints/mdn_best.pt", model, optimizer, config, epoch, step, best,
                            "mdn", vae_sha256=vae_sha)
        save_checkpoint(root / "checkpoints/mdn_last.pt", model, optimizer, config, epoch, step, best,
                        "mdn", vae_sha256=vae_sha)
        save_history(root / "metrics", history, "mdn_training")
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in vae.parameters())
    return root / "checkpoints/mdn_best.pt"


def evaluate_mdn(checkpoint_path, device_name=None):
    import math
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    seed_all(config["seed"] + 10000)
    device = select_device(device_name or config["device"])
    model, _ = load_mdn(checkpoint_path, device)
    loader = make_loaders(config, "mdn", checkpoint["vae_sha256"])["val"]
    nll = mdn_pass(model, loader, device, config["training"]["latent_mode"])
    seed_all(config["seed"] + 10000)
    baseline_total, count = 0.0, 0.0
    with torch.no_grad():
        for mu, logvar, _, mask in loader:
            mu, logvar, mask = [x.to(device) for x in (mu, logvar, mask)]
            z = posterior_latents(mu, logvar, config["training"]["latent_mode"])
            # Fixed unit-variance Gaussian centered at z_t (not fitted on validation).
            loss = (0.5 * (z[:, 1:] - z[:, :-1]).square() + 0.5 * math.log(2 * math.pi)).sum(-1)
            baseline_total += (loss * mask).sum().item()
            count += mask.sum().item()
    metrics = dict(one_step_nll=nll, stationary_unit_gaussian_nll=baseline_total / count,
                   units="nats/transition, summed over latent dimensions", latent_mode=config["training"]["latent_mode"],
                   context="reset per independent sequence window")
    write_json(Path(config["output_dir"]) / "metrics/mdn_evaluation.json", metrics)
    print(metrics)
    return metrics
