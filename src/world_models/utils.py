"""Configuration, deterministic seeds, device selection, and experiment artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device cpu or install a compatible PyTorch/macOS.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu.")
    return torch.device(name)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    for key in ("seed", "device", "output_dir", "data", "model", "training", "rollout"):
        if key not in config:
            raise ValueError(f"Missing configuration key: {key}")
    t, d, m = config["training"], config["data"], config["model"]
    for key in ("batch_size", "vae_epochs", "mdn_epochs", "seq_len"):
        if t[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if d["episodes"] < 2 or d["max_steps"] < 1 or not 0 < d["val_fraction"] < 1:
        raise ValueError("Use at least two episodes, positive max_steps, and 0 < val_fraction < 1.")
    if any(m[k] < 1 for k in ("latent_dim", "hidden_dim", "num_mixtures")):
        raise ValueError("Model dimensions must be positive.")
    if t["latent_mode"] not in ("sample", "mu"):
        raise ValueError("latent_mode must be sample or mu")
    return config


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    config: dict, epoch: int, step: int, best: float, kind: str,
                    **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                   epoch=epoch, step=step, best_metric=best, seed=config["seed"], kind=kind, **extra)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> dict:
    # Only load checkpoints produced by this project and trusted by the caller.
    return torch.load(path, map_location="cpu", weights_only=True)


def save_history(directory: Path, rows: list[dict], name: str) -> None:
    from .visualization import plot_history
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"{name}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / f"{name}.json", rows)
    plot_history(rows, directory / f"{name}.png")


def finite_or_raise(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}; stop and inspect the run.")
