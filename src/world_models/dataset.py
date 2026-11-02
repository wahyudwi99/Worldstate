"""Episode storage, persistent episode splits, and bounded-memory datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .envs import ACTION_HIGH, ACTION_LOW, frames_to_tensor
from .utils import fingerprint, read_json, write_json


def validate_episode(episode: dict) -> int:
    required = {"observations", "actions", "rewards", "terminated", "truncated"}
    if not required <= episode.keys():
        raise ValueError(f"Missing episode arrays: {required - episode.keys()}")
    obs, actions = episode["observations"], episode["actions"]
    n = len(actions)
    if n < 1 or obs.shape != (n + 1, 64, 64, 3) or obs.dtype != np.uint8:
        raise ValueError("observations must be uint8 [T+1,64,64,3], T >= 1")
    if actions.shape != (n, 3) or actions.dtype != np.float32:
        raise ValueError("actions must be float32 [T,3]")
    if not np.isfinite(actions).all() or np.any(actions < ACTION_LOW) or np.any(actions > ACTION_HIGH):
        raise ValueError("Invalid action values or bounds")
    for key, dtype in (("rewards", np.float32), ("terminated", np.bool_), ("truncated", np.bool_)):
        if episode[key].shape != (n,) or episode[key].dtype != dtype:
            raise ValueError(f"Invalid {key} shape or dtype")
    if not np.isfinite(episode["rewards"]).all():
        raise ValueError("Rewards must be finite")
    if np.any(episode["terminated"][:-1] | episode["truncated"][:-1]):
        raise ValueError("An episode contains transitions after termination/truncation")
    return n


def load_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        episode = {key: archive[key] for key in archive.files}
    validate_episode(episode)
    return episode


def episode_split(data_dir: Path, seed: int, val_fraction: float) -> dict:
    files = sorted(p.name for p in data_dir.glob("episode_*.npz"))
    if len(files) < 2:
        raise ValueError("At least two episodes are required for an episode-level train/validation split.")
    split_path = data_dir / "split.json"
    if split_path.exists():
        split = read_json(split_path)
        if (sorted(split["train"] + split["val"]) != files or split["seed"] != seed
                or split["val_fraction"] != val_fraction):
            raise ValueError("Dataset/split configuration changed. Use a new output directory.")
        return split
    shuffled = np.random.default_rng(seed).permutation(files).tolist()
    count = min(len(files) - 1, max(1, round(len(files) * val_fraction)))
    split = dict(train=sorted(shuffled[count:]), val=sorted(shuffled[:count]),
                 seed=seed, val_fraction=val_fraction)
    write_json(split_path, split)
    return split


class FrameDataset(Dataset):
    """Keep at most one decompressed episode in this worker's memory."""
    def __init__(self, paths: list[Path]):
        self.paths = paths
        lengths = []
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                lengths.append(len(archive["observations"]))
        self.ends = np.cumsum(lengths)
        self._index, self._frames = -1, None

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index: int):
        ep = int(np.searchsorted(self.ends, index, side="right"))
        offset = index - (int(self.ends[ep - 1]) if ep else 0)
        if ep != self._index:
            self._frames = load_episode(self.paths[ep])["observations"]
            self._index = ep
        return frames_to_tensor(self._frames[offset])


class SequenceDataset(Dataset):
    """Non-overlapping windows confined to one episode, padded and masked.

    Return full [L+1,D] posterior sequences. Sample once before shifting so
    the same adjacent frame has the same sampled latent inside a window.
    Each window begins with a reset LSTM state (truncated context).
    """
    def __init__(self, paths: list[Path], seq_len: int, vae_sha256: str):
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.paths, self.seq_len = paths, seq_len
        self.windows = []
        self._index, self._cache = -1, None
        for index, path in enumerate(paths):
            metadata = read_json(path.with_suffix(".json"))
            if metadata["vae_sha256"] != vae_sha256:
                raise ValueError(f"Stale latent cache: {path}")
            source = Path(metadata["source_path"])
            if not source.exists() or fingerprint(source) != metadata["source_sha256"]:
                raise ValueError(f"Source episode changed or moved; regenerate cache: {path}")
            with np.load(path, allow_pickle=False) as archive:
                n = len(archive["actions"])
                if (archive["mu"].shape != archive["logvar"].shape or len(archive["mu"]) != n + 1
                        or archive["actions"].shape != (n, 3)):
                    raise ValueError(f"Invalid latent cache alignment: {path}")
            self.windows.extend((index, start, min(seq_len, n - start)) for start in range(0, n, seq_len))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        ep, start, length = self.windows[index]
        if ep != self._index:
            with np.load(self.paths[ep], allow_pickle=False) as archive:
                self._cache = {key: archive[key] for key in archive.files}
            self._index = ep
        d = self._cache["mu"].shape[-1]
        mu = np.zeros((self.seq_len + 1, d), np.float32)
        logvar = np.zeros_like(mu)
        actions = np.zeros((self.seq_len, 3), np.float32)
        mask = np.zeros(self.seq_len, np.float32)
        mu[:length + 1] = self._cache["mu"][start:start + length + 1]
        logvar[:length + 1] = self._cache["logvar"][start:start + length + 1]
        actions[:length] = self._cache["actions"][start:start + length]
        mask[:length] = 1
        return tuple(torch.from_numpy(x) for x in (mu, logvar, actions, mask))
