"""One shared image transformation and the real CarRacing environment."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

ACTION_LOW = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.ones(3, dtype=np.float32)
RESIZE_CONVENTION = "RGB uint8; PIL bilinear full-frame resize to 64x64; no crop"


def preprocess(observation: np.ndarray) -> np.ndarray:
    """Return a 64x64x3 RGB uint8 image, preserving the full field of view."""
    array = np.asarray(observation)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Expected an HWC RGB uint8 observation.")
    return np.asarray(Image.fromarray(array).resize((64, 64), Image.Resampling.BILINEAR)).copy()


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert [..., H, W, C] uint8 to [..., C, H, W] float32 in [0,1]."""
    if frames.dtype != np.uint8 or frames.shape[-3:] != (64, 64, 3):
        raise ValueError("Expected preprocessed RGB uint8 frames with shape [...,64,64,3].")
    return torch.from_numpy(np.moveaxis(frames.copy(), -1, -3)).float().div_(255)


def make_env():
    import gymnasium as gym
    try:
        return gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    except gym.error.DependencyNotInstalled as exc:
        raise RuntimeError('CarRacing needs Box2D. Install swig, then pip install "gymnasium[box2d]".') from exc


class RandomPolicy:
    """Independent or piecewise-constant continuous random exploration."""
    def __init__(self, seed: int, correlated: bool, repeat: int = 5):
        if repeat < 1:
            raise ValueError("action_repeat must be positive")
        self.rng = np.random.default_rng(seed)
        self.correlated, self.repeat = correlated, repeat
        self.action = None

    def __call__(self, step: int) -> np.ndarray:
        if self.action is None or not self.correlated or step % self.repeat == 0:
            if self.correlated:
                # Forward-biased exploration; never press gas and brake together.
                self.action = np.array([self.rng.uniform(-1, 1), self.rng.uniform(0.2, 1), 0], dtype=np.float32)
                if self.rng.random() < 0.1:
                    self.action[1:] = [0, self.rng.uniform(0, 0.3)]
            else:
                self.action = self.rng.uniform(ACTION_LOW, ACTION_HIGH).astype(np.float32)
        return self.action.copy()
