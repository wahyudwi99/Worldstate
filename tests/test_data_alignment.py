"""Small synthetic tests: no simulator and no model training."""
import numpy as np
import pytest

from world_models.dataset import SequenceDataset, episode_split, load_episode, validate_episode
from world_models.envs import RandomPolicy, frames_to_tensor, preprocess
from world_models.utils import fingerprint, write_json


def episode(n=3):
    return dict(observations=np.stack([np.full((64, 64, 3), i, np.uint8) for i in range(n + 1)]),
                actions=np.zeros((n, 3), np.float32), rewards=np.zeros(n, np.float32),
                terminated=np.zeros(n, np.bool_), truncated=np.array([False] * (n - 1) + [True]))


def test_storage_alignment(tmp_path):
    data = episode()
    path = tmp_path / "episode_00000.npz"
    np.savez_compressed(path, **data)
    loaded = load_episode(path)
    assert validate_episode(loaded) == 3
    assert np.all(loaded["observations"][1:] - loaded["observations"][:-1] == 1)
    data["observations"] = data["observations"][:-1]
    with pytest.raises(ValueError, match="T\\+1"):
        validate_episode(data)


def test_split_by_episode(tmp_path):
    for index in range(5):
        np.savez_compressed(tmp_path / f"episode_{index:05d}.npz", **episode())
    split = episode_split(tmp_path, 42, 0.2)
    assert set(split["train"]).isdisjoint(split["val"])
    assert split == episode_split(tmp_path, 42, 0.2)


def test_windows_and_masks(tmp_path):
    paths = []
    for i in range(2):
        source = tmp_path / f"source_{i}.npz"
        np.savez_compressed(source, **episode())
        path = tmp_path / f"latent_{i}.npz"
        mu = np.arange(4, dtype=np.float32)[:, None] + i * 100
        np.savez_compressed(path, mu=mu, logvar=np.zeros_like(mu), actions=np.zeros((3, 3), np.float32))
        write_json(path.with_suffix(".json"), dict(vae_sha256="test", source_path=str(source), source_sha256=fingerprint(source)))
        paths.append(path)
    dataset = SequenceDataset(paths, 2, "test")
    assert len(dataset) == 4
    mu, _, _, mask = dataset[1]
    assert mu[:, 0].tolist() == [2, 3, 0]
    assert mask.tolist() == [1, 0]
    assert dataset[2][0][0, 0] == 100
    with pytest.raises(ValueError, match="Stale"):
        SequenceDataset(paths, 2, "wrong-checkpoint")


def test_preprocessing_and_policy():
    image = preprocess(np.full((96, 96, 3), 255, dtype=np.uint8))
    tensor = frames_to_tensor(image)
    assert tensor.shape == (3, 64, 64)
    assert tensor.min() == tensor.max() == 1
    policy = RandomPolicy(42, True, 5)
    first = policy(0)
    np.testing.assert_array_equal(first, policy(1))
    assert -1 <= first[0] <= 1 and np.all((first[1:] >= 0) & (first[1:] <= 1))


def test_collector_keeps_actual_transition_order(tmp_path, monkeypatch):
    """Use a fake environment; no Box2D or simulator launch is required."""
    import world_models.commands as commands

    class TinyEnvironment:
        def reset(self, seed):
            self.step_index = 0
            return np.zeros((96, 96, 3), np.uint8), {}

        def step(self, action):
            self.step_index += 1
            return np.full((96, 96, 3), self.step_index, np.uint8), 1.0, False, False, {}

        def close(self):
            pass

    monkeypatch.setattr(commands, "make_env", TinyEnvironment)
    config = dict(seed=42, output_dir=str(tmp_path), data=dict(episodes=2, max_steps=3,
                  policy="random", action_repeat=1, val_fraction=0.5))
    commands.collect(config)
    data = load_episode(tmp_path / "data/episode_00000.npz")
    assert data["observations"][:, 0, 0, 0].tolist() == [0, 1, 2, 3]
    assert data["truncated"].tolist() == [False, False, True]
    assert not data["terminated"].any()
