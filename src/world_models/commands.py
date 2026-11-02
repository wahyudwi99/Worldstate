"""CLI implementations for collection, encoding, inspection, and rollout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .dataset import episode_split, load_episode
from .envs import RESIZE_CONVENTION, RandomPolicy, frames_to_tensor, make_env, preprocess
from .mdn_rnn import log_prob, posterior_latents
from .rollout import imagine, warm_state
from .training import (evaluate_mdn, evaluate_vae, load_mdn, load_vae,
                       train_mdn, train_vae)
from .utils import (fingerprint, finite_or_raise, load_config, read_json, seed_all,
                    select_device, write_json)
from .visualization import comparison_gif, gif_to_video, image_grid, to_images


def parser(description: str, config: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    if config:
        p.add_argument("--config", default="configs/smoke.yaml", help="YAML configuration path")
    return p


def add_device(p):
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)


def configured(args):
    config = load_config(args.config)
    if getattr(args, "device", None):
        config["device"] = args.device
    return config


def collect(config):
    root = Path(config["output_dir"])
    directory = root / "data"
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("episode_*.npz")):
        raise ValueError("Episode files already exist. Use a new output_dir; collection never overwrites data.")
    seed_all(config["seed"])
    write_json(root / "collection_config.json", config)
    dependencies = {}
    for name in ("gymnasium", "numpy", "Pillow", "torch", "box2d-py", "pygame"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "not installed"
    settings = config["data"]
    env = make_env()
    rows = []
    try:
        for index in tqdm(range(settings["episodes"]), desc="Collect episodes"):
            seed = config["seed"] + index
            observation, _ = env.reset(seed=seed)
            policy = RandomPolicy(seed, settings["policy"] == "correlated", settings["action_repeat"])
            observations = [preprocess(observation)]
            actions, rewards, terminated_flags, truncated_flags = [], [], [], []
            collector_limit = False
            for step in range(settings["max_steps"]):
                action = policy(step)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                collector_limit = step + 1 == settings["max_steps"] and not (terminated or truncated)
                # The collector's own time limit is an external truncation, not termination.
                truncated = bool(truncated or collector_limit)
                observations.append(preprocess(next_observation))
                actions.append(action)
                rewards.append(reward)
                terminated_flags.append(terminated)
                truncated_flags.append(truncated)
                if terminated or truncated:
                    break
            name = f"episode_{index:05d}"
            episode = dict(observations=np.asarray(observations, dtype=np.uint8),
                           actions=np.asarray(actions, dtype=np.float32), rewards=np.asarray(rewards, dtype=np.float32),
                           terminated=np.asarray(terminated_flags, dtype=np.bool_),
                           truncated=np.asarray(truncated_flags, dtype=np.bool_))
            from .dataset import validate_episode
            validate_episode(episode)
            np.savez_compressed(directory / f"{name}.npz", **episode)
            metadata = dict(seed=seed, env_id="CarRacing-v3", policy=settings["policy"],
                            action_repeat=settings["action_repeat"], resize_convention=RESIZE_CONVENTION,
                            episode_steps=len(actions), collector_time_limit=collector_limit,
                            collected_at=datetime.now(timezone.utc).isoformat(), dependencies=dependencies)
            write_json(directory / f"{name}.json", metadata)
            rows.append(dict(episode=name, steps=len(actions), return_=float(sum(rewards))))
    finally:
        env.close()
    write_json(root / "metrics/collection.json", rows)
    episode_split(directory, config["seed"], settings["val_fraction"])
    image_grid([episode["observations"]], ["Collected frames"], root / "plots/collection.png")
    print(f"Collected {len(rows)} episodes in {directory}")


def collect_main():
    p = parser("Collect aligned real CarRacing episodes.")
    p.add_argument("--episodes", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--policy", choices=["random", "correlated"])
    p.add_argument("--action-repeat", type=int)
    p.add_argument("--output-dir")
    args = p.parse_args()
    config = configured(args)
    for key in ("episodes", "max_steps", "policy", "action_repeat"):
        value = getattr(args, key)
        if value is not None:
            config["data"][key] = value
    for key in ("seed", "output_dir"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["data"]["episodes"] < 2 or config["data"]["max_steps"] < 1:
        p.error("Use at least two episodes and a positive max-steps.")
    collect(config)


def inspect_main():
    p = parser("Validate all episode arrays and export a frame montage.")
    p.add_argument("--data-dir", type=Path)
    args = p.parse_args()
    config = configured(args)
    directory = args.data_dir or Path(config["output_dir"]) / "data"
    paths = sorted(directory.glob("episode_*.npz"))
    if not paths:
        p.error(f"No episodes in {directory}")
    rows = []
    for path in paths:
        episode = load_episode(path)
        rows.append(dict(episode=path.name, steps=len(episode["actions"]),
                         min_pixel=int(episode["observations"].min()), max_pixel=int(episode["observations"].max()),
                         total_reward=float(episode["rewards"].sum())))
    root = Path(config["output_dir"])
    image_grid([load_episode(paths[0])["observations"]], ["Real episode"], root / "plots/dataset_montage.png")
    write_json(root / "metrics/dataset_inspection.json", rows)
    print(f"Validated {len(paths)} episodes, {sum(r['steps'] for r in rows)} transitions. T+1/T alignment holds.")
    print("Array validation cannot prove semantic alignment; collector logic and tests cover transition order.")


def encode_dataset(config, vae_checkpoint, device_name=None):
    seed_all(config["seed"])
    device = select_device(device_name or config["device"])
    vae, _ = load_vae(vae_checkpoint, device)
    root = Path(config["output_dir"])
    destination = root / "latents"
    destination.mkdir(parents=True, exist_ok=True)
    paths = sorted((root / "data").glob("episode_*.npz"))
    if not paths:
        raise ValueError("No source episodes. Run collect.py first.")
    sha = fingerprint(vae_checkpoint)
    write_json(destination / "config.json", dict(config=config, vae_sha256=sha))
    with torch.no_grad():
        for path in tqdm(paths, desc="Encode episodes"):
            episode = load_episode(path)
            mus, logvars = [], []
            batch_size = config["training"]["batch_size"]
            for offset in range(0, len(episode["observations"]), batch_size):
                frames = frames_to_tensor(episode["observations"][offset:offset + batch_size]).to(device)
                mu, logvar = vae.encode(frames)
                finite_or_raise(mu, "latent mean")
                finite_or_raise(logvar, "latent log variance")
                mus.append(mu.cpu().numpy())
                logvars.append(logvar.cpu().numpy())
            np.savez_compressed(destination / path.name, mu=np.concatenate(mus), logvar=np.concatenate(logvars),
                                actions=episode["actions"], terminated=episode["terminated"], truncated=episode["truncated"])
            write_json((destination / path.name).with_suffix(".json"),
                       dict(vae_sha256=sha, source_path=str(path.resolve()), source_sha256=fingerprint(path),
                            frames=len(episode["observations"]), latent_dim=vae.latent_dim))
    print(f"Encoded {len(paths)} episodes into {destination}")


def encode_main():
    p = parser("Cache frozen VAE posterior means and log variances for every frame.")
    p.add_argument("--vae-checkpoint", required=True, type=Path)
    add_device(p)
    args = p.parse_args()
    encode_dataset(configured(args), args.vae_checkpoint)


def train_vae_main():
    p = parser("Train the ConvVAE with separate reconstruction and KL logging.")
    add_device(p)
    args = p.parse_args()
    train_vae(configured(args))


def eval_vae_main():
    p = parser("Evaluate held-out VAE frames and save original/reconstruction images.", config=False)
    p.add_argument("--checkpoint", required=True, type=Path)
    add_device(p)
    args = p.parse_args()
    evaluate_vae(args.checkpoint, args.device)


def train_mdn_main():
    p = parser("Train the LSTM mixture dynamics model on episode-safe latent windows.")
    p.add_argument("--vae-checkpoint", required=True, type=Path)
    add_device(p)
    args = p.parse_args()
    train_mdn(configured(args), args.vae_checkpoint)


def eval_mdn_main():
    p = parser("Evaluate held-out one-step NLL and a stationary Gaussian diagnostic.", config=False)
    p.add_argument("--checkpoint", required=True, type=Path)
    add_device(p)
    args = p.parse_args()
    evaluate_mdn(args.checkpoint, args.device)


def run_rollout(vae_checkpoint, mdn_checkpoint, *, episode_name=None, horizon=None,
                start=None, trajectories=None, device_name=None, deterministic=False, video=False):
    from .utils import load_checkpoint
    checkpoint = load_checkpoint(mdn_checkpoint)
    config = checkpoint["config"]
    if fingerprint(vae_checkpoint) != checkpoint["vae_sha256"]:
        raise ValueError("The rollout VAE does not match the dynamics checkpoint.")
    device = select_device(device_name or config["device"])
    vae, _ = load_vae(vae_checkpoint, device)
    model, _ = load_mdn(mdn_checkpoint, device)
    root = Path(config["output_dir"])
    split = episode_split(root / "data", config["seed"], config["data"]["val_fraction"])
    names = [Path(episode_name).name] if episode_name else split["val"]
    if any(name not in split["val"] for name in names):
        raise ValueError("Rollout evaluation requires an episode from the held-out validation split.")
    settings = config["rollout"]
    horizon = settings["horizon"] if horizon is None else horizon
    start = settings["start"] if start is None else start
    trajectories = settings["trajectories"] if trajectories is None else trajectories
    if horizon < 1 or start < 0 or trajectories < 1:
        raise ValueError("horizon and trajectories must be positive; start must be non-negative")
    rows, example_saved = [], False
    first_episode_samples = []
    for name in names:
        episode = load_episode(root / "data" / name)
        if start >= len(episode["actions"]):
            raise ValueError(f"start={start} is beyond episode {name}")
        length = min(horizon, len(episode["actions"]) - start)
        # Encode observed history only. Future target encoding below is used solely
        # for metrics and is never passed into warm_state() or imagine().
        with torch.no_grad():
            history_mu, history_logvar = _encode_frames(vae, episode["observations"][:start + 1], device,
                                                       config["training"]["batch_size"])
            future_mu, future_logvar = _encode_frames(vae, episode["observations"][start + 1:start + length + 1],
                                                     device, config["training"]["batch_size"])
        actions = torch.from_numpy(episode["actions"][:start + length]).to(device)[None]
        for trial in range(trajectories):
            seed = config["seed"] + 100000 + trial
            seed_all(seed)
            history = posterior_latents(history_mu, history_logvar, config["training"]["latent_mode"])[None]
            state = warm_state(model, history[:, :start], actions[:, :start])
            generated = imagine(model, history[:, start], actions[:, start:], state, deterministic)
            finite_or_raise(generated, "imagined latent")
            # Sample evaluation targets after imagination, so future data cannot
            # influence generation even through the random number stream.
            future = posterior_latents(future_mu, future_logvar, config["training"]["latent_mode"])[None]
            observed = torch.cat((history[:, start:start + 1], future[:, :-1]), dim=1)
            with torch.no_grad():
                params, _ = model(observed, actions[:, start:], state)
                teacher_nll = -log_prob(params, future)[0]
                imagined_frames = to_images(vae.decode(generated[0]))
                observed_reconstruction = to_images(vae.decode(future_mu))
            finite_or_raise(teacher_nll, "teacher-forced NLL")
            diagnostic_mse = (generated[0] - future_mu).square().mean(-1)
            for h in sorted(set([x for x in (1, 5, 10, 20) if x <= length] + [length])):
                rows.append(dict(episode=name, seed=seed, start=start, horizon=h,
                                 teacher_forced_nll=float(teacher_nll[:h].mean().item()),
                                 free_running_latent_mse_to_mu=float(diagnostic_mse[h - 1].item()),
                                 generation="mixture_expectation_diagnostic" if deterministic else "independent_mixture_sample"))
            if name == names[0]:
                first_episode_samples.append(imagined_frames)
            if not example_saved:
                real = episode["observations"][start + 1:start + length + 1]
                label = "Dream mean (diag)" if deterministic else "Dream sample"
                image_grid([real, observed_reconstruction, imagined_frames],
                           ["Real future", "VAE decode(mu)", label], root / "plots/rollout.png", max_columns=20)
                comparison_gif(real, imagined_frames, root / "plots/rollout.gif", imagined_label=label)
                if video:
                    gif_to_video(root / "plots/rollout.gif", root / "plots/rollout.mp4")
                example_saved = True
    image_grid(first_episode_samples, [f"Trajectory {i}" for i in range(len(first_episode_samples))],
               root / "plots/rollout_variation.png", max_columns=20)
    import csv
    directory = root / "metrics"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "rollout.json", rows)
    with open(directory / "rollout.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "rollout_config.json", dict(config=config, episodes=names, start=start, horizon=horizon,
                                                trajectories=trajectories, deterministic=deterministic, video=video))
    print(f"Evaluated {len(names)} held-out episodes with {trajectories} trajectories each; artifacts in {root}")
    return rows


@torch.no_grad()
def _encode_frames(vae, frames, device, batch_size):
    mus, logvars = [], []
    for offset in range(0, len(frames), batch_size):
        mu, logvar = vae.encode(frames_to_tensor(frames[offset:offset + batch_size]).to(device))
        mus.append(mu)
        logvars.append(logvar)
    return torch.cat(mus), torch.cat(logvars)


def rollout_main():
    p = parser("Compare teacher-forced predictions and free-running action replay on held-out episodes.", config=False)
    p.add_argument("--vae-checkpoint", required=True, type=Path)
    p.add_argument("--mdn-checkpoint", required=True, type=Path)
    p.add_argument("--video", action="store_true", help="Also export actual (left) vs generated (right) as MP4; needs video extra")
    p.add_argument("--episode", help="Held-out episode filename; default: every validation episode")
    p.add_argument("--horizon", type=int)
    p.add_argument("--start", type=int)
    p.add_argument("--trajectories", type=int)
    p.add_argument("--deterministic", action="store_true", help="Use mixture expectations as a diagnostic")
    add_device(p)
    args = p.parse_args()
    run_rollout(args.vae_checkpoint, args.mdn_checkpoint, episode_name=args.episode, horizon=args.horizon,
                start=args.start, trajectories=args.trajectories, device_name=args.device, deterministic=args.deterministic, video=args.video)
