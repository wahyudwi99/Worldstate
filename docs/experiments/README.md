# Experiment snapshot

[← Results overview](../../README.md) · [Run guide](../RUNNING.md)

Copied from local `outputs/smoke/` and `outputs/baseline/` on 2 November 2026.
These files preserve the evidence behind the README without committing datasets,
latent caches, or checkpoints. Media is copied unchanged into `docs/assets/`.
The generated `outputs/` directory remains ignored by Git.

Each run directory includes:

- `dataset_inspection.json`: episode counts and observed lengths.
- `split.json`: persistent episode-level train/validation split.
- `*_config.json`: resolved stage settings, including device overrides.
- `vae_evaluation.json`: held-out reconstruction, KL, and total loss.
- `mdn_evaluation.json`: held-out one-step NLL and Gaussian comparator.
- `*_training.csv`: epoch-level training and validation histories.
- `rollout.json`: individual episode/seed/horizon diagnostic records.

README values use four decimal places. Transition counts sum inspected episode
lengths. NLL improvement subtracts MDN NLL from its own stationary-Gaussian NLL.
Rollout MSE is the unweighted arithmetic mean of
`free_running_latent_mse_to_mu` across records sharing a horizon.
These are single-seed exploratory runs, with no repeated-run confidence intervals.

The JSON records are local artifact evidence; documentation review did not rerun
training or independently reproduce the experiments. Configured `auto` devices
and explicit stage overrides should be read from the respective stage configs.
