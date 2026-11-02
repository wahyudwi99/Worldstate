# Implementation notes

[← Project overview](../README.md) · [Run guide](RUNNING.md)

## Repository map

| File | Purpose |
|---|---|
| `scripts/pipeline.py` | Ordered subprocess runner, stage selection, dry run, exit-status manifest |
| `scripts/collect.py` | Generate real environment episodes |
| `scripts/inspect_dataset.py` | Validate arrays and save a montage |
| `scripts/train_vae.py`, `scripts/eval_vae.py` | Train and evaluate visual representations |
| `scripts/encode_dataset.py` | Cache frozen VAE posterior parameters |
| `scripts/train_mdn_rnn.py`, `scripts/eval_mdn_rnn.py` | Train and evaluate latent dynamics |
| `scripts/rollout.py` | Generate imagined frames with real recorded action replay |
| `src/world_models/envs.py` | Shared resizing, tensor conversion, exploration policies |
| `src/world_models/dataset.py` | Episode contract, persistent split, bounded-memory loaders |
| `src/world_models/vae.py` | Encoder, decoder, VAE loss |
| `src/world_models/mdn_rnn.py` | LSTM, mixture likelihood, independent sampling |
| `src/world_models/rollout.py` | History warm-up and free-running feedback loop |
| `src/world_models/training.py` | Optimization, checkpoints, held-out evaluation |
| `src/world_models/commands.py` | CLI argument parsing and stage orchestration |
| `src/world_models/visualization.py` | Headless plots, labeled grids, GIFs |
| `configs/` | Small smoke and larger baseline settings |
| `tests/` | Synthetic correctness tests, no simulator training |
| `docs/controller_next_steps.md` | Deferred controller design and compute considerations |

## Data contract and preprocessing

An episode is a compressed `.npz` containing:

| Array | Shape | Type |
|---|---|---|
| observations | `[T+1,64,64,3]` | uint8 |
| actions | `[T,3]` | float32 |
| rewards | `[T]` | float32 |
| terminated | `[T]` | bool |
| truncated | `[T]` | bool |

`observations[t]` plus `actions[t]` leads to `observations[t+1]`. The initial
observation and the final next observation are both retained. Steering lies in
`[-1,1]`; gas and brake lie in `[0,1]`. Reaching the collector's `max_steps`
sets the final `truncated` flag and records `collector_time_limit=true` in the
JSON metadata. Natural termination remains separate.

Raw RGB observations are resized from the full image to 64x64 with Pillow
bilinear interpolation, without cropping. Models receive float32 NCHW tensors in
`[0,1]`. All observed-frame paths share these transformations. Cached images are
not repeatedly resized during training.

Random exploration supports independent actions and correlated actions held for
`action_repeat` steps. Correlated exploration favors forward gas and occasionally
brakes; it is a simple collection heuristic, not a trained policy.

Splits are seeded and saved by episode **before** creating frames or latent
windows. Changing the dataset, split seed, or validation fraction requires a new
output directory. The loaders retain at most one decompressed episode per worker,
and use zero background workers by default to keep memory use simple on macOS.
Compressed NPZ cannot be memory mapped; shuffled access may decompress an episode
more than once. This is a throughput tradeoff for the small initial datasets.

Latent caches contain posterior `mu` and `logvar` for all T+1 frames, plus T
actions. Sidecar metadata stores the VAE checkpoint SHA-256 and source episode
SHA-256. Dynamics loading rejects mismatched caches. Source paths are recorded
as generated provenance; regenerate caches after moving the dataset to a new
location. Re-encoding after training a new VAE is mandatory.

## Models and mathematics

### Vision

The encoder uses four stride-2 convolution blocks:

```text
[B,3,64,64] -> [B,32,32,32] -> [B,64,16,16]
             -> [B,128,8,8] -> [B,256,4,4] -> mu, logvar [B,D]
```

A linear projection and four transpose convolutions decode `[B,D]` back to
`[B,3,64,64]`, with sigmoid output. D defaults to 32. Posterior log variance is
clamped to `[-20,10]` for numerical stability.

```text
z = mu + exp(0.5 * logvar) * epsilon, epsilon ~ N(0,I)
L_reconstruction = mean_batch(sum_pixels_channels((decoded - image)^2))
L_KL = mean_batch(-0.5 * sum_latent(1 + logvar - mu^2 - exp(logvar)))
L_VAE = L_reconstruction + beta * L_KL
```

Both terms are sums over their feature dimensions and means over examples.
`beta=1` therefore has a specific scale; do not interpret it as equivalent to
code that averages reconstruction over pixels. Logs report both terms separately.
Training uses posterior samples. Validation decodes the posterior mean for a
repeatable reconstruction diagnostic; it is not a Monte Carlo ELBO estimate.

### Memory

The LSTM receives concatenated `[z_t,a_t]` and produces parameters of the next
latent distribution. Default baseline hidden size is 256 and mixture count is 5;
the smoke config uses 64 and 3.

Each parameter tensor has shape `[B,T,D,K]`. This implementation uses an
**independent mixture index per latent dimension**:

```text
p(z_next | z, a, history) = product_d sum_k pi[d,k] * N(z_next[d]; mu[d,k], sigma[d,k]^2)
log pi = log_softmax(logits, mixture_axis)
log p = sum_d logsumexp_k(log pi[d,k] + log Normal[d,k])
L_MDN = -sum_valid_transitions(log p) / count_valid_transitions
```

NLL is in **nats per transition, summed over latent dimensions**. Log standard
deviations are clamped to `[-7,5]` in likelihood and sampling. Stable log-softmax
and log-sum-exp avoid explicitly summing tiny densities.

Training is teacher forced: observed `z_t` is input, actual adjacent `z_(t+1)`
is target. Each full window of L+1 posterior latents is sampled once before
shifting; an adjacent shared frame therefore has one consistent draw inside that
window. Set `training.latent_mode: mu` for deterministic debugging. VAE parameters
are frozen and excluded from the dynamics optimizer.

Windows do not cross episode boundaries. The final short window is padded and
masked. Hidden and cell states reset at **every window**, not only episode
boundaries: this deliberately uses truncated context for simple independent
minibatches. Validation NLL reports the same context convention. Rollout history
warm-up can provide longer context and is reported separately.

### Imagined rollouts and evaluation

At the configured start index s, the evaluator encodes observed frames through
s and consumes only transitions before s to warm the LSTM. It then supplies
`z_s` and recorded actions `a_s ... a_(s+H-1)` to the imagination function.
Every subsequent latent input is the previous generated latent. Future observed
latents are never inputs to this function.

Future frames are encoded separately for teacher-forced NLL and diagnostic
comparisons. Evaluation targets are sampled only after generation, so they do
not alter the generation random stream. Different trajectory seeds reveal
stochastic variation. No simulator step is needed for imagined generation.

Metrics include:

- Held-out VAE reconstruction, KL, and total loss.
- Held-out teacher-forced one-step MDN NLL.
- A fixed unit-variance stationary Gaussian centered at z_t as a simple baseline.
- Teacher-forced NLL over prefixes at horizons 1, 5, 10, and 20, where available.
- Free-running endpoint latent MSE to posterior mu at those horizons, as a diagnostic.

Teacher-forced NLL does not measure free-running trajectory quality. A stochastic
sample's MSE to posterior mu is also not a distributional quality score. We do
not claim that it or pixel drift fully measures learned dynamics. The visual
comparison shows real future, VAE reconstruction of that future, and an imagined
sample separately to distinguish representation error from prediction drift.

## Outputs and reproducibility

Each configured output directory contains:

```text
data/                 episode NPZ files, JSON metadata, persistent split.json
latents/              posterior caches and VAE/source provenance
checkpoints/          vae_best.pt, vae_last.pt, mdn_best.pt, mdn_last.pt
metrics/              CSV/JSON training histories and JSON/CSV evaluation metrics
plots/                collection montage, loss plots, reconstructions, rollout PNG/GIF
*_config.json         resolved experiment settings
pipeline_*.json       executed commands and exit statuses
```

Checkpoints include weights, optimizer state, epoch, training step, configuration,
seed, and best validation metric. Dynamics checkpoints also record the VAE hash.
The best checkpoint is selected on held-out validation loss. Training scripts
start a new optimization run; they do not currently implement optimizer resume.
Only load trusted checkpoints. Evaluation uses the configuration saved with the
checkpoint, with an optional device override.

Seeds control Python, NumPy, PyTorch, collection, train/validation split, and
trajectory sampling. Exact bitwise equality across hardware, PyTorch releases,
or MPS kernels is not guaranteed. Compare runs on the same backend when studying
small metric changes. Validation posterior sampling is re-seeded each epoch.

MPS failures are surfaced rather than silently restarting training. If an
operation fails on MPS, rerun that stage with `--device cpu`; keep both logs and
note the backend change. The small MDN is worth benchmarking on CPU as well.
Mixture-index sampling deliberately uses CPU multinomial, then returns indices
to the model device. There is no mixed precision or automatic large run.

## Differences from a full paper reproduction

This project uses Gymnasium CarRacing-v3 rather than the original environment
version, tiny exploratory datasets, a symmetric padded convolutional VAE, simple
random exploration, and independent sequence-window training with state resets.
It evaluates open-loop action replay and does not implement an interactive dream
Gymnasium environment, reward/termination predictors, policy evolution, or paper
score reproduction. Reconstruction loss, architecture details, and configurable
training settings should be treated as explicit implementation choices.

No controller is included. See [the deferred plan](controller_next_steps.md).
No cloud training, automatic hyperparameter search, or large simulator search is
run by the pipeline.

