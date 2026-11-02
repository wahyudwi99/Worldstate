# Running the experiments

[← Results and project overview](../README.md) · [Technical details](TECHNICAL.md)

Run all commands from the repository root. Existing local outputs are not bundled
with the documentation; use a new output directory when collecting a fresh dataset.

## Installation with uv on Apple Silicon

Use a native arm64 Python environment. Python 3.12 is a conservative starting
point; PyTorch and macOS versions must support one another. MPS availability is
checked at runtime. The project does not need CUDA.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first if
it is not already available. Run this command from the repository root:

```bash
uv sync --python 3.12 --extra dev
```

uv creates or updates `.venv`, installs this project in editable mode, and
includes the `dev` extra (`pytest`). `pyproject.toml` remains the dependency
source; `uv.lock` records resolved versions. Keep the generated lockfile in
version control. After a lockfile exists, use `uv sync --locked --extra dev`
to install without changing it. Use `uv lock --upgrade` to deliberately update
resolved versions, then sync again.

Run commands with `uv run --extra dev` as shown below; activating `.venv` is
optional. For example, add a runtime dependency with `uv add PACKAGE`, or a
development dependency with `uv add --optional dev PACKAGE`.

SWIG is configured as an extra Box2D build dependency so source builds can find
it inside uv's isolated build environment. If Box2D builds from source, a
compiler may also be required. On macOS, install Apple's command-line developer
tools if the installer reports that a compiler is missing. Do not use an
Intel/Rosetta Python for MPS.

## Start small and study each stage

The smoke and baseline experiments have saved results; the commands below reproduce
the workflow in a fresh output directory.

First inspect the commands without running the experiment:

```bash
uv run --extra dev python scripts/pipeline.py --help
uv run --extra dev python scripts/pipeline.py --config configs/smoke.yaml --device cpu --dry-run
```

Run just collection and inspection:

```bash
uv run --extra dev python scripts/pipeline.py --config configs/smoke.yaml --to-stage inspect
```

Inspect `outputs/smoke/plots/dataset_montage.png` and the episode files before
continuing. Run unit tests separately when ready; these use tiny synthetic data
and do not start CarRacing or perform multi-epoch training:

```bash
uv run --extra dev python -m pytest -q
```

Continue through the VAE stage:

```bash
uv run --extra dev python scripts/pipeline.py --config configs/smoke.yaml --from-stage train-vae --to-stage eval-vae --device cpu
```

Continue through latent caching and dynamics evaluation:

```bash
uv run --extra dev python scripts/pipeline.py --config configs/smoke.yaml --from-stage encode --to-stage eval-mdn --device cpu
```

Finally generate and evaluate imagined frames:

```bash
uv run --extra dev python scripts/pipeline.py --config configs/smoke.yaml --from-stage rollout --device cpu
```

For a fresh output directory, the entire pipeline is one command:

```bash
uv run --extra dev python scripts/pipeline.py --config configs/smoke.yaml --device cpu
```

Omit `--device cpu` to use the config's `auto` selection: **MPS, then CPU**.
CUDA is available only through an explicit `--device cuda` override.
The runner stops at the first failed stage and prints a resume command. Stage
resumption starts that stage again; it does not resume an interrupted optimizer
from its last step. Collection refuses to overwrite existing episodes. To run a
new experiment, copy the config and give it a new `output_dir`.

The smoke config collects only 2 episodes of at most 50 steps, then trains each
model for 2 epochs. This checks pipeline behavior, not prediction quality. Do not
expect coherent imagined driving after such a small run. For an intermediate
experiment, copy the smoke config and set 20 episodes and 300 steps. The baseline
config is a larger, manual next step; no scripts automatically escalate to it.

## Export actual vs generated as a side-by-side video

The comparison shows actual future frames on the left (`Real t+...`) and the
first generated trajectory on the right (`Dream sample`). Both sides use matching
future timesteps and recorded actions. The optional video extra uses
[ImageIO's FFmpeg plugin](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.ffmpeg.html)
to write an H.264 MP4. Playback preserves the comparison GIF's timing (150 ms
per frame); it is slowed down for inspection, rather than simulator real time.
Upscaling enlarges the 64x64 frames without adding model detail.

To export an existing `rollout.gif` without collecting data or training again,
activate your conda environment and run from the repository root:

```bash
uv pip install -e '.[dev,video]'
python scripts/export_video.py
```

The default output is `outputs/smoke/plots/rollout.mp4`. For another experiment:

```bash
python scripts/export_video.py --input outputs/my_run/plots/rollout.gif --output outputs/my_run/plots/rollout.mp4
```

To produce an MP4 while rerunning only generation from saved checkpoints:

```bash
python scripts/rollout.py --vae-checkpoint outputs/smoke/checkpoints/vae_best.pt --mdn-checkpoint outputs/smoke/checkpoints/mdn_best.pt --device cpu --video
```

If you use the uv-managed `.venv`, install with `uv sync --extra dev --extra video`
and prefix the Python commands above with `uv run --extra dev --extra video`.
The video extra is optional; ordinary rollout continues to produce PNG and GIF
without an MP4. Use the sync command above to install the optional video dependencies.

## Separate script interfaces

Every stage is also an independent script with `--help`:

```bash
uv run --extra dev python scripts/collect.py --config configs/smoke.yaml --episodes 2 --max-steps 50 --seed 42 --policy random
uv run --extra dev python scripts/inspect_dataset.py --config configs/smoke.yaml
uv run --extra dev python scripts/train_vae.py --config configs/smoke.yaml --device cpu
uv run --extra dev python scripts/eval_vae.py --checkpoint outputs/smoke/checkpoints/vae_best.pt --device cpu
uv run --extra dev python scripts/encode_dataset.py --config configs/smoke.yaml --vae-checkpoint outputs/smoke/checkpoints/vae_best.pt --device cpu
uv run --extra dev python scripts/train_mdn_rnn.py --config configs/smoke.yaml --vae-checkpoint outputs/smoke/checkpoints/vae_best.pt --device cpu
uv run --extra dev python scripts/eval_mdn_rnn.py --checkpoint outputs/smoke/checkpoints/mdn_best.pt --device cpu
uv run --extra dev python scripts/rollout.py --vae-checkpoint outputs/smoke/checkpoints/vae_best.pt --mdn-checkpoint outputs/smoke/checkpoints/mdn_best.pt --horizon 20 --device cpu
```

Choose either the pipeline or these separate commands for a fresh run. Do not
collect the same dataset twice. The first command's policy override is recorded
in collection metadata. Prefer editing a copied config for a complete experiment
so all settings stay together. `--output-dir` on collection alone does not update
the configs used by subsequent scripts.

`rollout.py` defaults to all validation episodes and supports `--episode
 episode_00001.npz`, `--start 5`, and `--trajectories 3`. The episode must belong
to the saved validation split. Horizon is capped to the remaining recorded
transitions; the actual evaluated horizons are recorded in metrics.
`--deterministic` replaces mixture sampling with the mixture expectation and is
explicitly a diagnostic, not a sample from the learned distribution.

