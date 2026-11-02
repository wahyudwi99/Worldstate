# Validation status

## Current saved experiment evidence — 2 November 2026

The checkout now contains smoke and baseline training histories, held-out VAE
and MDN evaluations, rollout records, PNG/GIF visualizations, and a baseline MP4.
A portable [snapshot](experiments/README.md) backs the [README results](../README.md).
These artifacts supersede the historical absence-of-results statements below.
The documentation update inspected existing results; it did not rerun training
or establish new unit-test or encoder-validation results. For execution, see
[the run guide](RUNNING.md).

## Historical implementation-session notes

The sections below describe earlier sessions, before the saved experiments.

## Checks performed in the implementation session

- Compiled all Python source, entry points, and test modules with Python 3.12
  `compileall`: passed.
- Ran `python scripts/pipeline.py --help`: passed.
- Parsed project metadata with Python's `tomllib`: passed.
- Statically checked all stage entry-point names and local module imports: passed.

These checks establish syntax and basic repository consistency only. They do not
establish runtime correctness of the models or simulator.

## Checks intentionally left for the project owner

The project owner requested no heavy testing. No environment episodes, model
training, unit tests, reconstruction evaluation, or imagined rollouts were run.
No training metrics or generated model artifacts are claimed.

A Python 3.12 virtual environment was created. A dependency installation attempt
failed because the session could not resolve the Python package server. There
were no ML dependencies preinstalled. Install the dependencies from the README
before running individual scripts or pytest.

## Suggested manual acceptance sequence

1. Run each script with `--help` after installing dependencies.
2. Run the pipeline with `--dry-run` and inspect the commands.
3. Run `--to-stage inspect`; inspect array metadata and the montage.
4. Run `uv run --extra dev python -m pytest -q` for small synthetic model/data tests.
5. Run only VAE training/evaluation; check finite losses and reconstruction PNG.
6. Encode the data and run MDN training/evaluation; inspect finite held-out NLL.
7. Run rollout and inspect metrics, real/reconstruction/imagined grid, GIF, and
   the trajectory-variation grid.

Stay on the smoke config initially. Finite losses and a working pipeline are
acceptance checks, not evidence of useful dynamics or paper-level performance.

## uv migration checks

- Parsed the updated `pyproject.toml` with `tomllib`: passed.
- Replaced pip installation instructions and script examples with `uv sync`
  and `uv run --extra dev` in the README.
- Attempted `uv lock` with uv 0.10.11: failed because DNS resolution for
  `pypi.org` was unavailable. No lockfile was generated and dependency
  installation was not validated. Run `uv sync --python 3.12 --extra dev`
  locally to resolve dependencies, generate `uv.lock`, and install them.
- No simulation, training, or unit tests were run during this migration.

## Side-by-side MP4 export

- Added optional `video` dependencies, `scripts/export_video.py`, and rollout
  `--video` to export actual (left) and imagined (right) frames.
- A focused test passed for panel alignment and variable GIF frame timing using
  a simulated writer; this does not establish actual encoder compatibility.
- ImageIO installation and lockfile refresh failed due to unavailable PyPI DNS.
  The existing lockfile has not been updated for the new optional video extra.
- Actual MP4 encoding has not been validated. No training was run for this change.
