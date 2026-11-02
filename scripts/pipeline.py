"""Run the separate experiment scripts in order, or print their commands."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ROOT

STAGES = ("collect", "inspect", "train-vae", "eval-vae", "encode", "train-mdn", "eval-mdn", "rollout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--from-stage", choices=STAGES, default="collect")
    parser.add_argument("--to-stage", choices=STAGES, default="rollout")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing any stage")
    args = parser.parse_args()
    # --help works with the standard library alone; execution needs project dependencies.
    try:
        import yaml
    except ImportError:
        parser.error("PyYAML is missing. Activate the project environment and install -e '.[dev]' first.")
    config_path = Path(args.config).resolve()
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["output_dir"]).resolve()
    if STAGES.index(args.from_stage) > STAGES.index(args.to_stage):
        parser.error("--from-stage must precede --to-stage")
    vae, mdn = root / "checkpoints/vae_best.pt", root / "checkpoints/mdn_best.pt"
    config_args = ["--config", str(config_path)]
    device_args = ["--device", args.device] if args.device else []
    stages = {
        "collect": ("collect.py", config_args),
        "inspect": ("inspect_dataset.py", config_args),
        "train-vae": ("train_vae.py", config_args + device_args),
        "eval-vae": ("eval_vae.py", ["--checkpoint", str(vae)] + device_args),
        "encode": ("encode_dataset.py", config_args + ["--vae-checkpoint", str(vae)] + device_args),
        "train-mdn": ("train_mdn_rnn.py", config_args + ["--vae-checkpoint", str(vae)] + device_args),
        "eval-mdn": ("eval_mdn_rnn.py", ["--checkpoint", str(mdn)] + device_args),
        "rollout": ("rollout.py", ["--vae-checkpoint", str(vae), "--mdn-checkpoint", str(mdn)] + device_args),
    }
    selected = STAGES[STAGES.index(args.from_stage):STAGES.index(args.to_stage) + 1]
    records = []
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = root / f"pipeline_{stamp}.json"
    for stage in selected:
        script, options = stages[stage]
        command = [sys.executable, str(ROOT / "scripts" / script), *options]
        print(f"[{stage}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False)
        records.append(dict(stage=stage, command=command, returncode=result.returncode))
        manifest.write_text(json.dumps(dict(config=str(config_path), stages=records), indent=2) + "\n")
        if result.returncode:
            print(f"Stage {stage} failed. Fix the issue, then resume with --from-stage {stage}.", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
