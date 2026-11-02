"""Export an existing side-by-side rollout GIF without rerunning the models."""
import argparse
from pathlib import Path

from _bootstrap import run
from world_models.visualization import gif_to_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/smoke/plots/rollout.gif"))
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke/plots/rollout.mp4"))
    parser.add_argument("--scale", type=int, default=2, help="Integer enlargement; default: 2")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files")
    gif_to_video(args.input, args.output, scale=args.scale)
    print(f"Saved actual (left) vs generated (right): {args.output}")


if __name__ == "__main__":
    run(main)
