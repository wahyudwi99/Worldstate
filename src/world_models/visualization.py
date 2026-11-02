"""Headless plots, labeled image grids, and animated comparisons."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def to_images(tensor) -> np.ndarray:
    return tensor.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).mul(255).byte().numpy()


def image_grid(rows: list[np.ndarray], labels: list[str], path: Path, max_columns: int = 10):
    if not rows or min(len(row) for row in rows) == 0:
        raise ValueError("Cannot visualize empty image rows")
    columns = min(max_columns, min(len(row) for row in rows))
    indices = np.linspace(0, min(len(row) for row in rows) - 1, columns).astype(int)
    canvas = Image.new("RGB", (150 + 64 * columns, 84 * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (frames, label) in enumerate(zip(rows, labels)):
        draw.text((5, r * 84 + 22), label, fill="black")
        for c, index in enumerate(indices):
            canvas.paste(Image.fromarray(frames[index]), (150 + c * 64, r * 84 + 16))
            draw.text((150 + c * 64, r * 84), str(index), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def comparison_gif(real: np.ndarray, imagined: np.ndarray, path: Path, imagined_label: str = "Imagined sample"):
    frames = []
    for index, (a, b) in enumerate(zip(real, imagined)):
        canvas = Image.new("RGB", (256, 154), "white")
        canvas.paste(Image.fromarray(a).resize((128, 128)), (0, 26))
        canvas.paste(Image.fromarray(b).resize((128, 128)), (128, 26))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 3), f"Real t+{index+1}", fill="black")
        draw.text((132, 3), imagined_label, fill="black")
        frames.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=150, loop=0)


def gif_to_video(source: Path, destination: Path, scale: int = 2):
    """Export the existing synchronized comparison GIF as a slowed-down MP4.

    Preserve GIF frame timing (150 ms per rollout frame) and both labeled panels.
    This is a diagnostic playback, rather than the simulator's native frame rate.
    """
    import math
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401 -- required by the FFMPEG plugin
    except ImportError as exc:
        raise RuntimeError("Video export needs the video extra: uv pip install -e '.[video]'") from exc
    if scale < 1:
        raise ValueError("scale must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as gif:
        # A common integer timebase preserves variable GIF frame durations too.
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(max(10, int(gif.info.get("duration", 150))))
        tick = math.gcd(*durations)
        width, height = gif.width * scale, gif.height * scale
        # H.264 yuv420p requires even dimensions; pad instead of stretching.
        size = (width + width % 2, height + height % 2)
        with imageio.get_writer(str(destination), format="FFMPEG", fps=1000 / tick,
                                codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                                output_params=["-movflags", "+faststart"]) as writer:
            for index, duration in enumerate(durations):
                gif.seek(index)
                frame = gif.convert("RGB").resize((width, height), Image.Resampling.NEAREST)
                canvas = Image.new("RGB", size, "white")
                canvas.paste(frame, (0, 0))
                for _ in range(duration // tick):
                    writer.append_data(np.asarray(canvas))


def plot_history(rows: list[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in rows[0]:
        if key != "epoch":
            ax.plot([row["epoch"] for row in rows], [row[key] for row in rows], label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (see README for reductions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
