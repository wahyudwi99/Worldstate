"""Check panel alignment and playback timing without requiring an encoder."""
import sys
from types import ModuleType

import numpy as np
from PIL import Image

from world_models.visualization import gif_to_video


def test_video_preserves_panels_and_variable_frame_timing(tmp_path, monkeypatch):
    frames = []
    for color in ("red", "green"):
        frame = Image.new("RGB", (8, 4), color)
        frame.paste(Image.new("RGB", (4, 4), "blue"), (4, 0))
        frames.append(frame)
    source = tmp_path / "comparison.gif"
    frames[0].save(source, save_all=True, append_images=frames[1:], duration=[100, 200])
    encoded, options = [], {}

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def append_data(self, frame):
            encoded.append(frame.copy())

    def get_writer(path, **kwargs):
        options.update(kwargs)
        return Writer()

    imageio = ModuleType("imageio")
    imageio.v2 = ModuleType("imageio.v2")
    imageio.v2.get_writer = get_writer
    monkeypatch.setitem(sys.modules, "imageio", imageio)
    monkeypatch.setitem(sys.modules, "imageio.v2", imageio.v2)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", ModuleType("imageio_ffmpeg"))
    gif_to_video(source, tmp_path / "comparison.mp4")
    assert options["fps"] == 10
    assert len(encoded) == 3  # 100 ms + 200 ms at 10 fps
    assert encoded[0].shape == (8, 16, 3)
    np.testing.assert_array_equal(encoded[0][4, 2], [255, 0, 0])
    np.testing.assert_array_equal(encoded[0][4, 12], [0, 0, 255])
    np.testing.assert_array_equal(encoded[1], encoded[2])
