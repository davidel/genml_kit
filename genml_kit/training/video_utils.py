"""Small helpers to write evaluation videos for RL runs.

Kept dependency-light: MP4 via ``imageio[ffmpeg]`` when available, otherwise
animated GIF via ``PIL``.  Both writers are exercised by unit tests so the
recording path is testable in CI without a system ffmpeg binary.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def write_video(frames, path, fps=30):
  """Write ``frames`` to ``path`` as MP4 (preferred) or animated GIF.

  Args:
    frames: Iterable of numpy RGB (H, W, 3) uint8 arrays.  ``None`` entries
      are skipped.
    path: Destination path.  If it ends in ``.gif``, an animated GIF is
      always written via PIL; otherwise MP4 via imageio[ffmpeg] is
      attempted first, degrading to a sibling GIF on failure (e.g. the
      ffmpeg plugin is unavailable).
    fps: Frames per second for the animation (default 30).

  Returns:
    The path written, or ``None`` if no frame could be encoded (e.g. the
    frames list is empty).
  """
  frames = [np.asarray(f) for f in frames if f is not None]
  if not frames:
    return None
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

  if path.endswith(".gif"):
    _write_gif(frames, path, fps=fps)
    return path

  try:
    _write_mp4(frames, path, fps=fps)
    return path
  except Exception as exc:  # noqa: BLE001 - degrade to GIF
    logger.warning("MP4 writer failed (%s); falling back to GIF.", exc)
    gif_path = os.path.splitext(path)[0] + ".gif"
    _write_gif(frames, gif_path, fps=fps)
    return gif_path


def _write_mp4(frames, path, fps=30):
  # Use the v3 `imageio` API: `imageio.v2` is deprecated and its shim may
  # be removed in future releases.
  import imageio

  writer = imageio.get_writer(
      path,
      fps=fps,
      codec="libx264",
      quality=8,
      pixelformat="yuv420p",
  )
  try:
    for frame in frames:
      writer.append_data(frame)
  finally:
    writer.close()


def _write_gif(frames, path, fps=30):
  """Write an animated GIF via PIL (no ffmpeg required).

  Args:
    frames: List of numpy RGB (H, W, 3) uint8 arrays.
    path: Destination ``.gif`` path.
    fps: Frames per second for the animation.
  """
  from PIL import Image

  duration_ms = max(1, int(1000.0 / fps))
  images = [Image.fromarray(f) for f in frames]
  images[0].save(
      path,
      save_all=True,
      append_images=images[1:],
      duration=duration_ms,
      loop=0,
  )
