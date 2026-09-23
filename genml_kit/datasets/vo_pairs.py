"""Synthetic VO pair dataset -- GT similarity + irreducible residual.

Rendering follows the data-generation math of ``vo/README.md`` section 2:
a ground plane is photographed by an oblique camera, frame B is rendered
through the ground-to-image homography of a moved camera, and the GT
similarity is the best similarity fit to that homography (Umeyama over
image corners).  The residual between homography and fit is what the
network can never remove; it is emitted as metadata for confidence
supervision.
"""

import collections

import numpy as np
import torch

from genml_kit.geometry.similarity import umeyama_similarity

VOPairMeta = collections.namedtuple("VOPairMeta",
                                    ["gt", "gt_residual", "terrain", "range_bin"])


def look_at_ground_h(cam_pos, cam_rpy, cam, _size):
  """Ground-to-image homography of a pinhole camera over a z=0 plane.

  The camera axes are derived from pitch/yaw: the optical axis ``f``
  points forward-down at *pitch* below horizontal, ``down = f x right``
  completes the right-handed frame.  World-to-camera is ``R^T (p - c)``
  with the camera axes as the rows of ``R^T``; the homography columns
  for the ground plane z=0 are then ``K r_x``, ``K r_y``, ``K t``.

  Args:
      cam_pos: (3,) camera position in world coordinates (z = height AGL).
      cam_rpy: (roll, pitch, yaw) in radians; yaw rotates the optical
          axis around world z, pitch tips it down from horizontal.
      cam: dict with 'fx', 'fy', 'cx', 'cy' intrinsics.
      _size: (W, H) image size, accepted for API symmetry with the
          sibling homography helpers; unused by the pinhole math (the
          intrinsics come from ``cam``).  The underscore prefix is an
          exception to the codebase no-``_``-argument rule, purely so
          Ruff's ARG001 does not flag the intentionally unused argument.

  Returns:
      (3, 3) numpy array mapping world-ground points (x, y, 1) to pixels.
  """
  fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
  roll, pitch, yaw = cam_rpy
  cp, sp = np.cos(pitch), np.sin(pitch)
  cyw, syw = np.cos(yaw), np.sin(yaw)
  cr, sr = np.cos(roll), np.sin(roll)
  up_w = np.array([0.0, 0.0, 1.0])
  f = np.array([cyw * cp, syw * cp, -sp])
  right = np.cross(f, up_w)
  right = right / np.linalg.norm(right)
  down = np.cross(f, right)
  # Roll rotates the camera frame about the optical axis.
  right_r = cr * right + sr * down
  down_r = -sr * right + cr * down
  r_cam_world = np.stack([right_r, down_r, f], axis=0)
  k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
  t = -r_cam_world @ cam_pos
  return np.concatenate(
      [k @ r_cam_world[:, 0:1], k @ r_cam_world[:, 1:2], k @ t.reshape(3, 1)], axis=1)


def sample_motion(rng, motion_cfg):
  """Sample one inter-frame motion from the outer envelope.

  Args:
      rng: numpy Generator.
      motion_cfg: dict with 'translation_frac', 'rot_deg', 'scale_range',
          'height_frac' keys.

  Returns:
      dict with 'dx', 'dy' (ground units), 'dyaw' (rad), 'ds' (height
      ratio) and 'dz' (height change, ground units).
  """
  half = motion_cfg["translation_frac"]
  dx = float(rng.uniform(-half, half))
  dy = float(rng.uniform(-half, half))
  rot = math_radians(motion_cfg["rot_deg"])
  dyaw = float(rng.uniform(-rot, rot))
  lo, hi = motion_cfg["scale_range"]
  ds = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
  dz = float(rng.uniform(-motion_cfg["height_frac"], motion_cfg["height_frac"]))
  return {"dx": dx, "dy": dy, "dyaw": dyaw, "ds": ds, "dz": dz}


def math_radians(deg):
  return float(np.deg2rad(deg))


def homography_to_similarity(h, size):
  """Best similarity fit to a homography + the irreducible residual.

  Fits the similarity that minimizes the mean corner reprojection error
  to the homography-mapped image corners (Umeyama on the four corners),
  then reports the mean corner distance that remains.  That residual is
  the foreshortening part the similarity family cannot absorb
  (vo/README.md section 2).

  Args:
      h: (3, 3) ground-to-image homography of frame B.
      size: (W, H) of the image.

  Returns:
      (sim_params dict with log_s/theta/t tensors, residual_px float).
  """
  width, height = size
  corners = np.array([
      [0.0, 0.0, 1.0],
      [width - 1, 0.0, 1.0],
      [width - 1, height - 1, 1.0],
      [0.0, height - 1, 1.0],
  ])
  mapped = corners @ h.T
  mapped = mapped[:, :2] / mapped[:, 2:3]
  src = torch.tensor(corners[:, :2], dtype=torch.float64).unsqueeze(0)
  dst = torch.tensor(mapped, dtype=torch.float64).unsqueeze(0)
  fit = umeyama_similarity(src, dst)
  mat = (params_to_matrix_np(fit.log_s[0].item(), fit.theta[0].item(),
                             fit.t[0].tolist()))
  warped = np.stack([mat @ c for c in corners])
  warped = warped[:, :2] / warped[:, 2:3]
  residual = float(np.linalg.norm(warped - mapped, axis=1).mean())
  sim = {
      "log_s": fit.log_s[0].item(),
      "theta": fit.theta[0].item(),
      "t": fit.t[0].tolist(),
  }
  return sim, residual


def params_to_matrix_np(log_s, theta, t):
  c, s = np.cos(theta), np.sin(theta)
  return np.array([
      [np.exp(log_s) * c, -np.exp(log_s) * s, t[0]],
      [np.exp(log_s) * s, np.exp(log_s) * c, t[1]],
      [0.0, 0.0, 1.0],
  ])


class VOPairDataset:
  """Generates (frame_a, frame_b, meta) synthetic VO training pairs.

  The base tile is procedurally textured (deterministic per index), so
  the dataset needs no external data and is fully seeded.  Frame B is
  rendered by sampling the base tile through the moved-camera homography.

  Args:
      length: number of pairs.
      size: (W, H) emitted image size.
      motion_cfg: outer-envelope dict, see :func:`sample_motion`.
      terrain: terrain class label string.
      seed: integer seed for this dataset instance.
      cam: intrinsic dict with 'fx', 'fy', 'cx', 'cy' keys.
      pitch_deg: camera pitch below horizontal, in degrees.  The default
          of 45.0 matches the oblique mounting described in ``vo/README.md``.
  """

  def __init__(self,
               length,
               size=(256, 256),
               motion_cfg=None,
               terrain="field",
               seed=0,
               cam=None,
               pitch_deg=45.0):
    self._length = length
    self._size = size
    self._motion_cfg = motion_cfg or {
        "translation_frac": 0.35,
        "rot_deg": 25.0,
        "scale_range": [0.6, 1.6],
        "height_frac": 0.2,
    }
    self._terrain = terrain
    self._seed = seed
    self._cam = cam or {
        "fx": 380.0,
        "fy": 380.0,
        "cx": size[0] / 2.0,
        "cy": size[1] / 2.0
    }
    self._pitch = math_radians(pitch_deg)

  def __len__(self):
    return self._length

  @property
  def size(self):
    """Public read-only access to image size for external consumers."""
    return self._size

  @property
  def terrain(self):
    """Public read-only access to terrain for external consumers."""
    return self._terrain

  @property
  def motion_cfg(self):
    """Public read-only access to motion config for external consumers."""
    return self._motion_cfg

  @property
  def cam(self):
    """Public read-only access to camera params for external consumers."""
    return self._cam

  @property
  def pitch(self):
    """Public read-only access to camera pitch for external consumers."""
    return self._pitch

  # Tile texture is a deliberate, fixed design: low-frequency value noise
  # (``NOISE_CELL_PX`` cells) plus a fine grid (``GRID_PERIOD_PX`` period,
  # ``GRID_LINE_PX`` stroke).  These statistics define the visual structure of the
  # synthetic terrain, not any camera behaviour, so they are hard-coded rather than
  # exposed as constructor options.
  # Base tile is this many image widths per side.
  TILE_SCALE = 4
  # Value-noise cell size in tile pixels.
  NOISE_CELL_PX = 8
  # Grid spacing, tile pixels.
  GRID_PERIOD_PX = 32
  # Grid stroke width, tile pixels.
  GRID_LINE_PX = 2
  # Grid contribution in [0, 1].
  GRID_WEIGHT = 0.25
  # Value-noise contribution in [0, 1].
  NOISE_WEIGHT = 0.75
  # Texture is normalized to this range.
  TILE_CLIP = (0.0, 1.0)

  def _base_tile(self, rng):
    """Procedural ground texture: value noise + grid, in [0, 1].

    The tile is deliberately larger than one image (``TILE_SCALE`` image
    widths per side): frame B's footprint grows with height change and
    translation, and the extra margin keeps it free of edge zeros.
    """
    side = self.TILE_SCALE * max(self._size)
    low = rng.random((side // self.NOISE_CELL_PX + 1, side // self.NOISE_CELL_PX + 1))
    image = np.kron(low, np.ones(
        (self.NOISE_CELL_PX, self.NOISE_CELL_PX)))[:side, :side]
    xs = np.arange(side)[None, :]
    ys = np.arange(side)[:, None]
    grid = ((xs % self.GRID_PERIOD_PX < self.GRID_LINE_PX) |
            (ys % self.GRID_PERIOD_PX < self.GRID_LINE_PX)).astype(
                np.float64) * self.GRID_WEIGHT
    return np.clip(image * self.NOISE_WEIGHT + grid, self.TILE_CLIP[0],
                   self.TILE_CLIP[1])

  def _render(self, base, h):
    """Sample ``base`` through homography ``h`` (backward map)."""
    width, height = self.size
    side = base.shape[0]
    h_inv = np.linalg.inv(h)
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones_like(xs)
    pts = np.stack([xs, ys, ones], axis=0).reshape(3, -1)
    mapped = h_inv @ pts
    u = mapped[0] / mapped[2]
    v = mapped[1] / mapped[2]
    # Center the tile on the frame-A footprint: tile pixel (x, y)
    # corresponds to ground-view coordinate (x - side/2, y - side/2).
    u = u + side / 2.0 - width / 2.0
    v = v + side / 2.0 - height / 2.0
    valid = (u >= 0) & (u < side - 1) & (v >= 0) & (v < side - 1)
    u = np.clip(u, 0, side - 2)
    v = np.clip(v, 0, side - 2)
    x0 = u.astype(np.int32)
    y0 = v.astype(np.int32)
    fx = u - x0
    fy = v - y0
    out = (base[y0, x0] * (1 - fx) * (1 - fy) + base[y0, x0 + 1] * fx * (1 - fy) +
           base[y0 + 1, x0] * (1 - fx) * fy + base[y0 + 1, x0 + 1] * fx * fy)
    out[~valid] = 0.0
    return out.reshape(height, width)

  def __getitem__(self, idx):
    rng = np.random.default_rng(self._seed + idx)
    base = self._base_tile(rng)
    height = float(rng.uniform(30.0, 80.0))
    motion = sample_motion(rng, self._motion_cfg)
    # Place camera A on the ray through the world origin: the optical
    # axis (pitch ``self.pitch`` down, yaw 45) hits the ground at the
    # origin, so the origin projects to the image center and every motion
    # of B moves content across the frame.  Frame B is A moved by the
    # sampled motion (translation in ground units, height change, yaw).
    pitch = self._pitch
    yaw0 = np.deg2rad(45.0)
    dist = np.array([
        np.cos(yaw0) * np.cos(pitch),
        np.sin(yaw0) * np.cos(pitch),
        -np.sin(pitch),
    ])
    cam_a = -height * dist
    rpy_a = np.array([0.0, pitch, yaw0])
    # Image-plane fracs -> ground units: one image width covers
    # W * height / fx ground units (approx, at nadir depth).
    width_units = self._size[0] * height / self._cam["fx"]
    # Ground translation in the (heading, perpendicular) frame; camera B
    # stays on the -dist ray so its footprint overlaps frame A's.
    ground_t = np.array([motion["dx"], motion["dy"], 0.0]) * width_units
    heading = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
    side = np.array([-np.sin(yaw0), np.cos(yaw0), 0.0])
    cam_b = -height * motion["ds"] * dist + ground_t[0] * heading \
        + ground_t[1] * side
    rpy_b = rpy_a + np.array([0.0, 0.0, motion["dyaw"]])
    h_a = look_at_ground_h(cam_a, rpy_a, self._cam, self._size)
    h_b = look_at_ground_h(cam_b, rpy_b, self._cam, self._size)
    frame_a = self._render(base, h_a)
    frame_b = self._render(base, h_b)
    sim, residual = homography_to_similarity(h_b @ np.linalg.inv(h_a), self._size)
    gt = {
        "log_s": torch.tensor(sim["log_s"]),
        "theta": torch.tensor(sim["theta"]),
        "t": torch.tensor(sim["t"]),
    }
    image_a = torch.tensor(frame_a, dtype=torch.float32).unsqueeze(0)
    image_b = torch.tensor(frame_b, dtype=torch.float32).unsqueeze(0)
    meta = VOPairMeta(gt=gt,
                      gt_residual=torch.tensor(residual, dtype=torch.float32),
                      terrain=self._terrain,
                      range_bin=int(
                          np.clip(np.searchsorted([0.1, 0.2, 0.3], abs(motion["dx"])),
                                  0, 3)))
    return {"image_a": image_a, "image_b": image_b, "meta": meta}
