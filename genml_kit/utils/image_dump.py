"""v2-compatible transform that optionally saves the input image to disk.

Usage::

    from genml_kit.utils.image_dump import ImageDump

    transform = v2.Compose([
        v2.RandomResizedCrop(...),
        ImageDump(save_dir="/tmp/debug", p=0.1, prefix="train"),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(...),
    ])

The transform always returns its input unchanged.  When triggered (with
probability *p*), the image is saved as a JPEG to *save_dir* with a
sequential filename.
"""

import os
import threading

import torch
import torchvision.transforms.v2 as v2
from PIL import Image


class ImageDump(v2.Transform):
  """Optionally save the input image to disk, always returning it unchanged.

  Parameters
  ----------
  save_dir : str
      Directory where images are saved.  Created automatically if it
      does not exist.
  p : float
      Probability of saving each image (0–1).
  prefix : str
      Filename prefix.  Final names follow the pattern
      ``{prefix}_{counter:06d}.jpg``.
  """

  _COUNTER = 0
  _LOCK = threading.Lock()

  def __init__(self, save_dir, p=0.1, prefix="dump"):
    super().__init__()
    self._save_dir = save_dir
    self._p = p
    self._prefix = prefix

  def _save(self, img):
    """Save a PIL image (or convert to one) and increment the counter."""
    if not os.path.isdir(self._save_dir):
      os.makedirs(self._save_dir, exist_ok=True)

    with ImageDump._LOCK:
      seq = ImageDump._COUNTER
      ImageDump._COUNTER += 1

    path = os.path.join(self._save_dir, f"{self._prefix}_{os.getpid()}_{seq:06d}.jpg")

    if isinstance(img, Image.Image):
      img.save(path)
    elif isinstance(img, torch.Tensor):
      self._save_tensor(img, path)

  def _save_tensor(self, tensor, path):
    """Convert a tensor to PIL and save.  Handles uint8 [0,255] and
    float32 [0,1] (or [0,255]) tensors."""
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
      # CHW → HWC
      t = tensor.permute(1, 2, 0)
    else:
      t = tensor

    # Move to CPU and convert to numpy.
    t = t.detach().cpu()

    if t.dtype.is_floating_point:
      # Treat as [0, 1] range.  Values outside are clipped.
      t = t.clamp(0.0, 1.0)
      # HWC float → uint8 [0,255]
      arr = (t.numpy() * 255).astype("uint8")
    else:
      # Already integer type (e.g. uint8).
      arr = t.numpy()

    # Single-channel: squeeze to (H, W) for PIL.
    if arr.ndim == 3 and arr.shape[2] == 1:
      arr = arr[:, :, 0]

    # JPEG doesn't support RGBA — drop alpha.
    if arr.ndim == 3 and arr.shape[2] == 4:
      arr = arr[:, :, :3]

    img = Image.fromarray(arr)
    img.save(path)

  def forward(self, *inputs):
    # Apply probability check.  self._p is evaluated once per call; if
    # the random draw misses, we skip saving entirely.
    if torch.rand(1).item() < self._p:
      # Extract the image from the (possibly nested) input.
      img = inputs[0] if len(inputs) == 1 else inputs
      self._save(img)
    return inputs[0] if len(inputs) == 1 else inputs

  def extra_repr(self):
    return f"save_dir={self._save_dir!r}, p={self._p}, prefix={self._prefix!r}"
