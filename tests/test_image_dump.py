"""Tests for genml_kit.image_dump.ImageDump."""

import os

import torch
from PIL import Image

from genml_kit.utils.image_dump import ImageDump


class TestImageDump:
  """Test the ImageDump v2 transform."""

  def test_saves_pil_image(self, tmp_path):
    dump = ImageDump(save_dir=str(tmp_path), p=1.0, prefix="test")
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))

    result = dump(img)

    # Must return the input unchanged.
    assert result is img
    files = list(tmp_path.glob("test_*.jpg"))
    assert len(files) == 1
    with Image.open(files[0]) as saved:
      assert saved.size == (64, 64)

  def test_saves_uint8_tensor(self, tmp_path):
    dump = ImageDump(save_dir=str(tmp_path), p=1.0, prefix="uint8")
    # CHW uint8 [0, 255]
    tensor = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)

    result = dump(tensor)

    assert result is tensor
    files = list(tmp_path.glob("uint8_*.jpg"))
    assert len(files) == 1

  def test_saves_float32_tensor(self, tmp_path):
    dump = ImageDump(save_dir=str(tmp_path), p=1.0, prefix="f32")
    # CHW float32 [0.0, 1.0]
    tensor = torch.rand(3, 32, 32)

    result = dump(tensor)

    assert result is tensor
    files = list(tmp_path.glob("f32_*.jpg"))
    assert len(files) == 1

  def test_does_not_save_when_p_is_zero(self, tmp_path):
    dump = ImageDump(save_dir=str(tmp_path), p=0.0)
    img = Image.new("RGB", (16, 16))

    dump(img)

    assert len(list(tmp_path.iterdir())) == 0

  def test_counter_is_sequential(self, tmp_path):
    # Reset class-level counter so test is deterministic.
    ImageDump._COUNTER = 0
    pid = os.getpid()
    dump1 = ImageDump(save_dir=str(tmp_path), p=1.0, prefix="seq")
    dump2 = ImageDump(save_dir=str(tmp_path), p=1.0, prefix="seq")
    img = Image.new("RGB", (16, 16))

    dump1(img)
    dump1(img)
    dump2(img)

    files = sorted(tmp_path.glob("seq_*.jpg"))
    assert len(files) == 3
    assert files[0].name == f"seq_{pid}_000000.jpg"
    assert files[1].name == f"seq_{pid}_000001.jpg"
    assert files[2].name == f"seq_{pid}_000002.jpg"

  def test_creates_directory_if_missing(self, tmp_path):
    target = tmp_path / "new_dir"
    dump = ImageDump(save_dir=str(target), p=1.0)
    img = Image.new("RGB", (16, 16))

    dump(img)

    assert target.is_dir()
    assert len(list(target.glob("*.jpg"))) == 1

  def test_grayscale_tensor(self, tmp_path):
    dump = ImageDump(save_dir=str(tmp_path), p=1.0, prefix="gray")
    # Single channel CHW
    tensor = torch.randint(0, 256, (1, 32, 32), dtype=torch.uint8)

    result = dump(tensor)

    assert result is tensor
    assert len(list(tmp_path.glob("gray_*.jpg"))) == 1

  def test_rgba_tensor(self, tmp_path):
    dump = ImageDump(save_dir=str(tmp_path), p=1.0, prefix="rgba")
    tensor = torch.randint(0, 256, (4, 32, 32), dtype=torch.uint8)

    result = dump(tensor)

    assert result is tensor
    assert len(list(tmp_path.glob("rgba_*.jpg"))) == 1

  def test_extra_repr(self):
    dump = ImageDump(save_dir="/tmp/test", p=0.25, prefix="dbg")
    r = dump.extra_repr()
    assert "/tmp/test" in r
    assert "p=0.25" in r
    assert "dbg" in r
