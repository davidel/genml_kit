"""Tests for external-script model loading (load_model / load_processor)."""

import os
import tempfile

import pytest
import torch

from genml_kit.models import load_model, load_processor


class TestScriptModel:

  def test_load_model_script(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "script.py")
      with open(path, "w") as f:
        f.write("import torch.nn as nn\n"
                "class Tiny(nn.Module):\n"
                "  def __init__(self, num_labels):\n"
                "    super().__init__()\n"
                "    self.head = nn.Linear(4, num_labels)\n"
                "  def forward(self, pixel_values):\n"
                "    return self.head(pixel_values)\n"
                "def create_model(num_labels, image_size=224, id2label=None,\n"
                "                 label2id=None, checkpoint_path=None, device='cpu',\n"
                "                 **kwargs):\n"
                "  return Tiny(num_labels=num_labels)\n")
      model = load_model(path, num_labels=5)
      assert isinstance(model, torch.nn.Module)
      assert model.head.out_features == 5

  def test_load_model_script_missing_fn(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "script.py")
      with open(path, "w") as f:
        f.write("def not_create_model():\n    pass\n")
      with pytest.raises(ValueError):
        load_model(path, num_labels=3)

  def test_load_model_script_checkpoint_passthrough(self):
    # The script is fully responsible for weights: it must receive
    # checkpoint_path and may use load_checkpoint_weights itself
    # (mirroring the ConvViT/UViTO loaders).  Here we assert the arg
    # is forwarded, and that a non-None value triggers the script's
    # own load branch.
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "script.py")
      with open(path, "w") as f:
        f.write("import torch\n"
                "import torch.nn as nn\n"
                "class Tiny(nn.Module):\n"
                "  def __init__(self):\n"
                "    super().__init__()\n"
                "    self.w = nn.Parameter(torch.zeros(1))\n"
                "def create_model(num_labels, image_size=224, id2label=None,\n"
                "                 label2id=None, checkpoint_path=None, device='cpu',\n"
                "                 **kwargs):\n"
                "  m = Tiny()\n"
                "  if checkpoint_path:\n"
                "    m.w.data.fill_(float(checkpoint_path))\n"
                "  return m\n")
      # checkpoint_path='3.0' -> the script sets w to 3.0.
      model = load_model(path, num_labels=1, checkpoint_path="3.0")
      assert model.w.item() == 3.0
      # checkpoint_path=None -> random init (w stays 0 as created).
      model2 = load_model(path, num_labels=1, checkpoint_path=None)
      assert model2.w.item() == 0.0


class TestScriptProcessor:

  def test_load_processor_script(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "script.py")
      with open(path, "w") as f:
        f.write("class Proc:\n"
                "  def __init__(self, image_size):\n"
                "    self.image_size = image_size\n"
                "    self.image_mean = [0.5, 0.5, 0.5]\n"
                "    self.image_std = [0.5, 0.5, 0.5]\n"
                "def create_processor(image_size=224, **kwargs):\n"
                "  return Proc(image_size=image_size)\n")
      proc = load_processor(path, image_size=448)
      assert proc.image_size == 448
      assert proc.image_mean == [0.5, 0.5, 0.5]

  def test_load_processor_script_missing_fn(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "script.py")
      with open(path, "w") as f:
        f.write("def nope():\n    pass\n")
      with pytest.raises(ValueError):
        load_processor(path)
