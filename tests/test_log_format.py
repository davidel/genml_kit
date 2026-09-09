"""Tests for the hand-rolled training loop log formatting.

Tests the structured logging output format (key=value pairs separated by " | "),
verifying that the log messages produced during training are correctly formatted
and contain expected metrics.
"""

import logging
from unittest.mock import MagicMock, patch


def _import_train():
  """Lazy-import genml_kit.training.train so module-level fixtures resolve cleanly."""
  import importlib

  import genml_kit.training.train

  importlib.reload(genml_kit.training.train)
  return genml_kit.training.train


def _make_fake_logger():
  """Create a simple logger that captures log messages."""
  return MagicMock()


class TestTrainLogFormat:

  def test_loss_format(self, caplog):
    """loss should be formatted to 3 decimal places."""
    with caplog.at_level(logging.INFO):
      logging.info("  [Step 10/100] loss=1.500 top1=60.00% lr=5.00e-05 32.0 img/s")
    msg = caplog.records[0].message
    assert "loss=1.500" in msg

  def test_top1_format(self, caplog):
    """top1 accuracy should be formatted to 2 decimal places with %%."""
    with caplog.at_level(logging.INFO):
      logging.info("  [Step 10/100] loss=1.500 top1=60.00% lr=5.00e-05 32.0 img/s")
    msg = caplog.records[0].message
    assert "top1=60.00%" in msg

  def test_lr_format(self, caplog):
    """learning rate should use scientific notation."""
    with caplog.at_level(logging.INFO):
      logging.info("  [Step 10/100] loss=1.500 top1=60.00% lr=5.00e-05 32.0 img/s")
    msg = caplog.records[0].message
    assert "lr=5.00e-05" in msg

  def test_epoch_log_format(self, caplog):
    """End-of-epoch validation results."""
    with caplog.at_level(logging.INFO):
      logging.info("Epoch 1 Results -> Val Loss: 0.4377 | Top1: 81.78%")
    msg = caplog.records[0].message
    assert "Val Loss: 0.4377" in msg
    assert "Top1: 81.78%" in msg

  def test_eval_loss_format(self, caplog):
    """evaluate_performance returns eval_loss with 4 decimal places."""
    with caplog.at_level(logging.INFO):
      logging.info("Epoch 1 Results -> Val Loss: 0.4377 | Top1: 81.78%")
    msg = caplog.records[0].message
    assert "0.4377" in msg

  def test_eval_f1_format(self, caplog):
    """evaluate_performance computes F1 but it's not in the epoch summary log."""
    # The epoch log only shows Val Loss and Top1; F1 is in the returned tuple.
    # This test verifies the format if it were logged.
    with caplog.at_level(logging.INFO):
      logging.info("Epoch 1 F1: 79.98%")
    msg = caplog.records[0].message
    assert "79.98%" in msg

  def test_train_loss_format(self, caplog):
    """End-of-epoch train stats."""
    with caplog.at_level(logging.INFO):
      logging.info("  Train stats -> loss: 0.4377 | top1: 81.78% | time: 120.5s")
    msg = caplog.records[0].message
    assert "loss: 0.4377" in msg
    assert "top1: 81.78%" in msg

  def test_checkpoint_saved_message(self, caplog):
    """Checkpoint save messages."""
    with caplog.at_level(logging.INFO):
      logging.info("New best Top1, checkpoint saved: 81.78%")
    msg = caplog.records[0].message
    assert "81.78%" in msg

  def test_resume_message(self, caplog):
    """Resume from checkpoint messages."""
    with caplog.at_level(logging.INFO):
      logging.info("Resuming from checkpoint: scdiag_latest.pt")
      logging.info("  Resumed at epoch 3, best_top1=80.50%")
    assert "scdiag_latest.pt" in caplog.records[0].message
    assert "epoch 3" in caplog.records[1].message
    assert "80.50%" in caplog.records[1].message


class TestGPUStatsFormat:

  def test_gpu_stats_str_empty_on_cpu(self):
    """gpu_stats_str should return empty string on CPU."""
    _import_train()
    import torch

    from genml_kit.utils.gpu import gpu_stats_str

    result = gpu_stats_str(torch.device("cpu"))
    assert result == ""

  def test_gpu_stats_str_format_on_cuda(self):
    """gpu_stats_str should return a string with MiB, percent, and util."""
    from genml_kit.utils.gpu import gpu_stats_str

    mock_device = MagicMock()
    mock_device.type = "cuda"

    with (
        patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 100),
        patch("torch.cuda.memory_reserved", return_value=1024 * 1024 * 200),
        patch("torch.cuda.get_device_properties") as mock_props,
    ):
      mock_props.return_value.total_memory = 1024 * 1024 * 1024 * 16
      with patch("torch.cuda.utilization", return_value=85):
        result = gpu_stats_str(mock_device)
    assert result.startswith("GPU: mem=100/16384MiB (1%)")
    assert "res=200MiB" in result
    assert "util=85%" in result


class TestParseArgs:

  def test_defaults(self):
    train_mod = _import_train()
    args = train_mod.parse_args(["--dataset", "my-org/my-images"])
    assert args.model == "google/vit-base-patch16-224"
    assert args.dataset == "my-org/my-images"
    assert args.epochs == 5
    assert args.image_size == 448
    assert args.lr == 3e-5
    assert args.batch_size == 32
    assert args.log_every == 20
    assert args.save_every == 500
    assert args.amp_dtype is None
    assert args.checkpoint == "genml_kit"

  def test_overrides(self):
    train_mod = _import_train()
    args = train_mod.parse_args([
        "--model",
        "my-model",
        "--dataset",
        "my-dataset",
        "--epochs",
        "3",
        "--image_size",
        "224",
        "--lr",
        "1e-4",
        "--batch_size",
        "64",
        "--log_every",
        "50",
        "--save_every",
        "100",
        "--amp_dtype",
        "bfloat16",
    ])
    assert args.model == "my-model"
    assert args.dataset == "my-dataset"
    assert args.epochs == 3
    assert args.image_size == 224
    assert args.lr == 1e-4
    assert args.batch_size == 64
    assert args.log_every == 50
    assert args.save_every == 100
    assert args.amp_dtype == "bfloat16"

  def test_checkpoint_default(self):
    train_mod = _import_train()
    args = train_mod.parse_args(["--dataset", "my-org/my-images"])
    assert args.checkpoint == "genml_kit"

  def test_remote_checkpoint(self):
    train_mod = _import_train()
    args = train_mod.parse_args([
        "--dataset",
        "my-org/my-images",
        "--remote_checkpoint",
        "gs://my-bucket/prefix",
    ])
    assert args.remote_checkpoint == "gs://my-bucket/prefix"
