"""End-to-end VO smoke test: dataset -> model -> loss -> eval -> registry."""

import pytest
import torch

from genml_kit.datasets.vo_pairs import VOPairDataset
from genml_kit.models.registry import is_custom_model, load_model
from genml_kit.training.vo.eval_vo import evaluate_sliced
from genml_kit.methods.vo_pair import VOPrediction
from genml_kit.training.vo.train_vo import STAGES, evaluate_vo, vo_losses


class _Cfg:
  w_log_s = 1.0
  w_theta = 1.0
  w_conf = 0.5
  w_photo = 0.1


def _collate(items):
  return torch.utils.data.default_collate(items)


def test_full_pipeline_trains_a_few_steps():
  dataset = VOPairDataset(length=8, size=(64, 64), seed=0)
  loader = torch.utils.data.DataLoader(dataset, batch_size=4)
  bundle = load_model("vo/npu-small", num_labels=0, image_size=64)
  model = bundle.model
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  for batch in loader:
    out = model(batch["image_a"], batch["image_b"])
    pred = VOPrediction(params=out.params, conf=out.conf)
    batch = dict(batch)
    batch["corners"] = out.corners
    batch["dc"] = out.dc
    loss, parts = vo_losses(pred, batch, _Cfg(), STAGES["supervised"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  for param in model.parameters():
    assert param.grad is None or torch.all(torch.isfinite(param.grad))


def test_evaluate_vo_runs():
  dataset = VOPairDataset(length=4, size=(64, 64), seed=1)
  loader = torch.utils.data.DataLoader(dataset, batch_size=2)
  bundle = load_model("vo/npu-small", num_labels=0, image_size=64)
  metrics = evaluate_vo(bundle.model, loader, torch.device("cpu"))
  # finite (not NaN)
  assert metrics.mce == metrics.mce


def _predictor(model, device):
  """item -> VOModelOutput with a leading batch of 1 (per-item call)."""

  def predict(item):
    with torch.no_grad():
      out = model(item["image_a"].unsqueeze(0).to(device),
                  item["image_b"].unsqueeze(0).to(device))
    return out

  return predict


def test_evaluate_sliced_groups_by_terrain():
  dataset = VOPairDataset(length=6, size=(64, 64), seed=2, terrain="field")
  bundle = load_model("vo/npu-small", num_labels=0, image_size=64)
  bundle.model.eval()
  rows, table = evaluate_sliced(dataset,
                                _predictor(bundle.model, torch.device("cpu")),
                                image_size=64)
  assert len(rows) == 1
  assert rows[0].slice == "field"
  assert rows[0].n == 6
  assert any("slice" in line for line in table)


def test_evaluate_sliced_error_columns_are_not_nan():
  """Regression: the error columns used to be dead (always NaN).

  The old signature took a model but never used it, and the helpers
  guarded on ``"pred" not in item``, which is never true for
  VOPairDataset items -- so dlog_s/dtheta/conf_mae always came back NaN.
  """
  dataset = VOPairDataset(length=4, size=(64, 64), seed=3, terrain="field")
  bundle = load_model("vo/npu-small", num_labels=0, image_size=64)
  bundle.model.eval()
  rows, _ = evaluate_sliced(dataset,
                            _predictor(bundle.model, torch.device("cpu")),
                            image_size=64)
  for row in rows:
    for value in (row.mce, row.dlog_s, row.dtheta, row.conf_mae, row.gt_fit_mce):
      assert value == value, "slice metric is NaN"


def test_evaluate_sliced_mce_matches_evaluate_vo():
  """A single-slice table must agree with the aggregate evaluate_vo mce.

  Pins that both evaluation paths share one metric implementation.
  """
  dataset = VOPairDataset(length=4, size=(64, 64), seed=4, terrain="field")
  bundle = load_model("vo/npu-small", num_labels=0, image_size=64)
  bundle.model.eval()
  loader = torch.utils.data.DataLoader(dataset, batch_size=2)
  aggregate = evaluate_vo(bundle.model, loader, torch.device("cpu"))
  rows, _ = evaluate_sliced(dataset,
                            _predictor(bundle.model, torch.device("cpu")),
                            image_size=64)
  assert rows[0].mce == pytest.approx(aggregate.mce, rel=1e-4, abs=1e-4)


def test_evaluate_sliced_handles_int_group_keys():
  """Regression: an int slice key (range_bin) must not crash format_table.

  format_table calls len() on every cell, so a non-str key raised
  TypeError.  The int-valued range_bin path had no test coverage.
  """
  dataset = VOPairDataset(length=8, size=(64, 64), seed=6, terrain="field")
  bundle = load_model("vo/npu-small", num_labels=0, image_size=64)
  bundle.model.eval()
  rows, table = evaluate_sliced(dataset,
                                _predictor(bundle.model, torch.device("cpu")),
                                group_key="range_bin",
                                image_size=64)
  assert rows
  assert all(isinstance(row.slice, int) for row in rows)
  assert len(table) == len(rows) + 2  # header + separator


def test_evaluate_sliced_gt_fit_scales_with_image_size():
  """The GT-fit column must follow image_size, not a hardcoded 64px.

  At a larger frame the same translation covers more pixels, so the
  reported GT-fit magnitude has to grow proportionally.
  """
  bundle = load_model("vo/npu-small", num_labels=0, image_size=64)
  bundle.model.eval()
  predict = _predictor(bundle.model, torch.device("cpu"))
  small, _ = evaluate_sliced(VOPairDataset(length=3,
                                           size=(64, 64),
                                           seed=5,
                                           terrain="field"),
                             predict,
                             image_size=64)
  large, _ = evaluate_sliced(VOPairDataset(length=3,
                                           size=(128, 128),
                                           seed=5,
                                           terrain="field"),
                             predict,
                             image_size=128)
  assert large[0].gt_fit_mce > small[0].gt_fit_mce


def test_registry_knows_the_vo_models():
  for name in ("vo/npu-small", "vo/edge-mid", "vo/station"):
    assert is_custom_model(name)


def test_registry_loads_all_profiles():
  for name, in_ch in (("vo/npu-small", 1), ("vo/edge-mid", 1), ("vo/station", 3)):
    bundle = load_model(name, num_labels=0, image_size=64, in_ch=in_ch)
    out = bundle.model(torch.zeros(1, in_ch, 64, 64), torch.zeros(1, in_ch, 64, 64))
    assert out.params.log_s.shape == (1,)
