"""End-to-end VO smoke test: dataset -> model -> loss -> eval -> registry."""

import torch

from genml_kit.datasets.vo_pairs import VOPairDataset
from genml_kit.models.registry import is_custom_model, load_model
from genml_kit.training.vo.eval_vo import evaluate_sliced
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
    pred = {
        "params": out["params"],
        "conf": out["conf"],
    }
    batch = dict(batch)
    batch["corners"] = out["corners"]
    batch["dc"] = out["dc"]
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
  assert metrics.mce == metrics.mce  # finite (not NaN)


def test_evaluate_sliced_groups_by_terrain():
  dataset = VOPairDataset(length=6, size=(64, 64), seed=2, terrain="field")
  bundle = load_model("vo/npu-small", num_labels=0, image_size=64)
  rows, table = evaluate_sliced(bundle.model, dataset, torch.device("cpu"))
  assert len(rows) == 1
  assert rows[0].slice == "field"
  assert rows[0].n == 6
  assert any("slice" in line for line in table)


def test_registry_knows_the_vo_models():
  for name in ("vo/npu-small", "vo/edge-mid", "vo/station"):
    assert is_custom_model(name)


def test_registry_loads_all_profiles():
  for name, in_ch in (("vo/npu-small", 1), ("vo/edge-mid", 1), ("vo/station", 3)):
    bundle = load_model(name, num_labels=0, image_size=64, in_ch=in_ch)
    out = bundle.model(torch.zeros(1, in_ch, 64, 64), torch.zeros(1, in_ch, 64, 64))
    assert out["params"].log_s.shape == (1,)
