"""Tests for genml_kit.optim_factory — optimizer and scheduler factories."""

import pytest
import torch.nn as nn
import torch.optim as optim

from genml_kit.training.optim_factory import (
    build_param_groups,
    build_param_groups_llrd,
    compute_params_depths,
    create_optimizer,
    create_scheduler,
    report_lr,
)


class TestBuildParamGroups:
  """Tests for build_param_groups."""

  def _make_model(self):
    """Create a small model with named params for testing."""

    class Toy(nn.Module):

      def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(10, 8)
        self.classifier = nn.Linear(8, 2)

      def forward(self, x):
        return self.classifier(self.backbone(x))

    return Toy()

  def test_no_groups_returns_single_group(self):
    model = self._make_model()
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params, lr=1e-3, weight_decay=0.01)
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3
    assert groups[0]["weight_decay"] == 0.01
    assert len(groups[0]["params"]) == 4  # 2 weight + 2 bias

  def test_regex_matching(self):
    model = self._make_model()
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params,
                                lr=1e-3,
                                weight_decay=0.01,
                                lr_groups=["backbone.*=1e-5"])
    assert len(groups) == 2
    backbone_params = [n for n in named_params if "backbone" in n]
    classifier_params = [n for n in named_params if "classifier" in n]
    # First group should be backbone
    assert groups[0]["lr"] == 1e-5
    assert len(groups[0]["params"]) == len(backbone_params)
    # Second group should be classifier (fallback)
    assert groups[1]["lr"] == 1e-3
    assert len(groups[1]["params"]) == len(classifier_params)

  def test_fallback_to_default_lr(self):
    model = self._make_model()
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params,
                                lr=1e-3,
                                weight_decay=0.01,
                                lr_groups=["backbone.*=1e-5"])
    # Classifier params should fall back to default lr
    assert groups[1]["lr"] == 1e-3

  def test_first_match_wins(self):
    model = self._make_model()
    named_params = dict(model.named_parameters())
    # Overlapping regexes: backbone.*weight matches only backbone.weight,
    # backbone.* matches both backbone.weight and backbone.bias.
    # backbone.weight goes to the first regex (first match wins).
    groups = build_param_groups(named_params,
                                lr=1e-3,
                                weight_decay=0.01,
                                lr_groups=["backbone.*weight=1e-5", "backbone.*=1e-6"])
    # Group 0: backbone.weight (matched first regex)
    assert groups[0]["lr"] == 1e-5
    assert len(groups[0]["params"]) == 1
    # Group 1: backbone.bias (second regex, backbone.*)
    assert groups[1]["lr"] == 1e-6
    assert len(groups[1]["params"]) == 1

  def test_frozen_params_excluded(self):
    model = self._make_model()
    # Freeze classifier
    for p in model.classifier.parameters():
      p.requires_grad = False
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params, lr=1e-3, weight_decay=0.01)
    # Only backbone params should be in the group
    assert len(groups[0]["params"]) == 2  # weight + bias

  def test_unmatched_regex_raises(self):
    model = self._make_model()
    named_params = dict(model.named_parameters())
    with pytest.raises(ValueError, match="matched no parameters"):
      build_param_groups(named_params,
                         lr=1e-3,
                         weight_decay=0.01,
                         lr_groups=["nonexistent.*=1e-3"])

  def test_weight_decay_applied_to_all_groups(self):
    model = self._make_model()
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params,
                                lr=1e-3,
                                weight_decay=0.05,
                                lr_groups=["backbone.*=1e-5", "classifier.*=1e-3"])
    for g in groups:
      assert g["weight_decay"] == 0.05

  def test_empty_params_returns_empty_list(self):
    model = self._make_model()
    # Freeze all params
    for p in model.parameters():
      p.requires_grad = False
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params, lr=1e-3, weight_decay=0.01)
    assert groups == []

  def test_multiple_groups_with_fallback(self):
    model = self._make_model()
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params,
                                lr=1e-3,
                                weight_decay=0.01,
                                lr_groups=["backbone.*=1e-5"])
    # Should have 2 groups: backbone (regex) and classifier (fallback)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["lr"] == 1e-3


class TestCreateOptimizer:
  """Tests for the optimizer factory."""

  @pytest.fixture
  def params(self):
    return nn.Linear(10, 2).parameters()

  def test_default_is_adamw(self, params):
    opt = create_optimizer(params)
    assert isinstance(opt, optim.AdamW)

  def test_adam(self, params):
    opt = create_optimizer(params, name="Adam")
    assert isinstance(opt, optim.Adam)

  def test_sgd(self, params):
    opt = create_optimizer(params, name="SGD")
    assert isinstance(opt, optim.SGD)

  def test_exact_match(self, params):
    opt = create_optimizer(params, name="AdamW")
    assert isinstance(opt, optim.AdamW)

  def test_case_sensitive_rejects_lowercase(self, params):
    with pytest.raises(ValueError, match="Unknown optimizer"):
      create_optimizer(params, name="adamw")

  def test_lr_forwarded(self, params):
    opt = create_optimizer(params, lr=0.123)
    assert opt.param_groups[0]["lr"] == 0.123

  def test_weight_decay_forwarded(self, params):
    opt = create_optimizer(params, weight_decay=0.5)
    assert opt.param_groups[0]["weight_decay"] == 0.5

  def test_extra_kwargs_forwarded(self, params):
    opt = create_optimizer(params, name="SGD", momentum=0.9)
    assert opt.param_groups[0]["momentum"] == 0.9

  def test_extra_kwargs_adamw_betas(self, params):
    opt = create_optimizer(params, betas=(0.9, 0.999))
    assert opt.param_groups[0]["betas"] == (0.9, 0.999)

  def test_unknown_optimizer_raises(self, params):
    with pytest.raises(ValueError, match="Unknown optimizer"):
      create_optimizer(params, name="nonexistent")

  def test_param_groups_input(self):
    """Test with param groups dict from build_param_groups."""
    model = nn.Sequential(nn.Linear(10, 8), nn.Linear(8, 2))
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params,
                                lr=1e-3,
                                weight_decay=0.01,
                                lr_groups=["0.*=1e-5", "1.*=1e-3"])
    opt = create_optimizer(groups, name="AdamW")
    assert isinstance(opt, optim.AdamW)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-5
    assert opt.param_groups[1]["lr"] == 1e-3

  def test_initial_lr_seeded(self):
    """Test that initial_lr is set for each param group."""
    model = nn.Sequential(nn.Linear(10, 8), nn.Linear(8, 2))
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params,
                                lr=1e-3,
                                weight_decay=0.01,
                                lr_groups=["0.*=1e-5", "1.*=1e-3"])
    opt = create_optimizer(groups, name="AdamW")
    for g in opt.param_groups:
      assert g["initial_lr"] == g["lr"]


class TestCreateScheduler:
  """Tests for the scheduler factory."""

  @pytest.fixture
  def optimizer(self):
    model = nn.Linear(10, 2)
    return optim.AdamW(model.parameters(), lr=1e-3)

  def test_default_returns_none(self, optimizer):
    sched = create_scheduler(optimizer)
    assert sched is None

  def test_none_returns_none(self, optimizer):
    sched = create_scheduler(optimizer, name=None)
    assert sched is None

  def test_cosine(self, optimizer):
    sched = create_scheduler(optimizer,
                             name="CosineAnnealingLR",
                             T_max=50,
                             eta_min=1e-5)
    assert isinstance(sched, optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 50
    assert sched.eta_min == pytest.approx(1e-5)

  def test_cosine_requires_t_max(self, optimizer):
    with pytest.raises(TypeError):
      create_scheduler(optimizer, name="CosineAnnealingLR")

  def test_step(self, optimizer):
    sched = create_scheduler(optimizer, name="StepLR", step_size=10, gamma=0.5)
    assert isinstance(sched, optim.lr_scheduler.StepLR)
    assert sched.step_size == 10
    assert sched.gamma == 0.5

  def test_case_sensitive_rejects_lowercase(self, optimizer):
    with pytest.raises(ValueError, match="Unknown scheduler"):
      create_scheduler(optimizer, name="cosine")

  def test_unknown_scheduler_raises(self, optimizer):
    with pytest.raises(ValueError, match="Unknown scheduler"):
      create_scheduler(optimizer, name="nonexistent")

  def test_custom_scheduler_script(self, optimizer, tmp_path):
    script = tmp_path / "my_sched.py"
    script.write_text("import torch.optim.lr_scheduler as s\n"
                      "def create_scheduler(optimizer, **kw):\n"
                      "    return s.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)\n")
    sched = create_scheduler(optimizer, name=str(script))
    assert isinstance(sched, optim.lr_scheduler.LambdaLR)

  def test_custom_scheduler_missing_fn(self, optimizer, tmp_path):
    script = tmp_path / "bad_sched.py"
    script.write_text("x = 42\n")
    with pytest.raises(ValueError, match="does not define"):
      create_scheduler(optimizer, name=str(script))


class TestCreateOptimizerScript:
  """Tests for create_optimizer with custom .py script dispatch."""

  def test_custom_optimizer_script(self, tmp_path):
    model = nn.Linear(10, 2)
    params = list(model.parameters())
    script = tmp_path / "my_opt.py"
    script.write_text("import torch.optim as o\n"
                      "def create_optimizer(params, **kw):\n"
                      "    return o.SGD(params, lr=kw.get('lr', 0.1))\n")
    # lr and weight_decay are forwarded to scripts, so kw['lr'] = 0.01.
    opt = create_optimizer(params, name=str(script), lr=0.01)
    assert isinstance(opt, optim.SGD)
    assert opt.param_groups[0]["lr"] == 0.01

  def test_custom_optimizer_script_param_groups(self, tmp_path):
    model = nn.Sequential(nn.Linear(10, 8), nn.Linear(8, 2))
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params,
                                lr=1e-3,
                                weight_decay=0.01,
                                lr_groups=["0.*=1e-5", "1.*=1e-3"])
    script = tmp_path / "my_opt.py"
    script.write_text("import torch.optim as o\n"
                      "def create_optimizer(params, **kw):\n"
                      "    return o.AdamW(params, lr=0.001)\n")
    opt = create_optimizer(groups, name=str(script))
    assert isinstance(opt, optim.AdamW)
    assert len(opt.param_groups) == 2

  def test_custom_optimizer_missing_fn(self, tmp_path):
    script = tmp_path / "bad_opt.py"
    script.write_text("x = 42\n")
    model = nn.Linear(10, 2)
    with pytest.raises(ValueError, match="does not define"):
      create_optimizer(list(model.parameters()), name=str(script))

  def test_script_opt_arg_forwarded(self, tmp_path):
    model = nn.Linear(10, 2)
    script = tmp_path / "my_opt.py"
    # Script reads lr, weight_decay, and momentum from **kwargs.
    script.write_text(
        "import torch.optim as o\n"
        "def create_optimizer(params, **kw):\n"
        "    return o.SGD(params, lr=kw['lr'], momentum=kw['momentum'])\n")
    # lr and weight_decay are forwarded by the factory.
    # Extra --opt_arg values (momentum) are forwarded too.
    opt = create_optimizer(
        list(model.parameters()),
        name=str(script),
        lr=0.05,
        momentum=0.9,
    )
    assert isinstance(opt, optim.SGD)
    assert opt.param_groups[0]["lr"] == 0.05  # forwarded from lr=
    assert opt.param_groups[0]["momentum"] == 0.9


class TestReportLr:
  """Tests for report_lr."""

  def test_single_group(self):
    model = nn.Linear(10, 2)
    opt = optim.AdamW(model.parameters(), lr=0.001)
    result = report_lr(opt)
    assert result == "lr=1.00e-03"

  def test_multi_group(self):
    model = nn.Sequential(nn.Linear(10, 8), nn.Linear(8, 2))
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params,
                                lr=1e-3,
                                weight_decay=0.01,
                                lr_groups=["0.*=1e-5", "1.*=1e-3"])
    opt = create_optimizer(groups, name="AdamW")
    result = report_lr(opt)
    assert result == "lr=[1.00e-05, 1.00e-03]"

  def test_tensorboard_single_group(self):
    model = nn.Linear(10, 2)
    opt = optim.AdamW(model.parameters(), lr=0.001)

    class FakeWriter:

      def __init__(self):
        self.scalars = {}

      def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)

    writer = FakeWriter()
    result = report_lr(opt, writer=writer, step=42)
    assert result == "lr=1.00e-03"
    assert writer.scalars["Train/lr"] == (0.001, 42)

  def test_tensorboard_multi_group(self):
    model = nn.Sequential(nn.Linear(10, 8), nn.Linear(8, 2))
    named_params = dict(model.named_parameters())
    groups = build_param_groups(named_params,
                                lr=1e-3,
                                weight_decay=0.01,
                                lr_groups=["0.*=1e-5", "1.*=1e-3"])
    opt = create_optimizer(groups, name="AdamW")

    class FakeWriter:

      def __init__(self):
        self.scalars = {}

      def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)

    writer = FakeWriter()
    result = report_lr(opt, writer=writer, step=42)
    assert result == "lr=[1.00e-05, 1.00e-03]"
    assert writer.scalars["Train/lr_group0"] == (1e-5, 42)
    assert writer.scalars["Train/lr_group1"] == (1e-3, 42)


class TestComputeParamsDepths:
  """Tests for compute_params_depths."""

  def test_basic_transformer(self):
    names = [
        "encoder.layer.0.self_attn.q_proj.weight",
        "encoder.layer.0.self_attn.k_proj.weight",
        "encoder.layer.1.self_attn.q_proj.weight",
        "encoder.layer.1.self_attn.k_proj.weight",
        "encoder.layer.11.self_attn.q_proj.weight",
        "encoder.layer.11.self_attn.k_proj.weight",
        "head.fc.weight",
        "head.fc.bias",
    ]
    depths = compute_params_depths(names)
    # All params in layer 0 share the same depth.
    assert depths["encoder.layer.0.self_attn.q_proj.weight"] == \
           depths["encoder.layer.0.self_attn.k_proj.weight"]
    # All params in layer 1 share the same depth.
    assert depths["encoder.layer.1.self_attn.q_proj.weight"] == \
           depths["encoder.layer.1.self_attn.k_proj.weight"]
    # Layer 11 is deeper than layer 1.
    assert depths["encoder.layer.11.self_attn.q_proj.weight"] > \
           depths["encoder.layer.1.self_attn.q_proj.weight"]
    # Non-block params (head) get depth 0.
    assert depths["head.fc.weight"] == 0
    assert depths["head.fc.bias"] == 0

  def test_no_numeric_keys(self):
    names = ["layer.weight", "layer.bias"]
    depths = compute_params_depths(names)
    # All at depth 0 since there are no numeric segments.
    assert all(d == 0 for d in depths.values())

  def test_empty(self):
    assert compute_params_depths([]) == {}


class TestBuildParamGroupsLlrd:
  """Tests for build_param_groups_llrd."""

  def _make_model(self):
    """Create a small model with named params for testing."""

    class Toy(nn.Module):

      def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict({
            "layer_0": nn.Linear(4, 4),
            "layer_1": nn.Linear(4, 4),
        })
        self.head = nn.Linear(4, 2)

    return Toy()

  def test_correct_number_of_groups(self):
    model = self._make_model()
    groups = build_param_groups_llrd(
        dict(model.named_parameters()),
        lr=1e-3,
        weight_decay=0.01,
        decay_factor=0.85,
    )
    # Each depth level produces up to 2 groups (with/without weight decay).
    # Verify no empty param lists.
    for g in groups:
      assert len(g["params"]) > 0

  def test_weight_decay_split(self):
    model = self._make_model()
    groups = build_param_groups_llrd(
        dict(model.named_parameters()),
        lr=1e-3,
        weight_decay=0.01,
        decay_factor=0.85,
    )
    for g in groups:
      if g["weight_decay"] == 0.0:
        # All params in this group should be 1-D (bias).
        for p in g["params"]:
          assert p.ndim == 1
      elif g["weight_decay"] == 0.01:
        # All params in this group should be >= 2-D (weight).
        for p in g["params"]:
          assert p.ndim > 1

  def test_lr_values_ordered(self):
    model = self._make_model()
    groups = build_param_groups_llrd(
        dict(model.named_parameters()),
        lr=1e-3,
        weight_decay=0.01,
        decay_factor=0.85,
    )
    # Collect unique (depth, lr) pairs.
    depth_lr = {}
    for g in groups:
      depth = g["depth"]
      lr = g["lr"]
      if depth not in depth_lr:
        depth_lr[depth] = lr
      else:
        # Same depth must have same lr.
        assert depth_lr[depth] == lr

    # Deeper levels should have higher lr (closer to base lr).
    depths_sorted = sorted(depth_lr.keys())
    for i in range(1, len(depths_sorted)):
      assert depth_lr[depths_sorted[i]] > depth_lr[depths_sorted[i - 1]]

  def test_decay_factor_one_means_equal_lr(self):
    model = self._make_model()
    groups = build_param_groups_llrd(
        dict(model.named_parameters()),
        lr=1e-3,
        weight_decay=0.01,
        decay_factor=1.0,
    )
    lrs = {g["lr"] for g in groups}
    # With decay_factor=1.0, all groups should have the same lr.
    assert len(lrs) == 1
    assert 1e-3 in lrs

  def test_optimizable(self):
    model = self._make_model()
    groups = build_param_groups_llrd(
        dict(model.named_parameters()),
        lr=1e-3,
        weight_decay=0.01,
        decay_factor=0.85,
    )
    opt = optim.AdamW(groups)
    assert len(opt.param_groups) == len(groups)
