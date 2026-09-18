"""Tests for RL CLI wiring (no option collisions, help renders)."""

import argparse

import pytest

from genml_kit.methods import get_method, list_methods
from genml_kit.pipelines import list_pipelines
from genml_kit.methods.rl_dqn import DQNMethod
from genml_kit.methods.rl_ppo import PPOMethod
from genml_kit.methods.rl_sac import SACMethod
from genml_kit.pipelines.rl import RLPipeline


class TestRLRegistration:
  """Verify RL components are registered and don't collide."""

  def test_dqn_registered(self):
    assert "dqn" in list_methods()
    assert get_method("dqn") is DQNMethod

  def test_ppo_registered(self):
    assert "ppo" in list_methods()
    assert get_method("ppo") is PPOMethod

  def test_sac_registered(self):
    assert "sac" in list_methods()
    assert get_method("sac") is SACMethod

  def test_rl_pipeline_registered(self):
    assert "rl" in list_pipelines()

  def test_no_name_collision(self):
    methods = list_methods()
    pipelines = list_pipelines()
    for name in ("dqn", "ppo", "sac"):
      assert methods.count(name) == 1
    assert pipelines.count("rl") == 1


class TestRLCLIHelp:
  """Verify --help renders without error for RL config."""

  def test_dqn_help_renders(self):
    parser = argparse.ArgumentParser()
    DQNMethod.add_args(parser)
    with pytest.raises(SystemExit) as exc_info:
      parser.parse_args(["--help"])
    assert exc_info.value.code == 0

  def test_ppo_help_renders(self):
    parser = argparse.ArgumentParser()
    PPOMethod.add_args(parser)
    with pytest.raises(SystemExit) as exc_info:
      parser.parse_args(["--help"])
    assert exc_info.value.code == 0

  def test_sac_help_renders(self):
    parser = argparse.ArgumentParser()
    SACMethod.add_args(parser)
    with pytest.raises(SystemExit) as exc_info:
      parser.parse_args(["--help"])
    assert exc_info.value.code == 0

  def test_pipeline_help_renders(self):
    parser = argparse.ArgumentParser()
    RLPipeline.add_args(parser)
    with pytest.raises(SystemExit) as exc_info:
      parser.parse_args(["--help"])
    assert exc_info.value.code == 0

  def test_no_option_collision(self):
    """All three methods + pipeline on one parser should work."""
    parser = argparse.ArgumentParser()
    DQNMethod.add_args(parser)
    PPOMethod.add_args(parser)
    SACMethod.add_args(parser)
    RLPipeline.add_args(parser)
    args = parser.parse_args([])
    # DQN defaults.
    assert args.gamma == 0.99
    assert args.ddqn is True
    # PPO defaults.
    assert args.ppo_gamma == 0.99
    assert args.ppo_clip_eps == 0.2
    # SAC defaults.
    assert args.sac_gamma == 0.99
    assert args.sac_tau == 0.005
    # Pipeline defaults.
    assert args.env_id == "CartPole-v1"
    assert args.warmup_steps == 1000

  def test_dqn_args_parse(self):
    parser = argparse.ArgumentParser()
    DQNMethod.add_args(parser)
    args = parser.parse_args([
        "--gamma", "0.95",
        "--epsilon_start", "0.8",
        "--epsilon_end", "0.05",
        "--epsilon_decay_steps", "20000",
        "--tau", "0.5",
        "--dueling",
    ])
    assert args.gamma == 0.95
    assert args.epsilon_start == 0.8
    assert args.epsilon_end == 0.05
    assert args.epsilon_decay_steps == 20000
    assert args.tau == 0.5
    assert args.dueling is True

  def test_ppo_args_parse(self):
    parser = argparse.ArgumentParser()
    PPOMethod.add_args(parser)
    args = parser.parse_args([
        "--ppo-clip-eps",
        "0.3",
        "--ppo-epochs",
        "8",
        "--ppo-lam",
        "0.9",
        "--ppo-rollout-len",
        "4096",
        "--ppo-entropy-coef",
        "0.05",
    ])
    assert args.ppo_clip_eps == 0.3
    assert args.ppo_epochs == 8
    assert args.ppo_lam == 0.9
    assert args.ppo_rollout_len == 4096
    assert args.ppo_entropy_coef == 0.05

  def test_sac_args_parse(self):
    parser = argparse.ArgumentParser()
    SACMethod.add_args(parser)
    args = parser.parse_args([
        "--sac-tau",
        "0.002",
        "--sac-alpha",
        "0.5",
        "--sac-no-auto-alpha",
    ])
    assert args.sac_tau == 0.002
    assert args.sac_alpha == 0.5
    assert args.sac_auto_alpha is False
