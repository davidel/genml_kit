"""Tests for RL CLI wiring (no option collisions, help renders)."""

import argparse

import pytest

from genml_kit.methods import get_method, list_methods
from genml_kit.pipelines import list_pipelines
from genml_kit.methods.rl_dqn import DQNMethod
from genml_kit.pipelines.rl import RLPipeline


class TestRLRegistration:
  """Verify RL components are registered and don't collide."""

  def test_dqn_registered(self):
    assert "dqn" in list_methods()
    assert get_method("dqn") is DQNMethod

  def test_rl_pipeline_registered(self):
    assert "rl" in list_pipelines()

  def test_no_name_collision(self):
    """RL names use prefix patterns and don't clash with existing names."""
    methods = list_methods()
    pipelines = list_pipelines()
    # RL method name unique.
    assert methods.count("dqn") == 1
    # RL pipeline name unique.
    assert pipelines.count("rl") == 1


class TestRLCLIHelp:
  """Verify --help renders without error for RL config."""

  def test_method_help_renders(self):
    parser = argparse.ArgumentParser()
    DQNMethod.add_args(parser)
    # --help raises SystemExit(0); capture it.
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
    """RL pipeline args don't shadow existing pipeline args."""
    parser = argparse.ArgumentParser()
    RLPipeline.add_args(parser)
    # Should parse without error.
    args = parser.parse_args([])
    assert args.env_id == "CartPole-v1"
    assert args.warmup_steps == 1000
    assert args.replay_capacity == 100_000
    assert args.steps_per_epoch == 1000

  def test_dqn_args_parse(self):
    parser = argparse.ArgumentParser()
    DQNMethod.add_args(parser)
    args = parser.parse_args([
        "--gamma",
        "0.95",
        "--epsilon_start",
        "0.8",
        "--epsilon_end",
        "0.05",
        "--epsilon_decay_steps",
        "20000",
        "--tau",
        "0.5",
        "--dueling",
    ])
    assert args.gamma == 0.95
    assert args.epsilon_start == 0.8
    assert args.epsilon_end == 0.05
    assert args.epsilon_decay_steps == 20000
    assert args.tau == 0.5
    assert args.dueling is True
