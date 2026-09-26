"""RL model selector tests: --model / --model_arg reach the constructors.

Covers the core configurability contract from plans/RL_CUSTOM_MODELS.md:
- ``--model`` selects any registered custom RL factory,
- ``--model_arg`` forwards constructor kwargs,
- unknown names fail fast via the registry / HF fallback,
- the SAC factory-return contract (Module or dict),
- ``_apply_model_extras`` runs for RL methods.
"""

import argparse
from unittest.mock import patch

import pytest
import torch

from genml_kit.methods import METHODS
from genml_kit.models.registry import is_custom_model
from genml_kit.models.rl.actor_critic import ActorCritic
from genml_kit.models.rl.qnetwork import DuelingQHead, QNetwork
from genml_kit.models.rl.sac_model import SACModel
from genml_kit.models.rl.sac_critic import SACCritic
from genml_kit.models.rl.spaces import space_spec
from genml_kit.pipelines.rl import RLPipeline, _ScriptedEnv


def _make_args(**overrides):
  """Minimal args for an RL method with model/model_arg passthrough."""
  defaults = dict(
      model=None,
      model_arg={},
      dqn_gamma=0.99,
      dqn_ddqn=True,
      dqn_dueling=False,
      dqn_epsilon_start=1.0,
      dqn_epsilon_end=0.02,
      dqn_epsilon_decay_steps=50_000,
      dqn_tau=1.0,
      dqn_target_update_freq=0,
      dqn_n_step=1,
      ppo_discrete=True,
      ppo_gamma=0.99,
      ppo_lam=0.95,
      ppo_clip_eps=0.2,
      ppo_epochs=2,
      ppo_mini_batch_size=32,
      ppo_entropy_coef=0.01,
      ppo_value_coef=0.5,
      ppo_vf_clip_eps=None,
      ppo_target_kl=None,
      ppo_log_std_init=0.0,
      ppo_log_std_min=-5.0,
      ppo_log_std_max=1.0,
      ppo_rollout_len=32,
      sac_gamma=0.99,
      sac_tau=0.005,
      sac_alpha=0.2,
      sac_auto_alpha=True,
      # _apply_model_extras fields (defaults to no-op).
      grad_checkpoint=False,
      lora=False,
      lora_r=8,
      lora_alpha=16,
      lora_dropout=0.0,
      source_checkpoint=None,
      param_init=None,
      freeze_patterns=None,
      param_rename=None,
      freeze=False,
  )
  defaults.update(overrides)
  return argparse.Namespace(**defaults)


def _make_pipeline(obs_dim=4, continuous=False, action_dim=2):
  """Return an RLPipeline with a scripted env and a populated space."""
  pipeline = RLPipeline()
  env = _ScriptedEnv(obs_dim=obs_dim,
                     max_episode_length=6,
                     continuous=continuous,
                     action_dim=action_dim)
  pipeline.env = env
  pipeline.space = space_spec(env.observation_space, env.action_space)
  return pipeline


class TestModelSelection:

  def test_dqn_default_model_still_works(self):
    """With no --model, DQN falls back to rl/qnet."""
    pipeline = _make_pipeline()
    method = METHODS.get("dqn")()
    method.wire_data(_make_args(), pipeline)
    model = method.build_model(_make_args(), torch.device("cpu"))
    assert isinstance(model, QNetwork)

  def test_dqn_dueling_shortcut_preserved(self):
    """--dqn_dueling still selects rl/qnet_dueling when --model is absent."""
    pipeline = _make_pipeline()
    method = METHODS.get("dqn")()
    method.wire_data(_make_args(), pipeline)
    model = method.build_model(_make_args(dqn_dueling=True), torch.device("cpu"))
    assert isinstance(model.online[1], DuelingQHead)

  def test_dqn_custom_model_selected(self):
    """--model rl/qnet_dueling overrides the default."""
    pipeline = _make_pipeline()
    method = METHODS.get("dqn")()
    method.wire_data(_make_args(), pipeline)
    args = _make_args(model="rl/qnet_dueling")
    model = method.build_model(args, torch.device("cpu"))
    assert isinstance(model.online[1], DuelingQHead)

  def test_ppo_custom_model_selected(self):
    """PPO --model selects the ActorCritic and drives sizes from space."""
    pipeline = _make_pipeline()
    method = METHODS.get("ppo")()
    method.wire_data(_make_args(), pipeline)
    args = _make_args(model="rl/actor_critic")
    model = method.build_model(args, torch.device("cpu"))
    # Discrete space -> Categorical actor with n_actions logits.
    assert model.backbone.net[-2].out_features == 256
    assert model.actor.logits.out_features == 2
    assert model.critic.fc.in_features == 256

  def test_unknown_model_fails_fast(self):
    """An unregistered --model name is not in the registry.

    ``load_model`` falls through to the HuggingFace backend for
    unregistered names, which fails loudly at load time with a 404
    ``OSError`` (the plan's ``no strict loading gate`` design).  We do
    not perform that network round-trip here; the deterministic contract
    is that the synthetic name is absent from the registry.
    """
    assert is_custom_model("rl/qnet") is True
    assert is_custom_model("rl/no_such_model") is False


class TestModelArgForwarding:

  def test_dqn_model_arg_reaches_constructor(self):
    """--model_arg hidden_dims=[512,256] reaches the QNetwork backbone."""
    pipeline = _make_pipeline()
    method = METHODS.get("dqn")()
    method.wire_data(_make_args(), pipeline)
    args = _make_args(model_arg={"hidden_dims": [512, 256]})
    model = method.build_model(args, torch.device("cpu"))
    backbone = model.online[0]
    assert backbone.net[0].out_features == 512
    assert backbone.net[0].in_features == 4

  def test_ppo_model_arg_reaches_constructor(self):
    pipeline = _make_pipeline()
    method = METHODS.get("ppo")()
    method.wire_data(_make_args(), pipeline)
    args = _make_args(model_arg={"hidden_dims": [64, 16]})
    model = method.build_model(args, torch.device("cpu"))
    assert model.backbone.net[0].out_features == 64


class TestSACFactoryContract:

  def test_single_module_uses_default_actor(self):
    """A single SACCritic serves as the critic; actor is rl/actor_critic."""
    pipeline = _make_pipeline(continuous=True)
    method = METHODS.get("sac")()
    method.wire_data(_make_args(), pipeline)
    model = method.build_model(_make_args(), torch.device("cpu"))
    assert isinstance(model, SACModel)
    assert isinstance(model.q1, SACCritic)
    assert isinstance(model.q2, SACCritic)
    assert model.q1 is not model.q2  # twins are distinct objects

  def test_dict_factory_supplies_both(self):
    """A dict {'actor':..., 'critic':...} is consumed by the method."""
    critic = SACCritic(obs_dim=4, action_dim=2)
    actor = ActorCritic(obs_dim=4, action_dim=2, discrete=False)
    pipeline = _make_pipeline(continuous=True)
    method = METHODS.get("sac")()
    method.wire_data(_make_args(), pipeline)
    with patch("genml_kit.methods.rl_sac.load_model",
               return_value={
                   "actor": actor,
                   "critic": critic
               }):
      model = method.build_model(_make_args(), torch.device("cpu"))
    assert isinstance(model, SACModel)
    assert model.actor is actor
    assert model.q1 is critic
    assert model.q2 is not critic

  def test_missing_dict_key_fails(self):
    """A dict missing 'actor'/'critic' raises a clear ValueError."""
    pipeline = _make_pipeline(continuous=True)
    method = METHODS.get("sac")()
    method.wire_data(_make_args(), pipeline)
    with patch("genml_kit.methods.rl_sac.load_model",
               return_value={"actor": None, "critic": None}), \
        pytest.raises(ValueError, match="missing"):
      method.build_model(_make_args(), torch.device("cpu"))

  def test_extra_dict_keys_ignored(self):
    """Extra keys beyond actor/critic are tolerated."""
    critic = SACCritic(obs_dim=4, action_dim=2)
    actor = ActorCritic(obs_dim=4, action_dim=2, discrete=False)
    pipeline = _make_pipeline(continuous=True)
    method = METHODS.get("sac")()
    method.wire_data(_make_args(), pipeline)
    with patch("genml_kit.methods.rl_sac.load_model",
               return_value={
                   "actor": actor,
                   "critic": critic,
                   "bonus": object()
               }):
      model = method.build_model(_make_args(), torch.device("cpu"))
    assert isinstance(model, SACModel)


class TestExtrasInvoked:

  def test_ortho_init_changes_weight_norm(self):
    """--param_init ortho runs through _apply_model_extras for DQN."""
    pipeline = _make_pipeline()
    method = METHODS.get("dqn")()
    method.wire_data(_make_args(), pipeline)
    args = _make_args(param_init="ortho")
    model = method.build_model(args, torch.device("cpu"))
    # SB3 init_orthogonal zeroes biases (a PyTorch-default-init weight
    # would have non-zero bias with probability 1).
    assert model.online[0].net[0].bias.detach().abs().sum().item() == 0.0
    assert model.online[0].net[0].bias.detach().abs().max().item() == 0.0

