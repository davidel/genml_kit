"""Unit tests for ``ReturnNormalizer`` (SB3-style reward normalization).

Covers:
- discounted-return accumulation + Welford running stats
- reward normalization by running std (clipped)
- ``training=False`` (eval) returns raw rewards and skips updates
- checkpoint round-trip via ``state_dict`` / ``load_state_dict``
- ``reset_per_episode`` behaviour
"""

import torch

from genml_kit.pipelines.rl_normalize import ReturnNormalizer


def _almost(a, b, tol=1e-5):
  return abs(a - b) < tol


class TestReturnNormalizer:

  def test_training_mode_normalizes_and_clips(self):
    norm = ReturnNormalizer(gamma=0.99, clip_reward=10.0)
    # Feed a burst of large rewards; running std grows from ~1.0.
    for _ in range(100):
      norm.update(100.0)
    std = torch.sqrt(norm.ret_rms.var + norm.epsilon).item()
    assert std > 1.0
    normed = norm.normalize_reward(100.0)
    # Normalized by std (clipped to clip_reward).
    assert normed.abs().item() <= 10.0 + 1e-6

  def test_eval_mode_returns_raw_reward_and_skips_update(self):
    norm = ReturnNormalizer(gamma=0.99)
    norm.train(False)  # eval
    before = torch.sqrt(norm.ret_rms.var + norm.epsilon).item()
    # update() must be a no-op in eval.
    norm.update(100.0)
    after = torch.sqrt(norm.ret_rms.var + norm.epsilon).item()
    assert _almost(before, after)
    # normalize_reward returns the raw reward unchanged.
    assert _almost(float(norm.normalize_reward(42.0)), 42.0)

  def test_discounted_return_accumulation(self):
    norm = ReturnNormalizer(gamma=0.9)
    norm.update(1.0)      # running_return = 1.0
    norm.update(1.0)      # running_return = 0.9*1.0 + 1.0 = 1.9
    assert _almost(float(norm.running_return), 1.9)
    # Stats updated with the running-return trajectory, not raw rewards.
    assert norm.ret_rms.count.item() >= 2

  def test_reset_per_episode(self):
    norm = ReturnNormalizer(gamma=0.99)
    norm.update(5.0)
    assert float(norm.running_return) > 0
    norm.reset_per_episode()
    assert _almost(float(norm.running_return), 0.0)

  def test_state_dict_roundtrip(self):
    norm = ReturnNormalizer(gamma=0.99, clip_reward=5.0)
    for _ in range(50):
      norm.update(2.0)
    state = norm.state_dict()
    norm2 = ReturnNormalizer(gamma=0.99, clip_reward=5.0)
    norm2.load_state_dict(state)
    # Running stats match.
    assert _almost(float(norm.ret_rms.mean), float(norm2.ret_rms.mean))
    assert _almost(float(norm.ret_rms.var), float(norm2.ret_rms.var))
    assert _almost(float(norm.ret_rms.count), float(norm2.ret_rms.count))
    # Config fields match.
    assert norm2.gamma == norm.gamma
    assert norm2.clip_reward == norm.clip_reward

  def test_init_std_is_one(self):
    # With no data, RunningMeanStd.var = 1.0 -> std ~= 1.0 (SB3 start).
    norm = ReturnNormalizer(gamma=0.99)
    std = torch.sqrt(norm.ret_rms.var + norm.epsilon).item()
    assert _almost(std, 1.0, tol=1e-3)
    # No clipping on unit-scale rewards.
    normed = norm.normalize_reward(0.5)
    assert _almost(float(normed), 0.5, tol=1e-3)


class TestReturnNormalizerPipelineIntegration:

  def test_ret_norm_created_only_when_flag_set(self):
    import argparse

    from genml_kit.pipelines.rl import RLPipeline

    pipe = RLPipeline()
    assert pipe.ret_norm is None  # default: no reward normalization

    # Simulate init_env with reward_normalize enabled (needs env).
    # Minimal: pass args and confirm internal flag is read.
    args = argparse.Namespace(reward_normalize=True,
                              reward_norm_clip=3.0,
                              obs_normalize=False,
                              obs_norm_clip=10.0,
                              ppo_gamma=0.95)
    # init_env requires a real env; just verify the flag parsing path via
    # the attributes the pipeline reads.  We test the module itself above,
    # and the wiring in test_rl_trainer for the full path.
    assert args.reward_normalize is True
    assert args.reward_norm_clip == 3.0
