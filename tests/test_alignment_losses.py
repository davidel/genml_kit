"""Tests for the alignment losses (losses/alignment.py).

Pure-function checks: sequence log-probability aggregation against a
hand-computed reference, the DPO sigmoid objective, and the Bradley-Terry
ranking loss -- no model or tokenizer needed.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from genml_kit.losses import LOSSES, load_loss
from genml_kit.losses.alignment import (bradley_terry_loss, dpo_loss, sequence_logps)


class TestSequenceLogps:

  def test_uniform_logits_match_log_uniform(self):
    # Uniform logits over V vocab -> log-prob of any token is -log(V).
    vocab = 8
    logits = torch.zeros(1, 3, vocab)
    labels = torch.tensor([[1, 2, 3]])
    logps = sequence_logps(logits, labels)
    assert logps.item() == pytest.approx(-math.log(vocab), rel=1e-5)

  def test_ignored_positions_excluded(self):
    vocab = 4
    logits = torch.zeros(1, 3, vocab)
    # Only the last position counts -> same -log(V).
    labels = torch.tensor([[-100, -100, 0]])
    logps = sequence_logps(logits, labels)
    assert logps.item() == pytest.approx(-math.log(vocab), rel=1e-5)

  def test_all_ignored_is_zero(self):
    logits = torch.zeros(1, 3, 4)
    labels = torch.full((1, 3), -100)
    assert sequence_logps(logits, labels).item() == 0.0

  def test_peaked_logits_high_logprob(self):
    # A confident (peaked) distribution yields a log-prob near 0.
    logits = torch.tensor([[[0.0, 50.0]]])
    labels = torch.tensor([[1]])
    assert sequence_logps(logits, labels).item() > -1e-3

  def test_matches_manual_mean(self):
    logits = torch.randn(2, 4, 6)
    labels = torch.tensor([[1, 2, 3, 4], [-100, 2, -100, 5]])
    logps = sequence_logps(logits, labels)
    log_probs = F.log_softmax(logits, dim=-1)
    for b, row in enumerate(labels.tolist()):
      vals = [log_probs[b, t, tok] for t, tok in enumerate(row) if tok != -100]
      assert logps[b].item() == pytest.approx(float(sum(vals) / len(vals)), rel=1e-5)


class TestDpoLoss:

  def test_zero_margin_is_log2(self):
    # Equal policy/reference log-ratios -> logit 0 -> -log sigmoid(0) = log 2.
    z = torch.zeros(3)
    loss, chosen, rejected = dpo_loss(z, z, z, z, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), rel=1e-5)
    assert torch.allclose(chosen, z) and torch.allclose(rejected, z)

  def test_favouring_chosen_lowers_loss(self):
    # Policy lifts chosen over reference -> positive logit -> smaller loss.
    zero = torch.zeros(1)
    good = torch.tensor([1.0])
    loss, _, _ = dpo_loss(good, zero, zero, zero, beta=1.0)
    assert loss.item() < math.log(2)

  def test_beta_scales_margin(self):
    one = torch.ones(1)
    zero = torch.zeros(1)
    _, chosen_lo, _ = dpo_loss(one, zero, zero, zero, beta=0.1)
    _, chosen_hi, _ = dpo_loss(one, zero, zero, zero, beta=1.0)
    assert chosen_hi.item() == pytest.approx(10 * chosen_lo.item())

  def test_rewards_are_detached(self):
    z = torch.zeros(2, requires_grad=True)
    _, chosen, rejected = dpo_loss(z, z, z, z, beta=1.0)
    assert not chosen.requires_grad and not rejected.requires_grad


class TestBradleyTerry:

  def test_equal_scores_is_log2(self):
    scores = torch.zeros(3)
    assert bradley_terry_loss(scores, scores).item() == pytest.approx(math.log(2))

  def test_chosen_above_rejected_lowers_loss(self):
    chosen = torch.tensor([1.0])
    rejected = torch.tensor([0.0])
    assert bradley_terry_loss(chosen, rejected).item() < math.log(2)

  def test_registered_in_losses(self):
    assert "dpo" in LOSSES.list_names()
    assert "bradley_terry" in LOSSES.list_names()
    assert load_loss("dpo") is dpo_loss
    assert load_loss("bradley_terry") is bradley_terry_loss
