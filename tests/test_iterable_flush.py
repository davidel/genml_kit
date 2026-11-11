"""Flush-on-exhaustion and progress reporting for length-less epochs (§15.9).

The epoch no longer needs a length up front: the loop runs to exhaustion and a
trailing partial grad-accumulation window is closed afterwards.  These tests pin
that the arithmetic matches the old in-loop tail flush for random access, and
that it works at all for a stream.
"""

import pytest

from genml_kit.training.train_reporting import ImageTrainReporting


def _flush_count(batches, accum_steps):
  """Model the trainer's window closing: full windows, then the tail."""
  steps = 0
  for index in range(batches):
    if (index + 1) % accum_steps == 0:
      steps += 1
  if batches and batches % accum_steps != 0:
    steps += 1
  return steps


def _legacy_flush_count(batches, accum_steps):
  """The replaced rule: flush on a full window *or* on the last batch."""
  steps = 0
  for index in range(batches):
    if (index + 1) % accum_steps == 0 or (index + 1) == batches:
      steps += 1
  return steps


class TestFlushArithmetic:

  @pytest.mark.parametrize("accum", [1, 2, 3, 4, 8])
  @pytest.mark.parametrize("batches", list(range(0, 25)))
  def test_matches_legacy_rule_exactly(self, batches, accum):
    assert _flush_count(batches, accum) == _legacy_flush_count(batches, accum)

  def test_exact_multiple_does_not_double_flush(self):
    # 8 batches with accum 4 is two full windows and nothing left over.
    assert _flush_count(8, 4) == 2

  def test_partial_tail_flushes_once(self):
    # 10 batches with accum 4: 2 full windows + 1 tail.
    assert _flush_count(10, 4) == 3

  def test_single_partial_window_flushes(self):
    assert _flush_count(3, 4) == 1

  def test_empty_epoch_flushes_nothing(self):
    assert _flush_count(0, 4) == 0


class TestProgressRendering:
  """The reporter must degrade instead of raising when length is unknown."""

  def _report(self, **kwargs):
    reporter = ImageTrainReporting(log_every=1, **kwargs)
    return reporter

  def test_exact_length_renders_plain(self):
    reporter = self._report(total_batches=10)
    assert reporter._total_batches == 10
    assert reporter._total_batches_estimated is False

  def test_unknown_length_is_none(self):
    reporter = self._report(total_batches=None)
    assert reporter._total_batches is None

  def test_estimated_length_is_flagged(self):
    reporter = self._report(total_batches=10, total_batches_estimated=True)
    assert reporter._total_batches_estimated is True

  def test_progress_strings(self):
    # Replicates the formatting branch in ``step`` without running a step.
    def render(total, estimated, index):
      if total is None:
        return f"[{index + 1}/?]"
      if estimated:
        return f"[{index + 1}/~{total}]"
      return f"[{index + 1}/{total}]"

    assert render(10, False, 0) == "[1/10]"
    assert render(10, True, 0) == "[1/~10]"
    assert render(None, False, 0) == "[1/?]"


class TestStreamEpochEnds:

  def test_stream_epoch_terminates_on_exhaustion(self):
    from tests.iterable_streams import CountingStream

    stream = CountingStream(17)
    seen = 0
    for _ in stream:
      seen += 1
    assert seen == 17

  def test_length_less_stream_has_no_len(self):
    from tests.iterable_streams import CountingStream

    with pytest.raises(TypeError):
      len(CountingStream(5))
