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
    # Exercise the reporter's own renderer (shared by the base and image
    # reporters), not a copy of its branches.
    assert self._report(total_batches=10)._progress(0) == "[1/10]"
    assert self._report(total_batches=10,
                        total_batches_estimated=True)._progress(0) == "[1/~10]"
    assert self._report(total_batches=None)._progress(0) == "[1/?]"

  def test_image_reporter_degrades_like_base(self):
    # The image reporter used to hardcode ``{total_batches}`` and print
    # ``[i/None]`` when the length was unknown; it now shares ``_progress``.
    assert self._report(total_batches=None)._progress(1249) == "[1250/?]"
    # In-range index keeps the '~' estimate form (overflow is covered below).
    assert self._report(total_batches=10,
                        total_batches_estimated=True)._progress(4) == "[5/~10]"

  def test_progress_at_denominator_is_not_exceeded(self):
    # The boundary itself must stay plain: overflow starts only past N.
    assert self._report(total_batches=10)._progress(9) == "[10/10]"
    assert self._report(total_batches=10,
                        total_batches_estimated=True)._progress(9) == "[10/~10]"

  def test_progress_tracks_numerator_when_estimate_exceeded(self):
    # A stream that yields more batches than estimated must not print a
    # numerator larger than its denominator; the denominator follows the
    # numerator and gains a '+' (the '~' still marks the estimate).
    assert self._report(total_batches=10,
                        total_batches_estimated=True)._progress(10) == "[11/~11+]"
    assert self._report(
        total_batches=10,
        total_batches_estimated=True)._progress(43049) == "[43050/~43050+]"

  def test_progress_tracks_numerator_when_exact_count_exceeded(self):
    # An exact count that is somehow exceeded overflows the same way, minus
    # the '~' estimate marker.
    assert self._report(total_batches=10)._progress(10) == "[11/11+]"


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
