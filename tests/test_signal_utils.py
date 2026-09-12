"""Tests for genml_kit.signal_utils (sigexcept context manager)."""

import logging
import os
import signal
import threading
import time

import pytest

from genml_kit.training.trainer import TrainingResult
from genml_kit.utils.signal import InterruptedException, sigexcept


class TestSigexcept:

  def test_signal_raises_and_records(self):
    with sigexcept("SIGUSR1") as interrupts:
      try:
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.2)  # let the handler raise at a bytecode boundary
        pytest.fail("expected InterruptedException")
      except InterruptedException:
        pass
    assert interrupts.received == ["SIGUSR1"]

  def test_handlers_restored_custom(self):
    flag = []

    def custom(signum, frame):
      flag.append(signum)

    original = signal.signal(signal.SIGUSR1, custom)
    try:
      with sigexcept("SIGUSR1"):
        pass
      assert signal.getsignal(signal.SIGUSR1) is custom
      os.kill(os.getpid(), signal.SIGUSR1)
      time.sleep(0.2)
      assert flag
    finally:
      signal.signal(signal.SIGUSR1, original)

  def test_handlers_restored_sig_ign(self):
    original = signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
      with sigexcept("SIGUSR1"):
        pass
      assert signal.getsignal(signal.SIGUSR1) is signal.SIG_IGN
    finally:
      signal.signal(signal.SIGUSR1, original)

  def test_unknown_name_logged_and_skipped(self, caplog):
    with (
        caplog.at_level(logging.WARNING),
        sigexcept("SIGUSR1,SIGDOESNOTEXIST") as interrupts,
    ):
      try:
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.2)
        pytest.fail("expected InterruptedException")
      except InterruptedException:
        pass
    assert any("SIGDOESNOTEXIST" in rec.getMessage() for rec in caplog.records)
    assert interrupts.received == ["SIGUSR1"]

  def test_unhandleable_rejected(self):
    with pytest.raises(ValueError, match="SIGKILL"), sigexcept("SIGKILL"):
      pass

  def test_latch_on_duplicates(self):
    # The second delivery happens inside the except block, i.e. while the
    # first exception is unwinding: it must be latched, not raised again.
    with sigexcept("SIGUSR1") as interrupts:
      try:
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.2)
        pytest.fail("expected InterruptedException")
      except InterruptedException:
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.2)
    assert interrupts.received == ["SIGUSR1", "SIGUSR1"]

  def test_off_main_thread_enter_rejected(self):
    results = []

    def worker():
      try:
        with sigexcept("SIGUSR1"):
          pass
      except ValueError as e:
        results.append(str(e))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert results
    assert "main thread" in results[0]

  def test_iterable_spec(self):
    # The iterable form must behave exactly like the string form.
    with sigexcept(["SIGUSR1"]) as interrupts:
      try:
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.2)
        pytest.fail("expected InterruptedException")
      except InterruptedException:
        pass
    assert interrupts.received == ["SIGUSR1"]

  def test_train_import_wiring(self):
    # Integration smoke: the training entries must expose the shared
    # loop result type (wiring reviewed via the train/pretrain patches).
    import genml_kit.pretrain.cli
    import genml_kit.training.train

    assert genml_kit.training.train.TrainingResult is TrainingResult
    assert genml_kit.pretrain.cli.TrainingResult is TrainingResult
