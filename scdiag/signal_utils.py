"""Convert handled signals into exceptions for graceful shutdown.

The default OS disposition of ``SIGTERM`` / ``SIGHUP`` terminates the
process immediately: no unwinding, no ``finally``, no checkpoint.  Python
only special-cases ``SIGINT``.  :class:`sigexcept` installs real handlers
for a user-selected signal set and turns their arrival into
:class:`InterruptedException` raised at the next bytecode boundary, so a
training loop's ``finally`` block runs and a consistent checkpoint lands
on disk before a clean exit (code 0).
"""

import logging
import signal
import threading

_UNHANDLEABLE = frozenset({"SIGKILL", "SIGSTOP"})


class InterruptedException(Exception):
  """Raised inside a ``sigexcept`` block when a handled signal arrives."""


class sigexcept:
  """Context manager converting selected signals into exceptions.

  Example::

      with sigexcept() as interrupts:      # default: SIGINT,SIGTERM,SIGHUP
        try:
          train_all_the_things()
        except InterruptedException:
          logging.warning("Interrupted by %s; saving checkpoint.",
                          interrupts.received)

  Attributes:
      received: Ordered names of the signals that fired, e.g.
          ``["SIGTERM"]`` or ``["SIGTERM", "SIGINT"]``.  Duplicates are
          kept, so a latched re-delivery appears twice.
  """

  def __init__(self, spec="SIGINT,SIGTERM,SIGHUP"):
    """Install handlers for the signals named in *spec*.

    Args:
        spec: Either a comma-separated string of signal names
            (``"SIGINT,SIGTERM,SIGHUP"``) or an iterable of names.
    """
    self._spec = spec
    self._resolved = []
    self._originals = []
    self._unwinding = False
    self.received = []

  def _parse_spec(self):
    """Resolve the spec to ``(name, signum)`` pairs.

    Names not defined on the current platform are logged and skipped;
    unhandleable signals (``SIGKILL``, ``SIGSTOP``) are rejected.

    Returns:
        List of ``(name, signum)`` tuples.

    Raises:
        ValueError: If a signal cannot be handled at all, or if no
            handleable signal remains after parsing.
    """
    tokens = ([t.strip() for t in self._spec.split(",")]
              if isinstance(self._spec, str) else list(self._spec))
    resolved = []
    for token in tokens:
      if not token:
        continue
      if token in _UNHANDLEABLE:
        raise ValueError(f"{token} cannot be handled; refusing no-op setup")
      sig = getattr(signal, token, None)
      if sig is None or not isinstance(sig, signal.Signals):
        logging.warning("sigexcept: signal %r not available on this platform; ignoring",
                        token)
        continue
      resolved.append((token, int(sig)))
    if not resolved:
      raise ValueError("sigexcept: no handleable signals in spec")
    return resolved

  def __enter__(self):
    if threading.current_thread() is not threading.main_thread():
      raise ValueError("sigexcept: signal handlers can only be installed from the "
                       "main thread")
    self._resolved = self._parse_spec()
    self.received = []
    self._unwinding = False
    self._originals = []
    for _name, sig in self._resolved:
      self._originals.append((sig, signal.getsignal(sig)))
      signal.signal(sig, self._handler)
    return self

  def _handler(self, signum, frame):
    name = signal.Signals(signum).name
    self.received.append(name)
    if self._unwinding:
      # A duplicate signal arrived while the first exception is already
      # unwinding (e.g. during the checkpoint save).  Latch: log and
      # return so the in-progress save completes.
      logging.warning("sigexcept: %s received while unwinding; latched", name)
      return
    self._unwinding = True
    raise InterruptedException(f"Signal {name} received")

  def __exit__(self, exc_type, exc_value, traceback):
    for sig, original in self._originals:
      signal.signal(sig, original)
    self._originals = []
    return False
