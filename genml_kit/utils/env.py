"""Environment-variable configuration helpers.

A single place to read typed configuration from the environment, so
deployments can pin or vary defaults without code changes (mirrors the
intent of ``genml_kit.utils.seed.resolve_seed``).
"""

import os

from genml_kit.utils.logging import fatal

_TRUE_TOKENS = frozenset(("1", "true", "yes", "on"))
_FALSE_TOKENS = frozenset(("0", "false", "no", "off"))


def getenv(name, vtype=int, defval=None):
  """Fetch environment variable *name* and convert it to *vtype*.

  Blank or unset variables yield *defval*.  Booleans accept the usual
  spellings (``1/true/yes/on`` and ``0/false/no/off``, case-insensitive);
  any other *vtype* is applied directly.  A value that cannot be converted
  is fatal, so misconfiguration fails fast instead of mid-run.

  Args:
      name: Environment variable name.
      vtype: Callable applied to the raw string (e.g. ``int``, ``bool``,
          ``str``).
      defval: Value returned when the variable is unset or blank.

  Returns:
      The converted value, or *defval*.
  """
  raw = (os.getenv(name) or "").strip()
  if not raw:
    return defval
  if vtype is bool:
    low = raw.lower()
    if low in _TRUE_TOKENS:
      return True
    if low in _FALSE_TOKENS:
      return False
    fatal(f"Environment variable {name} must be a boolean, got: {raw!r}")
  try:
    return vtype(raw)
  except ValueError:
    fatal(f"Environment variable {name} must be of type "
          f"{vtype.__name__}, got: {raw!r}")
