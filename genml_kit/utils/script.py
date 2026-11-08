"""Utilities for loading and executing user-supplied Python scripts."""

import atexit
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import os
import posixpath
import shutil
import sys
import tempfile
import urllib.request
from urllib.parse import urlparse

from genml_kit.utils.logging import fatal

# Per-process root for fetched (URL) scripts, created lazily by
# :func:`_temp_root`.  Kept as a single directory so cleanup is one ``rmtree``
# at interpreter exit rather than bookkeeping per fetched script.
_TEMP_ROOT = None


def is_script_spec(spec: str) -> bool:
  """Return True if *spec* looks like a script path or URL.

    A script spec is either:
    - A local path ending in ``.py``
    - An HTTP/HTTPS URL
    """
  return spec.endswith(".py") or spec.startswith(("http://", "https://"))


def _temp_root():
  """Lazily create (once per process) the root that holds fetched scripts.

    A fresh ``mkdtemp`` per process keeps concurrent instances from clobbering
    each other's in-flight modules, and the whole tree is removed in a single
    ``atexit`` pass rather than per fetched script.  Created lazily so purely
    local-path loads never touch the filesystem beyond the script itself.
  """
  global _TEMP_ROOT
  if _TEMP_ROOT is None:
    _TEMP_ROOT = tempfile.mkdtemp(prefix="genml_kit_extern_")
    atexit.register(shutil.rmtree, _TEMP_ROOT, ignore_errors=True)
  return _TEMP_ROOT


def _materialize(path_or_url):
  """Return a local filesystem path holding the script source.

    Local paths are returned unchanged.  URLs are fetched once and written into
    a dedicated per-URL subdirectory of :func:`_temp_root`; the subdir keeps
    scripts that share a basename (e.g. two ``model.py`` URLs) from colliding
    and gives the loaded module a real ``__file__`` on disk.
  """
  if not path_or_url.startswith(("http://", "https://")):
    return path_or_url

  with urllib.request.urlopen(path_or_url) as resp:
    code = resp.read().decode("utf-8")

  subdir = tempfile.mkdtemp(prefix="url_", dir=_temp_root())
  name = posixpath.basename(urlparse(path_or_url).path) or "script.py"
  if not name.endswith(".py"):
    name += ".py"
  path = os.path.join(subdir, name)
  with open(path, "w", encoding="utf-8") as f:
    f.write(code)
  return path


def _module_name(path):
  """Stable, collision-free module name derived from the source path.

    Keying on the absolute path (not the basename) means two scripts that share
    a filename register under different names, so neither shadows the other in
    ``sys.modules`` and both stay importable/picklable.
  """
  digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
  return f"genml_extern_{digest}"


class _ScriptSiblingFinder(importlib.abc.MetaPathFinder):
  """Resolve a script's *sibling* modules without mutating ``sys.path``.

    Every loaded script registers the directory that holds it.  When the import
    system asks for a top-level name that no earlier finder claimed, this finder
    resolves it against those directories.  Sitting *after* ``PathFinder`` in
    ``sys.meta_path``, it can never shadow a real installed module or stdlib
    name, and because the directories stay registered for the process lifetime it
    resolves imports performed *after* loading too -- e.g. a sibling imported
    lazily inside a function body to break an import cycle, which the previous
    ``sys.path`` insert/remove trick could not reach.
  """

  def __init__(self):
    self._dirs = []

  def add(self, directory):
    """Register *directory* to be searched for sibling modules (deduped)."""
    directory = os.path.abspath(directory)
    if directory not in self._dirs:
      self._dirs.append(directory)

  def find_spec(self, fullname, path=None, target=None):
    """Return a spec for a top-level *fullname* found in a script directory."""
    if "." in fullname:
      return None
    for directory in self._dirs:
      candidate = os.path.join(directory, fullname + ".py")
      if os.path.isfile(candidate):
        loader = importlib.machinery.SourceFileLoader(fullname, candidate)
        return importlib.util.spec_from_file_location(fullname,
                                                      candidate,
                                                      loader=loader)
    return None


# Process-wide finder, installed lazily on the first script load.  Appended at
# the end of ``sys.meta_path`` so the stdlib/site ``PathFinder`` always wins.
_SIBLING_FINDER = None


def _sibling_finder():
  """Return the shared sibling finder, installing it once on first use."""
  global _SIBLING_FINDER
  if _SIBLING_FINDER is None:
    _SIBLING_FINDER = _ScriptSiblingFinder()
    sys.meta_path.append(_SIBLING_FINDER)
  return _SIBLING_FINDER


def _load_script(path_or_url):
  """Load a Python script as a module and return its namespace.

    Handles both local files and HTTP/HTTPS URLs.  Unlike a bare ``exec``, the
    script is imported as a real module: it gets a proper ``__file__`` /
    ``__name__``, its directory is registered with :class:`_ScriptSiblingFinder`
    (so it can import siblings by plain name, eagerly or lazily), and classes it
    defines are picklable.  ``sys.path`` is never touched, and the module stays
    registered in ``sys.modules`` so pickled instances resolve later.
  """
  path = _materialize(path_or_url)
  name = _module_name(path)
  _sibling_finder().add(os.path.dirname(os.path.abspath(path)))
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    # Fall back to an explicit source loader for extensions importlib's
    # default hooks do not recognise.
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return vars(module)


def _extract_fn(namespace, path_or_url, fn_name):
  """Extract a callable *fn_name* from a script namespace."""
  fn = namespace.get(fn_name)
  if fn is None or not callable(fn):
    fatal(f"Script {path_or_url!r} does not define a callable '{fn_name}'.", ValueError)
  return fn


def load_extern(path_or_url, fn_name):
  """Load a Python script and return a callable extracted from it.

    The script is fetched from *path_or_url* (a local path or HTTP/HTTPS
    URL), compiled and executed, then the function *fn_name* is extracted
    from its namespace and returned.

    Args:
        path_or_url: Local file path or HTTP/HTTPS URL to the script.
        fn_name: Name of the callable to extract from the script.

    Returns:
        The extracted callable.

    Raises:
        FileNotFoundError: If a local path does not exist.
        ValueError: If the script does not define a callable *fn_name*.
  """
  namespace = _load_script(path_or_url)
  logging.info("Loading %s from %s", fn_name, path_or_url)
  return _extract_fn(namespace, path_or_url, fn_name)


def extern_call(path_or_url, fn_name, *args, **kwargs):
  """Load a Python script and call one of its top-level functions.

    The script is fetched from *path_or_url* (a local path or HTTP/HTTPS
    URL), compiled and executed, then the function *fn_name* is extracted
    from its namespace and called with the supplied positional and keyword
    arguments.

    Args:
        path_or_url: Local file path or HTTP/HTTPS URL to the script.
        fn_name: Name of the callable to extract from the script.
        *args: Positional arguments forwarded to the callable.
        **kwargs: Keyword arguments forwarded to the callable.

    Returns:
        Whatever *fn_name* returns.

    Raises:
        FileNotFoundError: If a local path does not exist.
        ValueError: If the script does not define a callable *fn_name*.
  """
  fn = load_extern(path_or_url, fn_name)
  return fn(*args, **kwargs)
