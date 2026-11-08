"""Tests for script_utils (local + URL script loading)."""

import importlib.util
import os
import pickle
import sys

import pytest

import genml_kit.utils.script as script_utils
from genml_kit.utils.script import extern_call, load_extern


class TestLoadExternLocal:

  def test_loads_local_function(self, tmp_path):
    script = tmp_path / "ext.py"
    script.write_text("def make(x):\n  return x * 2\n")
    fn = load_extern(str(script), "make")
    assert fn(21) == 42

  def test_extern_call_forwards_arguments(self, tmp_path):
    script = tmp_path / "ext.py"
    script.write_text("def add(a, b=0):\n  return a + b\n")
    assert extern_call(str(script), "add", 1, b=2) == 3

  def test_missing_fn_raises_value_error(self, tmp_path):
    script = tmp_path / "ext.py"
    script.write_text("x = 1\n")
    with pytest.raises(ValueError, match="does not define a callable"):
      load_extern(str(script), "nope")

  def test_missing_local_path_raises(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      load_extern(str(tmp_path / "does_not_exist.py"), "fn")


class TestScriptModuleSemantics:
  """The script is imported as a real module, not ``exec``'d into ``{}``."""

  def test_sibling_import_without_syspath_hack(self, tmp_path):
    (tmp_path / "siblib.py").write_text("def greet():\n  return 'sibling'\n")
    script = tmp_path / "ext.py"
    # No ``sys.path`` fiddling of any kind: the loader must make the sibling
    # importable on its own.
    script.write_text("import siblib\n"
                      "def make():\n"
                      "  return siblib.greet()\n")
    assert extern_call(str(script), "make") == "sibling"

  def test_deferred_sibling_import_resolves_after_load(self, tmp_path):
    # A sibling imported *inside a function body* (not at module top level)
    # must resolve too: the directory has to stay usable after loading, not
    # just while the script body executes.
    (tmp_path / "lazysib.py").write_text("def hint():\n  return 'lazy'\n")
    script = tmp_path / "ext.py"
    script.write_text("def make():\n"
                      "  import lazysib\n"
                      "  return lazysib.hint()\n")
    assert extern_call(str(script), "make") == "lazy"

  def test_deferred_sibling_import_breaks_a_cycle(self, tmp_path):
    # Mirrors the config <-> parallel_groups cycle: two siblings import each
    # other, one via a lazy import inside a function body.
    (tmp_path / "cyc_leaf.py").write_text("def core():\n"
                                          "  from cyc_mid import helper\n"
                                          "  return helper()\n")
    (tmp_path / "cyc_mid.py").write_text("from cyc_leaf import core\n"
                                         "def helper():\n"
                                         "  return 'cycle-ok'\n")
    script = tmp_path / "ext.py"
    script.write_text("from cyc_mid import helper\n"
                      "def build():\n"
                      "  from cyc_leaf import core\n"
                      "  return core()\n")
    assert extern_call(str(script), "build") == "cycle-ok"

  def test_sibling_does_not_shadow_stdlib(self, tmp_path):
    # A sibling whose name collides with a stdlib module must not win: the
    # finder sits after PathFinder, so the real module is always preferred.
    # ``find_spec`` consults ``sys.meta_path`` in order, so stdlib ``colorsys``
    # must resolve to the stdlib file, never the sibling of the same name.
    (tmp_path / "colorsys.py").write_text("marker = 'sibling'\n")
    script = tmp_path / "ext.py"
    script.write_text("def make():\n  return 1\n")
    load_extern(str(script), "make")  # registers tmp_path with the finder
    spec = importlib.util.find_spec("colorsys")
    assert spec is not None
    assert os.path.realpath(spec.origin) != os.path.realpath(
        str(tmp_path / "colorsys.py"))

  def test_sibling_finder_is_installed_once(self, tmp_path):
    script = tmp_path / "ext.py"
    script.write_text("def make():\n  return 1\n")
    load_extern(str(script), "make")
    load_extern(str(script), "make")
    finder = script_utils._sibling_finder()
    assert sys.meta_path.count(finder) == 1

  def test_dunder_file_points_at_the_script(self, tmp_path):
    script = tmp_path / "ext.py"
    script.write_text("def where():\n  return __file__\n")
    assert os.path.realpath(extern_call(str(script),
                                        "where")) == os.path.realpath(str(script))

  def test_classes_defined_in_script_are_picklable(self, tmp_path):
    script = tmp_path / "ext.py"
    script.write_text("class Tiny:\n"
                      "  def __init__(self):\n"
                      "    self.value = 7\n"
                      "def make():\n"
                      "  return Tiny()\n")
    obj = extern_call(str(script), "make")
    clone = pickle.loads(pickle.dumps(obj))
    assert type(clone).__name__ == "Tiny"
    assert clone.value == 7

  def test_sys_path_is_restored_after_load(self, tmp_path):
    script = tmp_path / "ext.py"
    script.write_text("def make():\n  return 1\n")
    before = list(sys.path)
    load_extern(str(script), "make")
    assert sys.path == before

  def test_script_dir_is_not_left_on_sys_path(self, tmp_path):
    script = tmp_path / "ext.py"
    script.write_text("def make():\n  return 1\n")
    load_extern(str(script), "make")
    assert os.path.abspath(
        str(tmp_path)) not in [os.path.abspath(p or ".") for p in sys.path]

  def test_same_basename_scripts_do_not_shadow_each_other(self, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "model.py").write_text("def name():\n  return 'a'\n")
    (dir_b / "model.py").write_text("def name():\n  return 'b'\n")
    assert extern_call(str(dir_a / "model.py"), "name") == "a"
    assert extern_call(str(dir_b / "model.py"), "name") == "b"
    # Both must remain registered, keyed on their distinct absolute paths.
    assert extern_call(str(dir_a / "model.py"), "name") == "a"


class TestLoadScriptFromUrl:
  """URL branch: fetches via urllib, compiles with the URL as filename."""

  class _FakeResponse:

    def __init__(self, payload):
      self._payload = payload

    def read(self):
      return self._payload.encode("utf-8")

    def __enter__(self):
      return self

    def __exit__(self, *exc_info):
      return False

  def test_https_url_executes_fetched_code(self, monkeypatch):
    payload = "def make():\n  return 'from-url'\n"
    seen = {}

    def fake_urlopen(url):
      seen["url"] = url
      return TestLoadScriptFromUrl._FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert extern_call("https://example.com/ext.py", "make") == "from-url"
    assert seen["url"] == "https://example.com/ext.py"

  def test_http_url_also_executes_fetched_code(self, monkeypatch):
    payload = "def make():\n  return 'from-http'\n"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url: TestLoadScriptFromUrl._FakeResponse(payload),
    )
    assert extern_call("http://example.com/ext.py", "make") == "from-http"

  def test_url_script_gets_real_file_under_temp_root(self, monkeypatch):
    payload = "def where():\n  return __file__\n"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url: TestLoadScriptFromUrl._FakeResponse(payload),
    )
    path = extern_call("https://example.com/ext.py", "where")
    assert path is not None
    assert os.path.isfile(path)
    root = os.path.abspath(script_utils._temp_root())
    assert os.path.abspath(path).startswith(root + os.sep)

  def test_same_basename_urls_use_distinct_subdirs(self, monkeypatch):
    payload = "def where():\n  return __file__\n"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url: TestLoadScriptFromUrl._FakeResponse(payload),
    )
    first = extern_call("https://example.com/a/model.py", "where")
    second = extern_call("https://other.example/b/model.py", "where")
    assert os.path.dirname(first) != os.path.dirname(second)


class TestTempRoot:
  """The fetched-script root is per-process and removed once at exit."""

  def test_root_is_created_lazily_and_is_stable(self):
    # Reset the module-global so this test owns the lifecycle.
    original = script_utils._TEMP_ROOT
    script_utils._TEMP_ROOT = None
    try:
      assert script_utils._TEMP_ROOT is None
      root = script_utils._temp_root()
      assert os.path.isdir(root)
      assert script_utils._temp_root() == root  # stable across calls
    finally:
      script_utils._TEMP_ROOT = original

  def test_local_load_does_not_create_temp_root(self, tmp_path):
    original = script_utils._TEMP_ROOT
    script_utils._TEMP_ROOT = None
    try:
      script = tmp_path / "ext.py"
      script.write_text("def make():\n  return 1\n")
      load_extern(str(script), "make")
      assert script_utils._TEMP_ROOT is None
    finally:
      script_utils._TEMP_ROOT = original
