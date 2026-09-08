"""Tests for script_utils (local + URL script loading)."""

import pytest

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
