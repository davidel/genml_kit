"""Tests for genml_kit.utils.env.getenv."""

import pytest

from genml_kit.utils.env import getenv


class TestGetenv:

  def test_unset_returns_default(self, monkeypatch):
    monkeypatch.delenv("GENML_TEST_UNSET", raising=False)
    assert getenv("GENML_TEST_UNSET") is None
    assert getenv("GENML_TEST_UNSET", defval="fallback") == "fallback"

  def test_blank_returns_default(self, monkeypatch):
    monkeypatch.setenv("GENML_TEST_BLANK", "   ")
    assert getenv("GENML_TEST_BLANK", defval=7) == 7

  def test_str_conversion(self, monkeypatch):
    monkeypatch.setenv("GENML_TEST_STR", "hello")
    assert getenv("GENML_TEST_STR", vtype=str) == "hello"

  def test_int_conversion(self, monkeypatch):
    monkeypatch.setenv("GENML_TEST_INT", "42")
    assert getenv("GENML_TEST_INT", vtype=int) == 42

  @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on"])
  def test_bool_true_tokens(self, monkeypatch, raw):
    monkeypatch.setenv("GENML_TEST_BOOL", raw)
    assert getenv("GENML_TEST_BOOL", vtype=bool) is True

  @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "No", "off"])
  def test_bool_false_tokens(self, monkeypatch, raw):
    monkeypatch.setenv("GENML_TEST_BOOL", raw)
    assert getenv("GENML_TEST_BOOL", vtype=bool) is False

  def test_bool_default_when_unset(self, monkeypatch):
    monkeypatch.delenv("GENML_TEST_BOOL", raising=False)
    assert getenv("GENML_TEST_BOOL", vtype=bool, defval=False) is False

  def test_bad_bool_is_fatal(self, monkeypatch):
    monkeypatch.setenv("GENML_TEST_BOOL", "maybe")
    with pytest.raises(RuntimeError, match="GENML_TEST_BOOL"):
      getenv("GENML_TEST_BOOL", vtype=bool)

  def test_bad_int_is_fatal(self, monkeypatch):
    monkeypatch.setenv("GENML_TEST_INT", "many")
    with pytest.raises(RuntimeError, match="GENML_TEST_INT"):
      getenv("GENML_TEST_INT", vtype=int)
