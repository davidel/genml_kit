"""Tests for the generic decorator-aware registry (utils/registry.py).

Covers the four decorator forms (bare class, named class, keyword-named
function, bare function), lookup/build/contains/unregister, and the
ValueError taxonomy for duplicate and unknown names.
"""

import pytest

from genml_kit.utils.registry import Registry


class TestRegisterForms:

  def test_bare_class_decorator_uses_NAME(self):
    registry = Registry("thing")

    @registry.register
    class Widget:
      NAME = "widget"

    assert registry.get("widget") is Widget
    assert not registry.contains("Widget")

  def test_named_class_decorator(self):
    registry = Registry("thing")

    @registry.register("gadget")
    class Gadget:
      pass

    assert registry.get("gadget") is Gadget

  def test_keyword_named_function_decorator(self):
    registry = Registry("thing")

    @registry.register(name="make_widget")
    def make_widget():
      return "widget"

    assert registry.get("make_widget") is make_widget

  def test_bare_function_decorator_uses_name(self):
    registry = Registry("thing")

    @registry.register
    def sprocket():
      return "sprocket"

    assert registry.get("sprocket") is sprocket

  def test_direct_call_with_name(self):
    registry = Registry("thing")

    def fn():
      return "x"

    registry.register(fn, name="fn")
    assert registry.get("fn") is fn

  def test_decorator_returns_original_object(self):
    """The decorator must return the class unchanged (name preserved)."""
    registry = Registry("thing")

    @registry.register("widget")
    class Widget:
      NAME = "widget"

    assert Widget.NAME == "widget"


class TestLookup:

  def test_list_names_sorted(self):
    registry = Registry("thing")

    @registry.register
    class Beta:
      NAME = "beta"

    @registry.register
    class Alpha:
      NAME = "alpha"

    assert registry.list_names() == ["alpha", "beta"]

  def test_contains(self):
    registry = Registry("thing")

    @registry.register
    class Widget:
      NAME = "widget"

    assert registry.contains("widget") is True
    assert registry.contains("nope") is False

  def test_build_instantiates(self):
    registry = Registry("thing")

    @registry.register
    class Widget:
      NAME = "widget"

      def __init__(self, size):
        self.size = size

    assert registry.build("widget", size=3).size == 3


class TestErrors:

  def test_duplicate_raises_value_error(self):
    registry = Registry("thing")

    @registry.register
    class Widget:
      NAME = "widget"

    with pytest.raises(ValueError, match="Duplicate thing name 'widget'"):

      @registry.register
      class WidgetAgain:
        NAME = "widget"

  def test_unknown_get_raises_value_error(self):
    registry = Registry("thing")

    @registry.register
    class Widget:
      NAME = "widget"

    with pytest.raises(ValueError, match="Unknown thing 'nope'") as excinfo:
      registry.get("nope")
    assert "widget" in str(excinfo.value)


class TestUnregister:

  def test_unregister_removes(self):
    registry = Registry("thing")

    @registry.register
    class Widget:
      NAME = "widget"

    registry.unregister("widget")
    assert not registry.contains("widget")

  def test_unregister_missing_is_noop(self):
    registry = Registry("thing")
    registry.unregister("nope")  # must not raise
