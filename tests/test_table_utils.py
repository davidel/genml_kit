"""Tests for genml_kit.table_utils.format_table."""

from genml_kit.utils.table import format_table


def test_basic_alignment():
  headers = ["Name", "Value"]
  rows = [["abc", "1"], ["ab", "22"]]
  lines = format_table(headers, rows)
  assert len(lines) == 4  # header + sep + 2 data rows
  # Header and data rows should have the same length.
  assert len(lines[0]) == len(lines[2])
  assert len(lines[0]) == len(lines[3])


def test_footer_alignment():
  headers = ["Name", "Count"]
  rows = [["abc", "100"], ["ab", "200"]]
  footer = ["TOTAL", "300"]
  lines = format_table(headers, rows, footer=footer)
  assert len(lines) == 5  # header + sep + 2 data + footer
  # All lines (except separator) should have the same length.
  line_len = len(lines[0])
  assert len(lines[2]) == line_len
  assert len(lines[3]) == line_len
  assert len(lines[4]) == line_len
  assert "TOTAL" in lines[4]
  assert "300" in lines[4]


def test_footer_widens_columns():
  """Footer values wider than any header or data cell widen the column."""
  headers = ["A", "B"]
  rows = [["x", "1"]]
  footer = ["long_footer_value", "2"]
  lines = format_table(headers, rows, footer=footer)
  # The 'A' column should be wide enough for 'long_footer_value'.
  assert "long_footer_value" in lines[3]


def test_left_aligned_footer():
  headers = ["Label", "Value"]
  rows = [["a", "1"]]
  footer = ["Total", "1"]
  aligns = ["left", "right"]
  lines = format_table(headers, rows, align=aligns, footer=footer)
  # Footer 'Total' should be left-aligned.
  assert "Total" in lines[3]
  # All data lines same length.
  assert len(lines[0]) == len(lines[2]) == len(lines[3])


def test_empty_rows():
  headers = ["A", "B"]
  footer = ["sum", "0"]
  lines = format_table(headers, [], footer=footer)
  assert len(lines) == 3  # header + sep + footer


def test_empty_headers_returns_empty():
  assert format_table([], []) == []
