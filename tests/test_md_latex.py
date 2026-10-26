"""Tests for scripts/md_latex.py (GitHub Markdown + KaTeX linter/emitter)."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "md_latex.py"
_spec = importlib.util.spec_from_file_location("md_latex", _SCRIPT)
check_md_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_md_math)


def _write(tmp_path, text):
  path = tmp_path / "doc.md"
  path.write_text(text, encoding="utf-8")
  return str(path)


def _rules(findings):
  return sorted({f.rule for f in findings})


def _check(tmp_path, text):
  return check_md_math.check_file(_write(tmp_path, text))


def _main(tmp_path, text, *args):
  return check_md_math._main([*args, _write(tmp_path, text)])


class TestR1BareDollar:

  def test_bare_dollar_flagged(self, tmp_path):
    findings = _check(tmp_path, "the $(1-d)$ term\n")
    assert _rules(findings) == ["R1"]

  def test_latex_paren_delimiter_flagged(self, tmp_path):
    findings = _check(tmp_path, "use \\(x\\) here\n")
    assert "R1" in _rules(findings)

  def test_extra_backtick_flagged(self, tmp_path):
    findings = _check(tmp_path, "a ``$x`$ span\n")
    assert "R1" in _rules(findings)

  def test_correct_form_clean(self, tmp_path):
    assert _check(tmp_path, "the $`(1-d)`$ term\n") == []

  def test_delimiter_documentation_not_flagged(self, tmp_path):
    # A line that *describes* the ``$ syntax is literal, not a math span.
    assert _check(tmp_path, "the `` ``$ `` pattern\n") == []


class TestR2CrossLineSpan:

  def test_span_crossing_newline_flagged(self, tmp_path):
    text = "weights $`\\alpha_{resp}, \\alpha_{feat},\n\\alpha_{rel}`$.\n"
    findings = _check(tmp_path, text)
    assert "R2" in _rules(findings)

  def test_single_line_span_clean(self, tmp_path):
    assert _check(tmp_path, "weights $`\\alpha_{rel}`$.\n") == []


class TestR3DisplayBlock:

  def test_missing_large_flagged(self, tmp_path):
    text = "\n$$\nx = 1\n$$\n\n"
    assert "R3" in _rules(_check(tmp_path, text))

  def test_dollar_not_alone_on_line_flagged(self, tmp_path):
    text = "\n$$ x = 1 $$\n\n"
    assert "R3" in _rules(_check(tmp_path, text))

  def test_blank_line_inside_flagged(self, tmp_path):
    text = "\n$$\n\\large\nx = 1\n\ny = 2\n$$\n\n"
    assert "R3" in _rules(_check(tmp_path, text))

  def test_well_formed_block_clean(self, tmp_path):
    text = "\n$$\n\\large\nx = 1\n$$\n\n"
    assert _check(tmp_path, text) == []


class TestR4Asterisk:

  def test_caret_asterisk_flagged(self, tmp_path):
    assert "R4" in _rules(_check(tmp_path, "$`V^*(s)`$\n"))

  def test_caret_braced_asterisk_flagged(self, tmp_path):
    assert "R4" in _rules(_check(tmp_path, "$`V^{*}(s)`$\n"))

  def test_ast_command_clean(self, tmp_path):
    assert _check(tmp_path, "$`V^{\\ast}(s)`$\n") == []


class TestR5BlockMarker:

  def test_leading_pipe_inside_display_flagged(self, tmp_path):
    text = "\n$$\n\\large\n|a| \\le r\n$$\n\n"
    assert "R5" in _rules(_check(tmp_path, text))

  def test_leading_digit_inside_display_flagged(self, tmp_path):
    text = "\n$$\n\\large\n1. x = 1\n$$\n\n"
    assert "R5" in _rules(_check(tmp_path, text))


class TestR6MalformedDelimiters:

  def test_double_backtick_flagged(self, tmp_path):
    assert "R6" in _rules(_check(tmp_path, "a ``$x`$ span\n"))

  def test_orphan_close_flagged(self, tmp_path):
    assert "R6" in _rules(_check(tmp_path, "uses $`n = 2`-3`$ here\n"))


class TestR7Emphasis:

  def test_math_close_then_emphasis_flagged(self, tmp_path):
    text = "the value $`x`$\n*emphasized* next\n"
    assert "R7" in _rules(_check(tmp_path, text))

  def test_single_star_straddle_flagged(self, tmp_path):
    # A single * opening before the span and closing right after it.
    assert "R7" in _rules(_check(tmp_path, "and *$`x`$* here\n"))

  def test_bold_wrapping_math_clean(self, tmp_path):
    # **bold** wrapping a math span renders correctly and is not flagged.
    assert _check(tmp_path, "- **pitch $`\\phi`$** below\n") == []


class TestR8BackslashComma:

  def test_display_single_backslash_flagged(self, tmp_path):
    text = "\n$$\n\\large\nx \\, y\n$$\n\n"
    assert "R8" in _rules(_check(tmp_path, text))

  def test_inline_double_backslash_flagged(self, tmp_path):
    assert "R8" in _rules(_check(tmp_path, "$`x \\\\, y`$\n"))

  def test_display_double_backslash_clean(self, tmp_path):
    text = "\n$$\n\\large\nx \\\\, y\n$$\n\n"
    assert _check(tmp_path, text) == []

  def test_display_single_brace_flagged(self, tmp_path):
    """Regression: single \\{ \\} in $$ loses the backslash (set braces
    vanish on GitHub).  See README.md:374."""
    text = "\n$$\n\\large\n\\mathcal{P}(i) = \\{\\, j \\,\\}\n$$\n\n"
    assert "R8" in _rules(_check(tmp_path, text))

  def test_display_single_pipe_flagged(self, tmp_path):
    text = "\n$$\n\\large\n\\left\\| x \\right\\|_2^2\n$$\n\n"
    assert "R8" in _rules(_check(tmp_path, text))

  def test_display_doubled_braces_clean(self, tmp_path):
    text = "\n$$\n\\large\n\\mathcal{P}(i) = \\\\{\\\\, j \\\\,\\\\}\n$$\n\n"
    assert _check(tmp_path, text) == []

  def test_display_command_backslash_not_flagged(self, tmp_path):
    """A backslash before a letter is a command, not an escape."""
    text = "\n$$\n\\large\n\\alpha + \\beta\n$$\n\n"
    assert _check(tmp_path, text) == []

  def test_fix_doubles_brace_in_display(self, tmp_path):
    path = _write(tmp_path, "\n$$\n\\large\n\\mathcal{P} = \\{x\\}\n$$\n\n")
    check_md_math.apply_fixes(path, check_md_math.check_file(path))
    text = Path(path).read_text(encoding="utf-8")
    assert "\\\\{x\\\\}" in text
    assert check_md_math.check_file(path) == []

  def test_fix_halves_backslash_in_inline(self, tmp_path):
    path = _write(tmp_path, "$`a \\\\, b`$\n")
    check_md_math.apply_fixes(path, check_md_math.check_file(path))
    text = Path(path).read_text(encoding="utf-8")
    assert text == "$`a \\, b`$\n"
    assert check_md_math.check_file(path) == []

  def test_inline_single_backslash_clean(self, tmp_path):
    assert _check(tmp_path, "$`x \\, y`$\n") == []


class TestR9UnicodeAndAscii:

  def test_unicode_in_math_flagged(self, tmp_path):
    assert "R9" in _rules(_check(tmp_path, "$`x \u2264 y`$\n"))

  def test_ascii_ge_in_math_flagged(self, tmp_path):
    assert "R9" in _rules(_check(tmp_path, "$`x >= y`$\n"))

  def test_unicode_in_prose_clean(self, tmp_path):
    # Prose unicode is legitimate English, never flagged.
    assert _check(tmp_path, "Python \u2265 3.9 and 6\u00d76 patches\n") == []

  def test_unicode_in_display_flagged(self, tmp_path):
    text = "\n$$\n\\large\nx \u2264 y\n$$\n\n"
    assert "R9" in _rules(_check(tmp_path, text))


class TestR10AbsoluteValue:

  def test_line_start_pipe_in_display_flagged(self, tmp_path):
    text = "\n$$\n\\large\n|a| \\le r\n$$\n\n"
    assert "R10" in _rules(_check(tmp_path, text))


class TestFix:

  def test_fix_caret_asterisk(self, tmp_path):
    path = _write(tmp_path, "$`V^*(s)`$\n")
    findings = check_md_math.check_file(path)
    check_md_math.apply_fixes(path, findings)
    assert Path(path).read_text(encoding="utf-8") == "$`V^{\\ast}(s)`$\n"

  def test_fix_inline_double_backslash(self, tmp_path):
    path = _write(tmp_path, "$`x \\\\, y`$\n")
    check_md_math.apply_fixes(path, check_md_math.check_file(path))
    assert Path(path).read_text(encoding="utf-8") == "$`x \\, y`$\n"

  def test_fix_is_idempotent(self, tmp_path):
    path = _write(tmp_path, "$`V^*(s)`$ and $`x \\\\, y`$\n")
    check_md_math.apply_fixes(path, check_md_math.check_file(path))
    first = Path(path).read_text(encoding="utf-8")
    assert check_md_math.check_file(path) == []
    check_md_math.apply_fixes(path, check_md_math.check_file(path))
    assert Path(path).read_text(encoding="utf-8") == first

  def test_fix_leaves_report_only_rules(self, tmp_path):
    text = "$`x >= y`$\n"
    path = _write(tmp_path, text)
    check_md_math.apply_fixes(path, check_md_math.check_file(path))
    assert Path(path).read_text(encoding="utf-8") == text


class TestCli:

  def test_clean_file_exit_zero(self, tmp_path):
    assert _main(tmp_path, "$`x`$\n") == 0

  def test_findings_exit_one(self, tmp_path):
    assert _main(tmp_path, "the $x$ term\n") == 1

  def test_usage_error_exit_two(self, tmp_path):
    path = _write(tmp_path, "$`x`$\n")
    assert check_md_math._main(["--bogus", path]) == 2


class TestRepoDocsClean:

  def test_repo_docs_have_no_findings(self):
    """The linter must be clean on this repository's own docs."""
    files = check_md_math._default_paths()
    assert files
    findings = []
    for path in files:
      findings.extend(check_md_math.check_file(path))
    assert findings == [], check_md_math.format_text(findings)
