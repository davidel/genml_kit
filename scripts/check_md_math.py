#!/usr/bin/env python3
"""Doc-math linter: catch GitHub Markdown + KaTeX rendering hazards.

GitHub runs its Markdown parser *before* handing math to KaTeX.  A number
of otherwise-innocent byte patterns therefore render as raw text instead of
math.  This script implements the checks R1-R10 documented in
``docs/MARKDOWN_LATEX.txt`` and reports violations as::

    file:line:col: [R<n>] message

The only supported inline form is ``$`...`$`` (dollar, backtick, content,
backtick, dollar).  The only supported display form is a ``$$`` fence with
``$$`` alone on its own line.

Regions are classified before any rule runs: fenced code blocks and inline
code spans are literals and are never checked; ``$$`` blocks are display
math; ``$`...`$`` spans are inline math; everything else is prose.  This
scoping is what makes the checks correct rather than noisy -- e.g. a
literal ``\\,`` inside a ``$$`` block is *correct*, while the same bytes
inside an inline span are a bug.

Exit codes: 0 = clean, 1 = findings, 2 = usage error.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import namedtuple

Finding = namedtuple('Finding', 'path line col rule message fixable')

# Two-character math delimiters (GitHub's only supported inline form).
_OPEN = '$`'
_CLOSE = '`$'

# Unicode math glyphs that have a LaTeX escape.  Only flagged in math
# position (see R9); prose usage of these is legitimate English.
UNICODE_MATH = {
    '×': '\\times',
    '÷': '\\div',
    '±': '\\pm',
    '≤': '\\le',
    '≥': '\\ge',
    '≠': '\\ne',
    '≈': '\\approx',
    '→': '\\to',
    '←': '\\leftarrow',
    '⇒': '\\Rightarrow',
    '⇔': '\\Leftrightarrow',
    '∑': '\\sum',
    '∏': '\\prod',
    '∫': '\\int',
    '√': '\\sqrt',
    '∞': '\\infty',
    '∂': '\\partial',
    '∈': '\\in',
    '∀': '\\forall',
    '∃': '\\exists',
    '⋅': '\\cdot',
    '∘': '\\circ',
    '⊕': '\\oplus',
    '⊗': '\\otimes',
    'π': '\\pi',
    'θ': '\\theta',
    'α': '\\alpha',
    'β': '\\beta',
    'γ': '\\gamma',
    'λ': '\\lambda',
    'μ': '\\mu',
    'σ': '\\sigma',
    'τ': '\\tau',
    'φ': '\\phi',
    'ω': '\\omega',
    'Δ': '\\Delta',
    'Σ': '\\Sigma',
    'Ω': '\\Omega',
}

# ASCII spellings that must not appear in math position (use \\ge/\\le/\\to).
ASCII_MATH = (('>=', '\\ge'), ('<=', '\\le'), ('->', '\\to'))

FENCE_RE = re.compile(r'^ {0,3}(`{3,}|~{3,})')
DISPLAY_TOGGLE_RE = re.compile(r'^ {0,3}\$\$ {0,3}$')
DISPLAY_MARKER_RE = re.compile(r'^ {0,3}([+\-*|#>]|\d+\.)')
DUP_BACKTICK_RE = re.compile(r'``+\$')
R4_RE = re.compile(r'\^\*|\^\{\*\}')

CODE = 'code'
MATH = 'math'
TEXT = 'text'


def _next_run(line, pos, run_len):
  """Index of the next backtick run of exactly ``run_len``, or -1."""
  n = len(line)
  j = pos
  while j < n:
    if line[j] == '`':
      j2 = j
      while j2 < n and line[j2] == '`':
        j2 += 1
      if j2 - j == run_len:
        return j
      j = j2
    else:
      j += 1
  return -1


def split_inline(line):
  """Split a line into (kind, start, end) spans of code / math / text.

  ``kind`` is one of ``CODE``, ``MATH`` or ``TEXT``.  An unterminated math
  open is returned as ``MATH`` running to end of line so R2 can flag it.
  """
  spans = []
  i = 0
  n = len(line)
  start = 0
  while i < n:
    if (line[i] == '$' and i + 1 < n and line[i + 1] == '`' and
        (i == 0 or line[i - 1] != '`')):
      if i > start:
        spans.append((TEXT, start, i))
      close = line.find(_CLOSE, i + 2)
      if close == -1:
        spans.append((MATH, i, n))
        return spans
      spans.append((MATH, i, close + 2))
      i = close + 2
      start = i
      continue
    if line[i] == '`':
      k = i
      while k < n and line[k] == '`':
        k += 1
      run = k - i
      close = _next_run(line, k, run)
      if close == -1:
        # An unterminated backtick run is not a code span; treat the rest
        # as text so R1/R6 can flag the malformed-delimiter bug.
        if i > start:
          spans.append((TEXT, start, i))
        spans.append((TEXT, i, n))
        return spans
      if i > start:
        spans.append((TEXT, start, i))
      spans.append((CODE, i, close + run))
      i = close + run
      start = i
      continue
    i += 1
  if start < n:
    spans.append((TEXT, start, n))
  return spans


def is_fence_start(line):
  return bool(FENCE_RE.match(line))


def is_fence_toggle(line, open_run):
  """Return True when ``line`` closes a fence opened with ``open_run``."""
  m = FENCE_RE.match(line)
  if not m:
    return False
  run = m.group(1)
  return run[0] == open_run[0] and len(run) >= len(open_run)


def classify(lines):
  """Yield (index, role, text) with role in {'fence', 'display', 'text'}."""
  in_fence = False
  fence_char = ''
  in_display = False
  for idx, line in enumerate(lines):
    if in_fence:
      if is_fence_toggle(line, fence_char * 3):
        in_fence = False
      yield idx, 'fence', line
      continue
    m = FENCE_RE.match(line)
    if m:
      in_fence = True
      fence_char = m.group(1)[0]
      yield idx, 'fence', line
      continue
    if DISPLAY_TOGGLE_RE.match(line):
      yield idx, 'display', line
      in_display = not in_display
      continue
    yield idx, ('display' if in_display else 'text'), line


def _emit(findings, path, idx, col, rule, message, fixable=False):
  findings.append(Finding(path, idx + 1, col + 1, rule, message, fixable))


def _check_r1(path, idx, line, spans, findings):
  """R1 - bare ``$...$`` / ``\\(...\\)`` and the `` ``$ `` extra backtick."""
  for kind, start, end in spans:
    if kind != TEXT:
      continue
    seg = line[start:end]
    for m in re.finditer(r'\$', seg):
      abs_idx = start + m.start()
      if abs_idx > 0 and line[abs_idx - 1] == '`':
        continue
      _emit(findings, path, idx, abs_idx, 'R1',
            'bare `$...$`; use $`...`$ (dollar-backtick)')
    for m in re.finditer(r'\\[()]', seg):
      _emit(findings, path, idx, start + m.start(), 'R1',
            'LaTeX inline delimiter \\(...\\) is not rendered; use $`...`$')
    for m in DUP_BACKTICK_RE.finditer(seg):
      _emit(findings, path, idx, start + m.start(), 'R1',
            'extra backtick before `$` breaks the span (``$...)', True)


def _check_r2(path, idx, line, spans, findings):
  """R2 - inline span whose closing `` `$ `` is not on the same line."""
  for kind, start, end in spans:
    if kind == MATH and line[end - 2:end] != _CLOSE:
      _emit(findings, path, idx, start, 'R2',
            'inline math span crosses a newline; keep $`...`$ on one line')


def _check_display_toggle(path, idx, line, spans, findings):
  """R3 - ``$$`` must be alone on its own line (nothing before or after)."""
  if DISPLAY_TOGGLE_RE.match(line):
    return
  for m in re.finditer(r'\$\$', line):
    kind = next((k for k, s, e in spans if s <= m.start() < e), None)
    if kind == TEXT:
      _emit(findings, path, idx, m.start(), 'R3',
            '$$ must be alone on its own line (nothing before or after)')


def _check_r4(path, idx, line, spans, findings):
  """R4 - ``^*`` / ``^{*}`` (must be ``^{\\ast}``)."""
  for kind, start, end in spans:
    if kind != MATH:
      continue
    for m in R4_RE.finditer(line[start:end]):
      _emit(findings, path, idx, start + m.start(), 'R4',
            'raw asterisk superscript; use ^{\\ast}', True)


def _check_r6(path, idx, line, spans, findings):
  """R6 - doubled backtick before ``$``, orphan `` `$ ``, stray backtick."""
  for kind, start, end in spans:
    if kind == MATH:
      body = line[start + 2:end - 2]
      for m in re.finditer(r'`', body):
        _emit(findings, path, idx, start + 2 + m.start(), 'R6',
              'stray backtick inside a math span (malformed delimiters)')
      continue
    if kind != TEXT:
      continue
    seg = line[start:end]
    for m in re.finditer(r'``+\$', seg):
      _emit(findings, path, idx, start + m.start(), 'R6',
            'doubled backtick before `$`; use a single backtick', True)
    for m in re.finditer(r'(?<!`)`\$', seg):
      _emit(findings, path, idx, start + m.start(), 'R6',
            'orphan `$ with no matching opener')


def _check_r7(path, idx, line, spans, next_line, findings):
  """R7 - single-``*`` emphasis straddling a math span.

  Bold (``**``) that *wraps* a math span renders correctly on GitHub and is
  not flagged; only a single ``*`` immediately adjacent to a math delimiter
  is the hazard described in ``docs/MARKDOWN_LATEX.txt`` section 7(a).
  """
  for kind, start, end in spans:
    if kind != MATH:
      continue
    if (end < len(line) and line[end] == '*' and
        (end + 1 >= len(line) or line[end + 1] != '*')):
      _emit(findings, path, idx, end, 'R7',
            'emphasis crosses a math span (math-close followed by *)')
    if (start > 0 and line[start - 1] == '*' and
        (start - 2 < 0 or line[start - 2] != '*')):
      _emit(findings, path, idx, start - 1, 'R7',
            'emphasis crosses a math span (* immediately before math-open)')
  stripped = line.rstrip()
  if stripped.endswith(_CLOSE) and next_line is not None:
    nxt = next_line.lstrip()
    if nxt.startswith('**') or nxt.startswith('*'):
      _emit(findings, path, idx, max(0,
                                     len(stripped) - 2), 'R7',
            'math-close at end of line followed by emphasis on next line')


def _check_math_unicode(path, idx, line, spans, findings):
  """R9 - raw unicode math symbols / ASCII spellings in math position."""
  for kind, start, end in spans:
    if kind != MATH:
      continue
    seg = line[start:end]
    for off, ch in enumerate(seg):
      if ch in UNICODE_MATH:
        _emit(findings, path, idx, start + off, 'R9',
              f'raw unicode math symbol {ch!r}; use {UNICODE_MATH[ch]}')
    for token, repl in ASCII_MATH:
      pos = seg.find(token)
      while pos != -1:
        _emit(findings, path, idx, start + pos, 'R9',
              f'ASCII {token!r} in math; use {repl}')
        pos = seg.find(token, pos + 1)


def _check_display_line(path, idx, line, findings):
  """R3/R5/R8/R9/R10 for a single line inside a ``$$`` block."""
  if DISPLAY_MARKER_RE.match(line):
    marker = DISPLAY_MARKER_RE.match(line).group(1)
    _emit(findings, path, idx, 0, 'R5',
          f'line inside $$ starts with a Markdown block marker ({marker!r})')
  if re.match(r'^ {0,3}\|.*\|', line):
    _emit(findings, path, idx, 0, 'R10',
          'absolute value at line start; use \\lvert...\\rvert')
  if re.search(r'(?<!\\)\\,', line):
    _emit(findings, path, idx, line.find('\\,'), 'R8',
          'single backslash thin space inside $$; write \\\\,', True)
  for off, ch in enumerate(line):
    if ch in UNICODE_MATH:
      _emit(findings, path, idx, off, 'R9',
            f'raw unicode math symbol {ch!r}; use {UNICODE_MATH[ch]}')
  for token, repl in ASCII_MATH:
    pos = line.find(token)
    while pos != -1:
      _emit(findings, path, idx, pos, 'R9', f'ASCII {token!r} in math; use {repl}')
      pos = line.find(token, pos + 1)


def _check_display_block(path, lines, roles, findings):
  """R3 - fence placement, \\large directive and blank-line hygiene."""
  toggles = [
      i for i in range(len(lines))
      if roles[i] == 'display' and DISPLAY_TOGGLE_RE.match(lines[i])
  ]
  for start, end in zip(toggles[0::2], toggles[1::2]):
    if start + 1 < len(lines):
      first = lines[start + 1]
      if not first.strip().startswith('\\large'):
        _emit(findings, path, start + 1, 0, 'R3',
              'first line after $$ must be the \\large directive')
    if start > 0 and lines[start - 1].strip() != '':
      _emit(findings, path, start, 0, 'R3', 'blank line required before $$')
    if end + 1 < len(lines) and lines[end + 1].strip() != '':
      _emit(findings, path, end, 0, 'R3', 'blank line required after $$')
    for i in range(start + 1, end):
      if lines[i].strip() == '':
        _emit(findings, path, i, 0, 'R3', 'blank line inside $$ block')


def _check_inline_r8(path, idx, line, spans, findings):
  """R8 - ``\\,`` (double) is wrong inside an inline ``$`...`$`` span."""
  for kind, start, end in spans:
    if kind != MATH:
      continue
    seg = line[start:end]
    pos = seg.find('\\\\,')
    while pos != -1:
      _emit(findings, path, idx, start + pos, 'R8',
            'double-backslash thin space inside inline $`...`$; use \\,', True)
      pos = seg.find('\\\\,', pos + 1)


def check_file(path):
  """Return the list of findings for ``path``."""
  findings = []
  with open(path, encoding='utf-8') as fh:
    lines = fh.read().split('\n')
  roles = [role for _, role, _ in classify(lines)]
  for idx, line in enumerate(lines):
    role = roles[idx]
    if role == 'fence':
      continue
    if role == 'display':
      if DISPLAY_TOGGLE_RE.match(line):
        continue
      _check_display_line(path, idx, line, findings)
      continue
    spans = split_inline(line)
    _check_display_toggle(path, idx, line, spans, findings)
    nxt = lines[idx + 1] if idx + 1 < len(lines) else None
    _check_r1(path, idx, line, spans, findings)
    _check_r2(path, idx, line, spans, findings)
    _check_r4(path, idx, line, spans, findings)
    _check_r6(path, idx, line, spans, findings)
    _check_r7(path, idx, line, spans, nxt, findings)
    _check_inline_r8(path, idx, line, spans, findings)
    _check_math_unicode(path, idx, line, spans, findings)
  _check_display_block(path, lines, roles, findings)
  findings.sort(key=lambda f: (f.line, f.col, f.rule))
  return findings


def _iter_md(paths):
  """Expand PATHS (files/dirs) to a sorted list of ``*.md`` files."""
  out = []
  for p in paths:
    if os.path.isdir(p):
      for root, _dirs, files in os.walk(p):
        for name in sorted(files):
          if name.endswith('.md'):
            out.append(os.path.join(root, name))
    elif os.path.isfile(p):
      out.append(p)
  return sorted(set(out))


def _default_paths():
  """All git-tracked ``*.md`` files, falling back to a filesystem walk."""
  try:
    res = subprocess.run(['git', 'ls-files', '*.md'],
                         capture_output=True,
                         text=True,
                         check=True)
    return [ln for ln in res.stdout.split('\n') if ln]
  except (OSError, subprocess.CalledProcessError):
    return _iter_md(['.'])


def _fix_line(line, findings_on_line):
  """Apply the safe fixes for one line, right-to-left by column."""
  new = line
  for f in sorted(findings_on_line, key=lambda x: x.col, reverse=True):
    col = f.col - 1
    if f.rule == 'R4':
      if new[col:col + 2] == '^*':
        new = new[:col] + '^{\\ast}' + new[col + 2:]
      elif new[col:col + 5] == '^{*}':
        new = new[:col] + '^{\\ast}' + new[col + 5:]
    elif f.rule in ('R1', 'R6'):
      m = re.compile(r'``+\$').match(new, col)
      if m:
        new = new[:col] + '`$' + new[m.end():]
    elif f.rule == 'R8':
      if new[col:col + 3] == '\\\\,':
        new = new[:col] + '\\,' + new[col + 3:]
      elif new[col:col + 2] == '\\,':
        new = new[:col] + '\\\\,' + new[col + 2:]
  return new


def apply_fixes(path, findings):
  """Apply the safe, idempotent subset of fixes in place.  Returns a count."""
  fixable = [f for f in findings if f.fixable]
  if not fixable:
    return 0
  with open(path, encoding='utf-8') as fh:
    lines = fh.read().split('\n')
  by_line = {}
  for f in fixable:
    by_line.setdefault(f.line - 1, []).append(f)
  applied = 0
  for i, line_findings in by_line.items():
    new = _fix_line(lines[i], line_findings)
    if new != lines[i]:
      lines[i] = new
      applied += len(line_findings)
  if applied:
    with open(path, 'w', encoding='utf-8') as fh:
      fh.write('\n'.join(lines))
  return applied


def format_text(findings):
  """Human-readable ``file:line:col: [R<n>] message`` lines."""
  return '\n'.join(
      f'{f.path}:{f.line}:{f.col}: [{f.rule}] {f.message}' for f in findings)


def _main(argv=None):
  parser = argparse.ArgumentParser(
      description='Check GitHub Markdown + KaTeX math rendering hazards.')
  parser.add_argument('paths',
                      nargs='*',
                      help='files or directories to check (default: all *.md)')
  parser.add_argument('--check', action='store_true', help='report findings (default)')
  parser.add_argument('--fix',
                      action='store_true',
                      help='apply the safe, idempotent subset of fixes in place')
  parser.add_argument('--json', action='store_true', help='machine-readable output')
  parser.add_argument('--quiet', action='store_true', help='summary only')
  args, unknown = parser.parse_known_args(argv)
  if unknown:
    sys.stderr.write(f'check_md_math: unknown arguments: {" ".join(unknown)}\n')
    return 2

  files = _iter_md(args.paths) if args.paths else _default_paths()
  all_findings = []
  for path in files:
    try:
      findings = check_file(path)
    except OSError as exc:
      sys.stderr.write(f'check_md_math: {exc}\n')
      return 2
    if args.fix:
      fixed = apply_fixes(path, findings)
      findings = check_file(path)
      if fixed and not args.quiet:
        sys.stderr.write(f'{path}: applied {fixed} fix(es)\n')
    all_findings.extend(findings)

  if args.json:
    print(json.dumps([f._asdict() for f in all_findings], indent=2))
  elif args.quiet:
    print(f'{len(all_findings)} finding(s) in {len(files)} file(s)')
  else:
    if all_findings:
      print(format_text(all_findings))
    print(f'{len(all_findings)} finding(s) in {len(files)} file(s)')
  return 1 if all_findings else 0


if __name__ == '__main__':
  sys.exit(_main())
