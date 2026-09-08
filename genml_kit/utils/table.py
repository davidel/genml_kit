"""Lightweight table formatter for aligned log output.

All functions return a list of strings (one per line) with no trailing
newline.  Column widths are computed automatically from the data.
"""


def format_table(headers, rows, align=None, prefix="  ", footer=None):
  """Build an aligned text table.

  Parameters
  ----------
  headers : list[str]
      Column header labels.
  rows : list[list[str]]
      One list of pre-formatted cell strings per data row.
  align : list[str] | None
      Per-column alignment: ``'left'`` or ``'right'`` (default).
      If *None* every column defaults to right-aligned.
  prefix : str
      Indentation prepended to every line.
  footer : list[str] | None
      Optional summary row rendered after the separator, using the
      same column alignment as data rows.

  Returns
  -------
  list[str]
      Lines: header, separator, data rows, optional footer.
  """
  if not headers:
    return []
  n_cols = len(headers)
  if align is None:
    align = ["right"] * n_cols
  # Compute column widths: max of header, all data cells, and footer.
  widths = [len(h) for h in headers]
  for row in rows:
    for i, cell in enumerate(row):
      widths[i] = max(widths[i], len(cell))
  if footer:
    for i, cell in enumerate(footer):
      widths[i] = max(widths[i], len(cell))

  def _fmt(text, col_idx):
    w = widths[col_idx]
    if align[col_idx] == "left":
      return f"{text:<{w}}"
    return f"{text:>{w}}"

  sep = prefix + "-" * (sum(widths) + n_cols - 1)
  hdr = prefix + " ".join(_fmt(h, i) for i, h in enumerate(headers))
  lines = [hdr, sep]
  for row in rows:
    lines.append(prefix + " ".join(_fmt(cell, i) for i, cell in enumerate(row)))
  if footer:
    lines.append(prefix + " ".join(_fmt(cell, i) for i, cell in enumerate(footer)))
  return lines
