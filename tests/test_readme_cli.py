"""Collect and execute the marked CLI example fences from the READMEs.

A markdown HTML comment (invisible on GitHub) immediately above a ```bash
fence opts that fence in.  Each marked fence is executed as a bounded smoke
test that writes everything under a removable temp dir.  The package is
imported via ``PYTHONPATH`` (no install); the ``genml-kit-*`` console scripts
are rewritten to ``python -m genml_kit...`` so the suite needs no editable
install.
"""

import pytest

from tests.readme_cli_harness import _README_GLOBS, REPO_ROOT, read_markers
from tests.readme_cli_runner import run_spec


def _collect_specs():
  specs = []
  for pattern in _README_GLOBS:
    for md in sorted(REPO_ROOT.glob(pattern)):
      specs.extend(read_markers(md))
  return specs


def _param(spec):
  marks = [pytest.mark.slow]
  # A gated fence runs offline against a fixture, so it carries no
  # network/service marks and stays in the default suite.
  if spec.network and not spec.gated:
    marks.append(pytest.mark.network)
  if spec.service and not spec.gated:
    # Umbrella marker selects every service fence; the per-service marker
    # (hf, s3, ...) selects one.  service implies network (both emitted).
    marks.append(pytest.mark.service)
    marks.append(getattr(pytest.mark, spec.service))
  if spec.launcher is not None:
    # A launcher (torchrun) fence is multi-process; opted out of the default
    # suite and run explicitly with ``pytest -m distributed``.
    marks.append(pytest.mark.distributed)
  return pytest.param(spec, marks=marks, id=spec.name)


_SPECS = _collect_specs()


@pytest.mark.parametrize("spec", [_param(s) for s in _SPECS])
def test_readme_cli(spec, tmp_path):
  if spec.skip:
    pytest.skip(spec.skip)
  run_spec(spec, tmp_path)
