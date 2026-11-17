#!/usr/bin/env bash
#
# Run one README-CLI pytest marker suite, tolerating exit code 5.
#
# A suite label names the *reason* its marked README examples are isolated
# from the default run (see ``addopts`` in pyproject.toml and the comment
# block above ``jobs:`` in ci.yml); it is not merely a ``pytest -m``
# expression.
#
# Usage: readme_cli_suite.sh <suite>
#   network      Examples that need the internet but no credentials: they
#                download public Hub models/datasets.  Selects
#                ``network and not service`` because ``service:`` implies
#                ``network:``; without that exclusion a service-marked example
#                would run twice (here and under ``service``).
#   distributed  The multi-process ``launcher: torchrun`` example.  Separate
#                because it needs a different launcher and its own
#                descendant-process reaper.
#   service      Examples needing a third-party service (hf, s3, r2, gcs,
#                github, wandb), reachable by credentials or an endpoint
#                override.  They SKIP -- not fail -- when neither is present,
#                so fork PRs stay green.
#   proxied      ``service and not hf``: service examples that talk to an
#                emulator or proxy rather than the real service.  Excludes hf
#                so the ``service`` step is not re-run.  Collects nothing
#                yet; it is a placeholder, hence the exit-code-5 tolerance
#                below.
#
# Exit code 5 from pytest means "no tests collected": not an error while the
# marker suites are still being populated.  Any other non-zero code is
# propagated so a real failure still fails CI.
set +e
case "${1}" in
  network)
    marker='network and not service'
    label='network'
    ;;
  distributed)
    marker='distributed'
    label='distributed'
    ;;
  service)
    marker='service'
    label='service'
    ;;
  proxied)
    marker='service and not hf'
    label='proxied'
    ;;
  *)
    echo "usage: ${0##*/} <network|distributed|service|proxied>" >&2
    exit 2
    ;;
esac
pytest -q -m "$marker"
code=$?
if [ "$code" -eq 5 ]; then
  echo "No $label README CLI tests collected yet."
  exit 0
fi
exit "$code"
