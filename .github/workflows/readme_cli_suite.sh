#!/usr/bin/env bash
#
# Run one README-CLI pytest marker suite, tolerating exit code 5.
#
# Usage: readme_cli_suite.sh <suite>
#   network      tokenless network examples
#   distributed  multi-process (torchrun) examples
#   service      external-service examples (HF_TOKEN gated datasets)
#   proxied      proxied / emulated services (== service, minus hf, so the
#                gated step above is not re-run)
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
