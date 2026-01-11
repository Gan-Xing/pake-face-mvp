#!/usr/bin/env bash

# Aggregate lint and type-check results without stopping at first failure.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

status=0

run_step() {
  local title="$1"
  shift

  echo "================================================================"
  echo ">>> ${title}"
  echo "================================================================"

  "$@"
  local code=$?

  if [[ $code -ne 0 ]]; then
    status=$code
    echo ">>> ${title} failed (exit ${code})"
  else
    echo ">>> ${title} passed"
  fi

  echo
}

run_step "ESLint (eslint)" env ESLINT_USE_FLAT_CONFIG=true npx eslint .
run_step "TypeScript (tsc --noEmit)" npx tsc --noEmit

exit $status
