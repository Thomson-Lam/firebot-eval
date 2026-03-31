#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_benchmark_context

print_stage_banner "Stage 4/5: Validation-only pilot sweeps"
if [[ "$RUN_PILOT_SWEEP" != "1" ]]; then
  echo "Skipping pilot sweeps (RUN_PILOT_SWEEP=$RUN_PILOT_SWEEP)"
  exit 0
fi

for algo in "${ALGO_ORDER[@]}"; do
  pilot_sweep_algo "$algo"
done

echo
echo "Stage 4 complete."
