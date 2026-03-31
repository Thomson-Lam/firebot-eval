#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_benchmark_context

print_stage_banner "Stage 2/5: Smoke training and reproducibility canary"
for algo in "${ALGO_ORDER[@]}"; do
  train_smoke "$algo" "$ARTIFACT_ROOT"
  if [[ "$RUN_REPRO_CANARY" == "1" ]]; then
    repro_canary_smoke "$algo"
  fi
done

echo
echo "Stage 2 complete."
