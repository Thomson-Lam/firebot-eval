#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_benchmark_context

print_stage_banner "Stage 1/5: Karpathy one-record overfit checks"
if [[ "$RUN_KARPATHY_CHECK" != "1" ]]; then
  echo "Skipping Karpathy checks (RUN_KARPATHY_CHECK=$RUN_KARPATHY_CHECK)"
  exit 0
fi

for algo in "${ALGO_ORDER[@]}"; do
  karpathy_check "$algo"
done

echo
echo "Stage 1 complete."
