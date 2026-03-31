#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_benchmark_context

print_stage_banner "Stage 3/5: Smoke evaluation"
uv run python -m src.models.evaluate_agents \
  --agents ppo,a2c,dqn,greedy,random \
  --ppo-model "$ARTIFACT_ROOT/smoke/ppo/seed_${SMOKE_SEED}/best_model.zip" \
  --a2c-model "$ARTIFACT_ROOT/smoke/a2c/seed_${SMOKE_SEED}/best_model.zip" \
  --dqn-model "$ARTIFACT_ROOT/smoke/dqn/seed_${SMOKE_SEED}/best_model.zip" \
  --seeds "$SMOKE_SEED" \
  --episodes "$SMOKE_EVAL_EPISODES" \
  --run-label smoke \
  --output "$ARTIFACT_ROOT/smoke/eval_smoke.json"

echo
echo "Stage 3 complete."
