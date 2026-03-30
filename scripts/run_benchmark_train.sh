#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is not installed or not on PATH."
  echo "Install: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-outputs/benchmark}"
# Default smoke length is one canonical checkpoint interval.
SMOKE_TIMESTEPS="${SMOKE_TIMESTEPS:-20000}"
SMOKE_SEED="${SMOKE_SEED:-11}"
SMOKE_EVAL_EPISODES="${SMOKE_EVAL_EPISODES:-5}"
FINAL_SEEDS_CSV="${FINAL_SEEDS_CSV:-11,22,33,44,55}"
ALGO_ORDER_CSV="${ALGO_ORDER_CSV:-ppo,a2c,dqn}"

TRAIN_DATASET="data/static/scenario_parameter_records_seeded_train.json"
VAL_DATASET="data/static/scenario_parameter_records_seeded_val.json"
HOLDOUT_DATASET="data/static/scenario_parameter_records_seeded_holdout.json"

for dataset in "$TRAIN_DATASET" "$VAL_DATASET" "$HOLDOUT_DATASET"; do
  if [[ ! -f "$dataset" ]]; then
    echo "ERROR: Missing dataset '$dataset'."
    echo "Run dataset build first: uv run python -m src.ingestion.static_dataset --target-count 50000 --raw-alberta-csv data/static/fp-historical-wildfire-data-2006-2025.csv"
    exit 1
  fi
done

IFS=',' read -r -a ALGO_ORDER <<< "$ALGO_ORDER_CSV"
IFS=',' read -r -a FINAL_SEEDS <<< "$FINAL_SEEDS_CSV"

echo "== Benchmark training configuration =="
echo "artifact_root      : $ARTIFACT_ROOT"
echo "algo_order         : $ALGO_ORDER_CSV"
echo "smoke_seed         : $SMOKE_SEED"
echo "smoke_timesteps    : $SMOKE_TIMESTEPS"
echo "smoke_eval_episodes: $SMOKE_EVAL_EPISODES"
echo "final_seeds        : $FINAL_SEEDS_CSV"
echo
echo "Note: full runs keep default trainer timesteps/envs/checkpoint cadence."
echo

train_smoke() {
  local algo="$1"
  echo "[SMOKE] Training $algo (seed=$SMOKE_SEED, timesteps=$SMOKE_TIMESTEPS)"
  uv run python -m src.models.train_rl_agent \
    --algo "$algo" \
    --run-label smoke \
    --seed "$SMOKE_SEED" \
    --timesteps "$SMOKE_TIMESTEPS" \
    --artifact-root "$ARTIFACT_ROOT"
}

train_final() {
  local algo="$1"
  local seed="$2"
  echo "[FINAL] Training $algo (seed=$seed, default timesteps/envs)"
  uv run python -m src.models.train_rl_agent \
    --algo "$algo" \
    --run-label final \
    --seed "$seed" \
    --artifact-root "$ARTIFACT_ROOT"
}

echo "== Stage 1/3: Algorithm smoke training =="
for algo in "${ALGO_ORDER[@]}"; do
  train_smoke "$algo"
done

echo
echo "== Stage 2/3: Smoke evaluation (load + score sanity check) =="
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
# TODO: Check seeds for correctness! 
echo "== Stage 3/3: Full 5-seed benchmark training =="
for algo in "${ALGO_ORDER[@]}"; do
  for seed in "${FINAL_SEEDS[@]}"; do
    train_final "$algo" "$seed"
  done
done

echo
echo "All runs finished. Artifacts are under '$ARTIFACT_ROOT'."
