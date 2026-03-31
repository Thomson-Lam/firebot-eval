#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is not installed or not on PATH."
  echo "Install: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-training-outputs/training_1/outputs/benchmark}"
RUN_LABEL="${RUN_LABEL:-final}"
EVAL_SEEDS_CSV="${EVAL_SEEDS_CSV:-11,22,33,44,55}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
AGENTS="${AGENTS:-ppo,a2c,dqn,greedy,random}"

DATA_VARIANT="${DATA_VARIANT:-v2}"
DATASET_BASE="data/static/${DATA_VARIANT}"
TRAIN_DATASET="${TRAIN_DATASET:-$DATASET_BASE/scenario_parameter_records_seeded_train.json}"
VAL_DATASET="${VAL_DATASET:-$DATASET_BASE/scenario_parameter_records_seeded_val.json}"
HOLDOUT_DATASET="${HOLDOUT_DATASET:-$DATASET_BASE/scenario_parameter_records_seeded_holdout.json}"

OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/$RUN_LABEL/eval}"
INCLUDE_FAMILY_HOLDOUT="${INCLUDE_FAMILY_HOLDOUT:-0}"
INCLUDE_TEMPORAL_HOLDOUT="${INCLUDE_TEMPORAL_HOLDOUT:-0}"
NO_NORMALIZED_BURN="${NO_NORMALIZED_BURN:-0}"

for dataset in "$TRAIN_DATASET" "$VAL_DATASET" "$HOLDOUT_DATASET"; do
  if [[ ! -f "$dataset" ]]; then
    echo "ERROR: Missing dataset '$dataset'."
    exit 1
  fi
done

IFS=',' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_CSV"
mkdir -p "$OUTPUT_DIR"

echo "== Benchmark evaluation configuration =="
echo "artifact_root          : $ARTIFACT_ROOT"
echo "run_label              : $RUN_LABEL"
echo "eval_seeds             : $EVAL_SEEDS_CSV"
echo "eval_episodes_per_seed : $EVAL_EPISODES"
echo "agents                 : $AGENTS"
echo "output_dir             : $OUTPUT_DIR"
echo "data_variant           : $DATA_VARIANT"
echo "train_dataset          : $TRAIN_DATASET"
echo "val_dataset            : $VAL_DATASET"
echo "holdout_dataset        : $HOLDOUT_DATASET"
echo

for seed in "${EVAL_SEEDS[@]}"; do
  seed_trimmed="${seed// /}"
  ppo_model="$ARTIFACT_ROOT/$RUN_LABEL/ppo/seed_${seed_trimmed}/best_model.zip"
  a2c_model="$ARTIFACT_ROOT/$RUN_LABEL/a2c/seed_${seed_trimmed}/best_model.zip"
  dqn_model="$ARTIFACT_ROOT/$RUN_LABEL/dqn/seed_${seed_trimmed}/best_model.zip"

  for model_path in "$ppo_model" "$a2c_model" "$dqn_model"; do
    if [[ ! -f "$model_path" ]]; then
      echo "ERROR: Missing model '$model_path'."
      echo "Run training first: ./scripts/run_benchmark_train.sh"
      exit 1
    fi
  done

  output_json="$OUTPUT_DIR/seed_${seed_trimmed}.json"
  echo "[EVAL] seed=$seed_trimmed -> $output_json"

  cmd=(
    uv run python -m src.models.evaluate_agents
    --agents "$AGENTS"
    --ppo-model "$ppo_model"
    --a2c-model "$a2c_model"
    --dqn-model "$dqn_model"
    --train-dataset "$TRAIN_DATASET"
    --val-dataset "$VAL_DATASET"
    --holdout-dataset "$HOLDOUT_DATASET"
    --seeds "$seed_trimmed"
    --episodes "$EVAL_EPISODES"
    --run-label "$RUN_LABEL"
    --output "$output_json"
  )

  if [[ "$INCLUDE_FAMILY_HOLDOUT" == "1" ]]; then
    cmd+=(--include-family-holdout)
  fi
  if [[ "$INCLUDE_TEMPORAL_HOLDOUT" == "1" ]]; then
    cmd+=(--include-temporal-holdout)
  fi
  if [[ "$NO_NORMALIZED_BURN" == "1" ]]; then
    cmd+=(--no-normalized-burn)
  fi

  "${cmd[@]}"
done

echo
echo "Evaluation complete. Results are in '$OUTPUT_DIR'."
