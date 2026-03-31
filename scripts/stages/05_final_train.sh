#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
init_benchmark_context

print_stage_banner "Stage 5/5: Full 5-seed benchmark training"
for algo in "${ALGO_ORDER[@]}"; do
  final_hparam_source="defaults"
  final_hparam_file="$ARTIFACT_ROOT/pilot_sweeps/$algo/pilot_winner.json"
  declare -a final_hparam_flags=()

  if [[ "$USE_PILOT_WINNERS" == "1" ]]; then
    if [[ -f "$final_hparam_file" ]]; then
      mapfile -t final_hparam_flags < <(load_winner_flags "$algo")
      final_hparam_source="pilot_winner"
    else
      echo "WARNING: Missing pilot winner file for $algo at $final_hparam_file; using defaults."
    fi
  fi

  case "$algo" in
    ppo) final_timesteps="$CANONICAL_PPO_TIMESTEPS" ;;
    a2c) final_timesteps="$CANONICAL_A2C_TIMESTEPS" ;;
    dqn) final_timesteps="$CANONICAL_DQN_TIMESTEPS" ;;
    *)
      echo "ERROR: unsupported algo '$algo'"
      exit 1
      ;;
  esac

  echo "[FINAL] $algo hyperparameters source: $final_hparam_source"
  if [[ ${#final_hparam_flags[@]} -gt 0 ]]; then
    echo "[FINAL] $algo hyperparameter flags: ${final_hparam_flags[*]}"
  fi

  for seed in "${FINAL_SEEDS[@]}"; do
    train_final "$algo" "$seed" "$final_timesteps" "${final_hparam_flags[@]}"
  done
done

echo
echo "Stage 5 complete. Artifacts are under '$ARTIFACT_ROOT'."
