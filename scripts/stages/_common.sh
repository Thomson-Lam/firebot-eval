#!/usr/bin/env bash

init_benchmark_context() {
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  cd "$ROOT_DIR"

  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is not installed!"
    exit 1
  fi

  ARTIFACT_ROOT="${ARTIFACT_ROOT:-outputs/benchmark}"
  SMOKE_TIMESTEPS="${SMOKE_TIMESTEPS:-20000}"
  SMOKE_SEED="${SMOKE_SEED:-11}"
  SMOKE_EVAL_EPISODES="${SMOKE_EVAL_EPISODES:-5}"
  KARPATHY_TIMESTEPS="${KARPATHY_TIMESTEPS:-10000}"
  KARPATHY_SEED="${KARPATHY_SEED:-11}"
  KARPATHY_FAMILY="${KARPATHY_FAMILY:-center,medium,A}"
  KARPATHY_CHECKPOINT_EVAL_EPISODES="${KARPATHY_CHECKPOINT_EVAL_EPISODES:-1}"
  PILOT_TIMESTEPS="${PILOT_TIMESTEPS:-40000}"
  PILOT_SEED="${PILOT_SEED:-11}"
  FINAL_SEEDS_CSV="${FINAL_SEEDS_CSV:-11,22,33,44,55}"
  ALGO_ORDER_CSV="${ALGO_ORDER_CSV:-ppo,a2c,dqn}"
  RUN_KARPATHY_CHECK="${RUN_KARPATHY_CHECK:-1}"
  RUN_PILOT_SWEEP="${RUN_PILOT_SWEEP:-1}"
  USE_PILOT_WINNERS="${USE_PILOT_WINNERS:-1}"
  RUN_REPRO_CANARY="${RUN_REPRO_CANARY:-1}"
  REPRO_CANARY_TOL="${REPRO_CANARY_TOL:-1e-9}"

  CANONICAL_CHECKPOINT_INTERVAL=20000
  CANONICAL_CHECKPOINT_EVAL_EPISODES=20
  CANONICAL_FINAL_EVAL_EPISODES=100
  CANONICAL_PPO_TIMESTEPS=200000
  CANONICAL_A2C_TIMESTEPS=200000
  CANONICAL_DQN_TIMESTEPS=200000

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
}

print_stage_banner() {
  local title="$1"
  echo
  echo "== $title =="
}

train_smoke() {
  local algo="$1"
  local artifact_root="$2"
  echo "[SMOKE] Training $algo (seed=$SMOKE_SEED, timesteps=$SMOKE_TIMESTEPS)"
  uv run python -m src.models.train_rl_agent \
    --algo "$algo" \
    --run-label smoke \
    --seed "$SMOKE_SEED" \
    --timesteps "$SMOKE_TIMESTEPS" \
    --train-dataset "$TRAIN_DATASET" \
    --val-dataset "$VAL_DATASET" \
    --holdout-dataset "$HOLDOUT_DATASET" \
    --checkpoint-interval "$CANONICAL_CHECKPOINT_INTERVAL" \
    --checkpoint-eval-episodes "$CANONICAL_CHECKPOINT_EVAL_EPISODES" \
    --final-eval-episodes "$CANONICAL_FINAL_EVAL_EPISODES" \
    --artifact-root "$artifact_root"
}

repro_canary_smoke() {
  local algo="$1"
  local canary_root="$ARTIFACT_ROOT/repro_canary"

  echo "[REPRO] Re-running smoke for $algo with identical seed/config"
  train_smoke "$algo" "$canary_root"

  uv run python scripts/canary.py \
    --baseline-root "$ARTIFACT_ROOT" \
    --candidate-root "$canary_root" \
    --run-label smoke \
    --algo "$algo" \
    --seed "$SMOKE_SEED" \
    --tol "$REPRO_CANARY_TOL"
}

build_single_record_dataset() {
  local input_path="$1"
  local output_path="$2"
  local split_name="$3"
  uv run python - "$input_path" "$output_path" "$split_name" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
split = sys.argv[3]

payload = json.loads(src.read_text())
records = payload.get("records", []) if isinstance(payload, dict) else payload
if not records:
    raise SystemExit(f"No records found in {src}")

records_sorted = sorted(records, key=lambda rec: float(rec.get("base_spread_prob", 1.0)))
record = dict(records_sorted[0])
record["split"] = split
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(
    json.dumps(
        {
            "schema_version": payload.get("schema_version", 3) if isinstance(payload, dict) else 3,
            "generated_at": payload.get("generated_at") if isinstance(payload, dict) else None,
            "split": split,
            "record_count": 1,
            "records": [record],
        },
        indent=2,
    )
)
print(f"Wrote one-record {split} dataset -> {dst}")
PY
}

karpathy_check() {
  local algo="$1"
  local karpathy_root="$ARTIFACT_ROOT/karpathy"
  local one_record_dir="$karpathy_root/one_record_datasets"
  local train_one="$one_record_dir/train_single.json"
  local val_one="$one_record_dir/val_single.json"
  local holdout_one="$one_record_dir/holdout_single.json"

  if [[ ! -f "$train_one" || ! -f "$val_one" || ! -f "$holdout_one" ]]; then
    build_single_record_dataset "$TRAIN_DATASET" "$train_one" "train"
    build_single_record_dataset "$VAL_DATASET" "$val_one" "val"
    build_single_record_dataset "$HOLDOUT_DATASET" "$holdout_one" "holdout"
  fi

  echo "[KARPATHY] Training $algo on one-record datasets (seed=$KARPATHY_SEED, timesteps=$KARPATHY_TIMESTEPS)"
  uv run python -m src.models.train_rl_agent \
    --algo "$algo" \
    --run-label karpathy \
    --seed "$KARPATHY_SEED" \
    --timesteps "$KARPATHY_TIMESTEPS" \
    --envs 1 \
    --train-family "$KARPATHY_FAMILY" \
    --val-family "$KARPATHY_FAMILY" \
    --train-dataset "$train_one" \
    --val-dataset "$val_one" \
    --holdout-dataset "$holdout_one" \
    --checkpoint-interval 1000 \
    --checkpoint-eval-episodes "$KARPATHY_CHECKPOINT_EVAL_EPISODES" \
    --final-eval-episodes 20 \
    --artifact-root "$ARTIFACT_ROOT"

  uv run python - "$ARTIFACT_ROOT/karpathy/$algo/seed_${KARPATHY_SEED}/checkpoint_metrics.json" "$algo" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
algo = sys.argv[2]
metrics = json.loads(path.read_text())
if not metrics:
    raise SystemExit(f"No checkpoint metrics found for {algo}: {path}")

first = metrics[0]["splits"]["train"]
last = metrics[-1]["splits"]["train"]
print(
    f"[OVERFIT][{algo}] train return {first['mean_return']:.2f} -> {last['mean_return']:.2f}; "
    f"train asset survival {first['asset_survival_rate']:.3f} -> {last['asset_survival_rate']:.3f}"
)
PY
}

pilot_sweep_algo() {
  local algo="$1"
  local base_root="$ARTIFACT_ROOT/pilot_sweeps/$algo"
  local cfg_id cfg_args
  local -a configs

  case "$algo" in
    ppo)
      configs=(
        "lr3e4_n512_ent001 --learning-rate 3e-4 --n-steps 512 --ent-coef 0.01"
        "lr1e4_n1024_ent005 --learning-rate 1e-4 --n-steps 1024 --ent-coef 0.005"
        "lr5e4_n256_ent002 --learning-rate 5e-4 --n-steps 256 --ent-coef 0.02"
      )
      ;;
    a2c)
      configs=(
        "lr7e4_n5_ent001 --learning-rate 7e-4 --n-steps 5 --ent-coef 0.01"
        "lr3e4_n20_ent005 --learning-rate 3e-4 --n-steps 20 --ent-coef 0.005"
        "lr1e3_n10_ent002 --learning-rate 1e-3 --n-steps 10 --ent-coef 0.02"
      )
      ;;
    dqn)
      configs=(
        "lr1e4_ef02_eps005_tu1000_buf100k --learning-rate 1e-4 --exploration-fraction 0.2 --exploration-final-eps 0.05 --target-update-interval 1000 --replay-buffer-size 100000"
        "lr3e4_ef03_eps01_tu500_buf50k --learning-rate 3e-4 --exploration-fraction 0.3 --exploration-final-eps 0.1 --target-update-interval 500 --replay-buffer-size 50000"
        "lr5e5_ef01_eps002_tu2000_buf200k --learning-rate 5e-5 --exploration-fraction 0.1 --exploration-final-eps 0.02 --target-update-interval 2000 --replay-buffer-size 200000"
      )
      ;;
    *)
      echo "ERROR: unsupported algo '$algo' for pilot sweep"
      exit 1
      ;;
  esac

  echo "[PILOT] $algo validation-focused pilot sweep"
  for spec in "${configs[@]}"; do
    cfg_id="${spec%% *}"
    cfg_args="${spec#* }"
    echo "[PILOT] $algo config=$cfg_id"
    # shellcheck disable=SC2086
    uv run python -m src.models.train_rl_agent \
      --algo "$algo" \
      --run-label pilot \
      --seed "$PILOT_SEED" \
      --timesteps "$PILOT_TIMESTEPS" \
      --train-dataset "$TRAIN_DATASET" \
      --val-dataset "$VAL_DATASET" \
      --holdout-dataset "$HOLDOUT_DATASET" \
      --checkpoint-interval "$CANONICAL_CHECKPOINT_INTERVAL" \
      --checkpoint-eval-episodes "$CANONICAL_CHECKPOINT_EVAL_EPISODES" \
      --final-eval-episodes "$CANONICAL_FINAL_EVAL_EPISODES" \
      --artifact-root "$base_root/$cfg_id" \
      $cfg_args
  done

  uv run python - "$base_root" "$algo" "$PILOT_SEED" "$PILOT_TIMESTEPS" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
algo = sys.argv[2]
seed = int(sys.argv[3])
pilot_timesteps = int(sys.argv[4])


def cli_flags_from_hyperparams(algo_name: str, hp: dict) -> list[str]:
    if algo_name in {"ppo", "a2c"}:
        return [
            "--learning-rate",
            str(hp["learning_rate"]),
            "--n-steps",
            str(hp["n_steps"]),
            "--ent-coef",
            str(hp["ent_coef"]),
        ]
    if algo_name == "dqn":
        return [
            "--learning-rate",
            str(hp["learning_rate"]),
            "--exploration-fraction",
            str(hp["exploration_fraction"]),
            "--exploration-final-eps",
            str(hp["exploration_final_eps"]),
            "--target-update-interval",
            str(int(hp["target_update_interval"])),
            "--replay-buffer-size",
            str(int(hp["buffer_size"])),
        ]
    raise ValueError(f"Unsupported algo for cli flag mapping: {algo_name}")


rows = []
for cfg_dir in sorted(base.iterdir()):
    if not cfg_dir.is_dir():
        continue
    best_path = cfg_dir / "pilot" / algo / f"seed_{seed}" / "best_checkpoint.json"
    run_cfg_path = cfg_dir / "pilot" / algo / f"seed_{seed}" / "config.json"
    if not best_path.exists() or not run_cfg_path.exists():
        continue
    payload = json.loads(best_path.read_text())
    run_cfg = json.loads(run_cfg_path.read_text())
    hp = run_cfg.get("hyperparameters", {})
    val = payload.get("val_metrics", {})
    rows.append(
        {
            "config": cfg_dir.name,
            "asset_survival": float(val.get("asset_survival_rate", 0.0)),
            "mean_return": float(val.get("mean_return", 0.0)),
            "containment": float(val.get("containment_success_rate", 0.0)),
            "step": int(payload.get("selected_train_steps", 0)),
            "hyperparameters": hp,
            "cli_flags": cli_flags_from_hyperparams(algo, hp),
        }
    )

rows.sort(key=lambda x: (x["asset_survival"], x["mean_return"], x["containment"]), reverse=True)
print(f"[PILOT][{algo}] leaderboard (validation checkpoint metric):")
for i, row in enumerate(rows, start=1):
    print(
        f"  {i:>2}. {row['config']:<36} "
        f"asset_survival={row['asset_survival']:.3f} "
        f"return={row['mean_return']:.2f} "
        f"containment={row['containment']:.3f} "
        f"step={row['step']}"
    )

if rows:
    winner = {
        "algo": algo,
        "pilot_seed": seed,
        "pilot_timesteps": pilot_timesteps,
        "selection_metric": "val.asset_survival_rate",
        "tie_breakers": ["val.mean_return", "val.containment_success_rate"],
        "selected": {
            "config": rows[0]["config"],
            "asset_survival": rows[0]["asset_survival"],
            "mean_return": rows[0]["mean_return"],
            "containment": rows[0]["containment"],
            "selected_checkpoint_step": rows[0]["step"],
            "hyperparameters": rows[0]["hyperparameters"],
            "cli_flags": rows[0]["cli_flags"],
        },
        "candidates": [
            {
                "config": row["config"],
                "asset_survival": row["asset_survival"],
                "mean_return": row["mean_return"],
                "containment": row["containment"],
                "selected_checkpoint_step": row["step"],
            }
            for row in rows
        ],
    }
    winner_path = base / "pilot_winner.json"
    winner_path.write_text(json.dumps(winner, indent=2))
    print(f"[PILOT][{algo}] selected candidate -> {rows[0]['config']}")
    print(f"[PILOT][{algo}] wrote winner file -> {winner_path}")
else:
    print(f"[PILOT][{algo}] no completed pilot runs found")
PY
}

load_winner_flags() {
  local algo="$1"
  local winner_path="$ARTIFACT_ROOT/pilot_sweeps/$algo/pilot_winner.json"
  if [[ ! -f "$winner_path" ]]; then
    return 0
  fi
  uv run python - "$winner_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
for token in payload.get("selected", {}).get("cli_flags", []):
    print(str(token))
PY
}

train_final() {
  local algo="$1"
  local seed="$2"
  local timesteps="$3"
  local -a extra_args=("${@:4}")
  echo "[FINAL] Training $algo (seed=$seed, timesteps=$timesteps)"
  uv run python -m src.models.train_rl_agent \
    --algo "$algo" \
    --run-label final \
    --seed "$seed" \
    --timesteps "$timesteps" \
    --train-dataset "$TRAIN_DATASET" \
    --val-dataset "$VAL_DATASET" \
    --holdout-dataset "$HOLDOUT_DATASET" \
    --checkpoint-interval "$CANONICAL_CHECKPOINT_INTERVAL" \
    --checkpoint-eval-episodes "$CANONICAL_CHECKPOINT_EVAL_EPISODES" \
    --final-eval-episodes "$CANONICAL_FINAL_EVAL_EPISODES" \
    --artifact-root "$ARTIFACT_ROOT" \
    "${extra_args[@]}"
}
