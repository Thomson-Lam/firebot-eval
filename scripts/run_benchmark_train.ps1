Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "'uv' is not installed or not on PATH. Install: https://docs.astral.sh/uv/getting-started/installation/"
}

$ArtifactRoot = if ($env:ARTIFACT_ROOT) { $env:ARTIFACT_ROOT } else { "outputs/benchmark" }
$SmokeTimesteps = if ($env:SMOKE_TIMESTEPS) { [int]$env:SMOKE_TIMESTEPS } else { 20000 }
$SmokeSeed = if ($env:SMOKE_SEED) { [int]$env:SMOKE_SEED } else { 11 }
$SmokeEvalEpisodes = if ($env:SMOKE_EVAL_EPISODES) { [int]$env:SMOKE_EVAL_EPISODES } else { 5 }
$KarpathyTimesteps = if ($env:KARPATHY_TIMESTEPS) { [int]$env:KARPATHY_TIMESTEPS } else { 10000 }
$KarpathySeed = if ($env:KARPATHY_SEED) { [int]$env:KARPATHY_SEED } else { 11 }
$PilotTimesteps = if ($env:PILOT_TIMESTEPS) { [int]$env:PILOT_TIMESTEPS } else { 40000 }
$PilotSeed = if ($env:PILOT_SEED) { [int]$env:PILOT_SEED } else { 11 }
$FinalSeedsCsv = if ($env:FINAL_SEEDS_CSV) { $env:FINAL_SEEDS_CSV } else { "11,22,33,44,55" }
$AlgoOrderCsv = if ($env:ALGO_ORDER_CSV) { $env:ALGO_ORDER_CSV } else { "ppo,a2c,dqn" }
$RunKarpathyCheck = if ($env:RUN_KARPATHY_CHECK) { [int]$env:RUN_KARPATHY_CHECK } else { 1 }
$RunPilotSweep = if ($env:RUN_PILOT_SWEEP) { [int]$env:RUN_PILOT_SWEEP } else { 1 }
$UsePilotWinners = if ($env:USE_PILOT_WINNERS) { [int]$env:USE_PILOT_WINNERS } else { 1 }

# Frozen canonical benchmark protocol values.
$CanonicalCheckpointInterval = 20000
$CanonicalCheckpointEvalEpisodes = 20
$CanonicalFinalEvalEpisodes = 100
$CanonicalPpoTimesteps = 200000
$CanonicalA2cTimesteps = 200000
$CanonicalDqnTimesteps = 200000

$TrainDataset = "data/static/scenario_parameter_records_seeded_train.json"
$ValDataset = "data/static/scenario_parameter_records_seeded_val.json"
$HoldoutDataset = "data/static/scenario_parameter_records_seeded_holdout.json"

foreach ($dataset in @($TrainDataset, $ValDataset, $HoldoutDataset)) {
    if (-not (Test-Path $dataset)) {
        Write-Error "Missing dataset '$dataset'. Run dataset build first: uv run python -m src.ingestion.static_dataset --target-count 50000 --raw-alberta-csv data/static/fp-historical-wildfire-data-2006-2025.csv"
    }
}

$AlgoOrder = $AlgoOrderCsv -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
$FinalSeeds = $FinalSeedsCsv -split "," | ForEach-Object { [int]$_.Trim() }

Write-Host "== Benchmark training configuration =="
Write-Host "artifact_root        : $ArtifactRoot"
Write-Host "algo_order           : $AlgoOrderCsv"
Write-Host "smoke_seed           : $SmokeSeed"
Write-Host "smoke_timesteps      : $SmokeTimesteps"
Write-Host "smoke_eval_episodes  : $SmokeEvalEpisodes"
Write-Host "karpathy_seed        : $KarpathySeed"
Write-Host "karpathy_timesteps   : $KarpathyTimesteps"
Write-Host "pilot_seed           : $PilotSeed"
Write-Host "pilot_timesteps      : $PilotTimesteps"
Write-Host "final_seeds          : $FinalSeedsCsv"
Write-Host "run_karpathy_check   : $RunKarpathyCheck"
Write-Host "run_pilot_sweep      : $RunPilotSweep"
Write-Host "use_pilot_winners    : $UsePilotWinners"
Write-Host "canonical_ckpt_int   : $CanonicalCheckpointInterval"
Write-Host "canonical_ckpt_eval  : $CanonicalCheckpointEvalEpisodes"
Write-Host "canonical_final_eval : $CanonicalFinalEvalEpisodes"
Write-Host ""

function Invoke-SmokeTrain {
    param([Parameter(Mandatory = $true)][string]$Algo)

    Write-Host "[SMOKE] Training $Algo (seed=$SmokeSeed, timesteps=$SmokeTimesteps)"
    & uv @(
        "run", "python", "-m", "src.models.train_rl_agent",
        "--algo", $Algo,
        "--run-label", "smoke",
        "--seed", "$SmokeSeed",
        "--timesteps", "$SmokeTimesteps",
        "--train-dataset", $TrainDataset,
        "--val-dataset", $ValDataset,
        "--holdout-dataset", $HoldoutDataset,
        "--checkpoint-interval", "$CanonicalCheckpointInterval",
        "--checkpoint-eval-episodes", "$CanonicalCheckpointEvalEpisodes",
        "--final-eval-episodes", "$CanonicalFinalEvalEpisodes",
        "--artifact-root", $ArtifactRoot
    )
}

function Build-SingleRecordDataset {
    param(
        [Parameter(Mandatory = $true)][string]$InputPath,
        [Parameter(Mandatory = $true)][string]$OutputPath,
        [Parameter(Mandatory = $true)][string]$SplitName
    )

    @'
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

record = dict(records[0])
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
'@ | & uv run python - $InputPath $OutputPath $SplitName
}

function Invoke-KarpathyCheck {
    param([Parameter(Mandatory = $true)][string]$Algo)

    $KarpathyRoot = Join-Path -Path $ArtifactRoot -ChildPath "karpathy"
    $OneRecordDir = Join-Path -Path $KarpathyRoot -ChildPath "one_record_datasets"
    $TrainOne = Join-Path -Path $OneRecordDir -ChildPath "train_single.json"
    $ValOne = Join-Path -Path $OneRecordDir -ChildPath "val_single.json"
    $HoldoutOne = Join-Path -Path $OneRecordDir -ChildPath "holdout_single.json"

    if ((-not (Test-Path $TrainOne)) -or (-not (Test-Path $ValOne)) -or (-not (Test-Path $HoldoutOne))) {
        Build-SingleRecordDataset -InputPath $TrainDataset -OutputPath $TrainOne -SplitName "train"
        Build-SingleRecordDataset -InputPath $ValDataset -OutputPath $ValOne -SplitName "val"
        Build-SingleRecordDataset -InputPath $HoldoutDataset -OutputPath $HoldoutOne -SplitName "holdout"
    }

    Write-Host "[KARPATHY] Training $Algo on one-record datasets (seed=$KarpathySeed, timesteps=$KarpathyTimesteps)"
    & uv @(
        "run", "python", "-m", "src.models.train_rl_agent",
        "--algo", $Algo,
        "--run-label", "karpathy",
        "--seed", "$KarpathySeed",
        "--timesteps", "$KarpathyTimesteps",
        "--envs", "1",
        "--train-dataset", $TrainOne,
        "--val-dataset", $ValOne,
        "--holdout-dataset", $HoldoutOne,
        "--checkpoint-interval", "1000",
        "--checkpoint-eval-episodes", "10",
        "--final-eval-episodes", "20",
        "--artifact-root", $ArtifactRoot
    )

    $CheckpointPath = Join-Path -Path $ArtifactRoot -ChildPath "karpathy/$Algo/seed_$KarpathySeed/checkpoint_metrics.json"
    @'
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
    f"[KARPATHY][{algo}] train return {first['mean_return']:.2f} -> {last['mean_return']:.2f}; "
    f"train asset survival {first['asset_survival_rate']:.3f} -> {last['asset_survival_rate']:.3f}"
)
'@ | & uv run python - $CheckpointPath $Algo
}

function Get-PilotConfigs {
    param([Parameter(Mandatory = $true)][string]$Algo)

    switch ($Algo) {
        "ppo" {
            return @(
                @{ Id = "lr3e4_n512_ent001"; Flags = @("--learning-rate", "3e-4", "--n-steps", "512", "--ent-coef", "0.01") },
                @{ Id = "lr1e4_n1024_ent005"; Flags = @("--learning-rate", "1e-4", "--n-steps", "1024", "--ent-coef", "0.005") },
                @{ Id = "lr5e4_n256_ent002"; Flags = @("--learning-rate", "5e-4", "--n-steps", "256", "--ent-coef", "0.02") }
            )
        }
        "a2c" {
            return @(
                @{ Id = "lr7e4_n5_ent001"; Flags = @("--learning-rate", "7e-4", "--n-steps", "5", "--ent-coef", "0.01") },
                @{ Id = "lr3e4_n20_ent005"; Flags = @("--learning-rate", "3e-4", "--n-steps", "20", "--ent-coef", "0.005") },
                @{ Id = "lr1e3_n10_ent002"; Flags = @("--learning-rate", "1e-3", "--n-steps", "10", "--ent-coef", "0.02") }
            )
        }
        "dqn" {
            return @(
                @{ Id = "lr1e4_ef02_eps005_tu1000_buf100k"; Flags = @("--learning-rate", "1e-4", "--exploration-fraction", "0.2", "--exploration-final-eps", "0.05", "--target-update-interval", "1000", "--replay-buffer-size", "100000") },
                @{ Id = "lr3e4_ef03_eps01_tu500_buf50k"; Flags = @("--learning-rate", "3e-4", "--exploration-fraction", "0.3", "--exploration-final-eps", "0.1", "--target-update-interval", "500", "--replay-buffer-size", "50000") },
                @{ Id = "lr5e5_ef01_eps002_tu2000_buf200k"; Flags = @("--learning-rate", "5e-5", "--exploration-fraction", "0.1", "--exploration-final-eps", "0.02", "--target-update-interval", "2000", "--replay-buffer-size", "200000") }
            )
        }
        default {
            throw "Unsupported algo '$Algo' for pilot sweep"
        }
    }
}

function Invoke-PilotSweep {
    param([Parameter(Mandatory = $true)][string]$Algo)

    $BaseRoot = Join-Path -Path $ArtifactRoot -ChildPath "pilot_sweeps/$Algo"
    $Configs = Get-PilotConfigs -Algo $Algo

    Write-Host "[PILOT] $Algo validation-focused pilot sweep"
    foreach ($cfg in $Configs) {
        $cfgId = [string]$cfg.Id
        $cfgFlags = [string[]]$cfg.Flags
        Write-Host "[PILOT] $Algo config=$cfgId"

        $cmdArgs = @(
            "run", "python", "-m", "src.models.train_rl_agent",
            "--algo", $Algo,
            "--run-label", "pilot",
            "--seed", "$PilotSeed",
            "--timesteps", "$PilotTimesteps",
            "--train-dataset", $TrainDataset,
            "--val-dataset", $ValDataset,
            "--holdout-dataset", $HoldoutDataset,
            "--checkpoint-interval", "$CanonicalCheckpointInterval",
            "--checkpoint-eval-episodes", "$CanonicalCheckpointEvalEpisodes",
            "--final-eval-episodes", "$CanonicalFinalEvalEpisodes",
            "--artifact-root", "$BaseRoot/$cfgId"
        )
        $cmdArgs += $cfgFlags
        & uv @cmdArgs
    }

    @'
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
'@ | & uv run python - $BaseRoot $Algo $PilotSeed $PilotTimesteps
}

function Get-WinnerFlags {
    param([Parameter(Mandatory = $true)][string]$Algo)

    $winnerPath = Join-Path -Path $ArtifactRoot -ChildPath "pilot_sweeps/$Algo/pilot_winner.json"
    if (-not (Test-Path $winnerPath)) {
        return @()
    }

    $winner = Get-Content $winnerPath -Raw | ConvertFrom-Json -AsHashtable
    if ($null -eq $winner.selected) {
        return @()
    }
    if ($null -eq $winner.selected.cli_flags) {
        return @()
    }
    return [string[]]$winner.selected.cli_flags
}

function Invoke-FinalTrain {
    param(
        [Parameter(Mandatory = $true)][string]$Algo,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][int]$Timesteps,
        [Parameter()][string[]]$ExtraArgs = @()
    )

    Write-Host "[FINAL] Training $Algo (seed=$Seed, timesteps=$Timesteps)"
    $cmdArgs = @(
        "run", "python", "-m", "src.models.train_rl_agent",
        "--algo", $Algo,
        "--run-label", "final",
        "--seed", "$Seed",
        "--timesteps", "$Timesteps",
        "--train-dataset", $TrainDataset,
        "--val-dataset", $ValDataset,
        "--holdout-dataset", $HoldoutDataset,
        "--checkpoint-interval", "$CanonicalCheckpointInterval",
        "--checkpoint-eval-episodes", "$CanonicalCheckpointEvalEpisodes",
        "--final-eval-episodes", "$CanonicalFinalEvalEpisodes",
        "--artifact-root", $ArtifactRoot
    )
    $cmdArgs += $ExtraArgs
    & uv @cmdArgs
}

Write-Host "== Stage 1/5: Karpathy one-record overfit checks =="
if ($RunKarpathyCheck -eq 1) {
    foreach ($algo in $AlgoOrder) {
        Invoke-KarpathyCheck -Algo $algo
    }
}
else {
    Write-Host "Skipping Karpathy checks (RUN_KARPATHY_CHECK=$RunKarpathyCheck)"
}

Write-Host ""
Write-Host "== Stage 2/5: Algorithm smoke training =="
foreach ($algo in $AlgoOrder) {
    Invoke-SmokeTrain -Algo $algo
}

Write-Host ""
Write-Host "== Stage 3/5: Smoke evaluation (load + score sanity check) =="
& uv @(
    "run", "python", "-m", "src.models.evaluate_agents",
    "--agents", "ppo,a2c,dqn,greedy,random",
    "--ppo-model", "$ArtifactRoot/smoke/ppo/seed_$SmokeSeed/best_model.zip",
    "--a2c-model", "$ArtifactRoot/smoke/a2c/seed_$SmokeSeed/best_model.zip",
    "--dqn-model", "$ArtifactRoot/smoke/dqn/seed_$SmokeSeed/best_model.zip",
    "--seeds", "$SmokeSeed",
    "--episodes", "$SmokeEvalEpisodes",
    "--run-label", "smoke",
    "--output", "$ArtifactRoot/smoke/eval_smoke.json"
)

Write-Host ""
Write-Host "== Stage 4/5: Validation-only pilot sweeps (hyperparameter feasibility) =="
if ($RunPilotSweep -eq 1) {
    foreach ($algo in $AlgoOrder) {
        Invoke-PilotSweep -Algo $algo
    }
}
else {
    Write-Host "Skipping pilot sweeps (RUN_PILOT_SWEEP=$RunPilotSweep)"
}

Write-Host ""
Write-Host "== Stage 5/5: Full 5-seed benchmark training =="
foreach ($algo in $AlgoOrder) {
    $finalTimesteps = switch ($algo) {
        "ppo" { $CanonicalPpoTimesteps }
        "a2c" { $CanonicalA2cTimesteps }
        "dqn" { $CanonicalDqnTimesteps }
        default { throw "Unsupported algo '$algo'" }
    }

    $finalFlags = @()
    $hparamSource = "defaults"
    if ($UsePilotWinners -eq 1) {
        $finalFlags = Get-WinnerFlags -Algo $algo
        if ($finalFlags.Count -gt 0) {
            $hparamSource = "pilot_winner"
        }
        else {
            Write-Warning "Missing or empty pilot winner file for $algo under $ArtifactRoot/pilot_sweeps/$algo; using defaults."
        }
    }

    Write-Host "[FINAL] $algo hyperparameters source: $hparamSource"
    if ($finalFlags.Count -gt 0) {
        Write-Host "[FINAL] $algo hyperparameter flags: $($finalFlags -join ' ')"
    }

    foreach ($seed in $FinalSeeds) {
        Invoke-FinalTrain -Algo $algo -Seed $seed -Timesteps $finalTimesteps -ExtraArgs $finalFlags
    }
}

Write-Host ""
Write-Host "All runs finished. Artifacts are under '$ArtifactRoot'."
