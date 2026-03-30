Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "'uv' is not installed or not on PATH. Install: https://docs.astral.sh/uv/getting-started/installation/"
}

$ArtifactRoot = if ($env:ARTIFACT_ROOT) { $env:ARTIFACT_ROOT } else { "outputs/benchmark" }
# Default smoke length is one canonical checkpoint interval.
$SmokeTimesteps = if ($env:SMOKE_TIMESTEPS) { [int]$env:SMOKE_TIMESTEPS } else { 20000 }
$SmokeSeed = if ($env:SMOKE_SEED) { [int]$env:SMOKE_SEED } else { 11 }
$SmokeEvalEpisodes = if ($env:SMOKE_EVAL_EPISODES) { [int]$env:SMOKE_EVAL_EPISODES } else { 5 }
$FinalSeedsCsv = if ($env:FINAL_SEEDS_CSV) { $env:FINAL_SEEDS_CSV } else { "11,22,33,44,55" }
$AlgoOrderCsv = if ($env:ALGO_ORDER_CSV) { $env:ALGO_ORDER_CSV } else { "ppo,a2c,dqn" }

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
Write-Host "artifact_root      : $ArtifactRoot"
Write-Host "algo_order         : $AlgoOrderCsv"
Write-Host "smoke_seed         : $SmokeSeed"
Write-Host "smoke_timesteps    : $SmokeTimesteps"
Write-Host "smoke_eval_episodes: $SmokeEvalEpisodes"
Write-Host "final_seeds        : $FinalSeedsCsv"
Write-Host ""
Write-Host "Note: full runs keep default trainer timesteps/envs/checkpoint cadence."
Write-Host ""

function Invoke-SmokeTrain {
    param(
        [Parameter(Mandatory = $true)][string]$Algo
    )

    Write-Host "[SMOKE] Training $Algo (seed=$SmokeSeed, timesteps=$SmokeTimesteps)"
    uv run python -m src.models.train_rl_agent `
        --algo $Algo `
        --run-label smoke `
        --seed $SmokeSeed `
        --timesteps $SmokeTimesteps `
        --artifact-root $ArtifactRoot
}

function Invoke-FinalTrain {
    param(
        [Parameter(Mandatory = $true)][string]$Algo,
        [Parameter(Mandatory = $true)][int]$Seed
    )

    Write-Host "[FINAL] Training $Algo (seed=$Seed, default timesteps/envs)"
    uv run python -m src.models.train_rl_agent `
        --algo $Algo `
        --run-label final `
        --seed $Seed `
        --artifact-root $ArtifactRoot
}

Write-Host "== Stage 1/3: Algorithm smoke training =="
foreach ($algo in $AlgoOrder) {
    Invoke-SmokeTrain -Algo $algo
}

Write-Host ""
# TODO: Check seed! 
Write-Host "== Stage 2/3: Smoke evaluation (load + score sanity check) =="
uv run python -m src.models.evaluate_agents `
    --agents ppo,a2c,dqn,greedy,random `
    --ppo-model "$ArtifactRoot/smoke/ppo/seed_$SmokeSeed/best_model.zip" `
    --a2c-model "$ArtifactRoot/smoke/a2c/seed_$SmokeSeed/best_model.zip" `
    --dqn-model "$ArtifactRoot/smoke/dqn/seed_$SmokeSeed/best_model.zip" `
    --seeds "$SmokeSeed" `
    --episodes $SmokeEvalEpisodes `
    --run-label smoke `
    --output "$ArtifactRoot/smoke/eval_smoke.json"

Write-Host ""
Write-Host "== Stage 3/3: Full 5-seed benchmark training =="
foreach ($algo in $AlgoOrder) {
    foreach ($seed in $FinalSeeds) {
        Invoke-FinalTrain -Algo $algo -Seed $seed
    }
}

Write-Host ""
Write-Host "All runs finished. Artifacts are under '$ArtifactRoot'."
