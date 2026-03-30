Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "'uv' is not installed or not on PATH. Install: https://docs.astral.sh/uv/getting-started/installation/"
}

$ArtifactRoot = if ($env:ARTIFACT_ROOT) { $env:ARTIFACT_ROOT } else { "outputs/benchmark" }
$RunLabel = if ($env:RUN_LABEL) { $env:RUN_LABEL } else { "final" }
$EvalSeedsCsv = if ($env:EVAL_SEEDS_CSV) { $env:EVAL_SEEDS_CSV } else { "11,22,33,44,55" }
$EvalEpisodes = if ($env:EVAL_EPISODES) { [int]$env:EVAL_EPISODES } else { 100 }
$Agents = if ($env:AGENTS) { $env:AGENTS } else { "ppo,a2c,dqn,greedy,random" }

$TrainDataset = if ($env:TRAIN_DATASET) { $env:TRAIN_DATASET } else { "data/static/scenario_parameter_records_seeded_train.json" }
$ValDataset = if ($env:VAL_DATASET) { $env:VAL_DATASET } else { "data/static/scenario_parameter_records_seeded_val.json" }
$HoldoutDataset = if ($env:HOLDOUT_DATASET) { $env:HOLDOUT_DATASET } else { "data/static/scenario_parameter_records_seeded_holdout.json" }

$DefaultOutputDir = Join-Path -Path $ArtifactRoot -ChildPath "$RunLabel/eval"
$OutputDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { $DefaultOutputDir }
$IncludeFamilyHoldout = if ($env:INCLUDE_FAMILY_HOLDOUT) { [int]$env:INCLUDE_FAMILY_HOLDOUT } else { 0 }
$IncludeTemporalHoldout = if ($env:INCLUDE_TEMPORAL_HOLDOUT) { [int]$env:INCLUDE_TEMPORAL_HOLDOUT } else { 0 }
$NoNormalizedBurn = if ($env:NO_NORMALIZED_BURN) { [int]$env:NO_NORMALIZED_BURN } else { 0 }

foreach ($dataset in @($TrainDataset, $ValDataset, $HoldoutDataset)) {
    if (-not (Test-Path $dataset)) {
        Write-Error "Missing dataset '$dataset'."
    }
}

$EvalSeeds = $EvalSeedsCsv -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "== Benchmark evaluation configuration =="
Write-Host "artifact_root          : $ArtifactRoot"
Write-Host "run_label              : $RunLabel"
Write-Host "eval_seeds             : $EvalSeedsCsv"
Write-Host "eval_episodes_per_seed : $EvalEpisodes"
Write-Host "agents                 : $Agents"
Write-Host "output_dir             : $OutputDir"
Write-Host ""

foreach ($seed in $EvalSeeds) {
    $PpoModel = Join-Path -Path $ArtifactRoot -ChildPath "$RunLabel/ppo/seed_$seed/best_model.zip"
    $A2cModel = Join-Path -Path $ArtifactRoot -ChildPath "$RunLabel/a2c/seed_$seed/best_model.zip"
    $DqnModel = Join-Path -Path $ArtifactRoot -ChildPath "$RunLabel/dqn/seed_$seed/best_model.zip"

    foreach ($modelPath in @($PpoModel, $A2cModel, $DqnModel)) {
        if (-not (Test-Path $modelPath)) {
            Write-Error "Missing model '$modelPath'. Run training first: ./scripts/run_benchmark_train.ps1"
        }
    }

    $OutputJson = Join-Path -Path $OutputDir -ChildPath "seed_$seed.json"
    Write-Host "[EVAL] seed=$seed -> $OutputJson"

    $CmdArgs = @(
        "run", "python", "-m", "src.models.evaluate_agents",
        "--agents", $Agents,
        "--ppo-model", $PpoModel,
        "--a2c-model", $A2cModel,
        "--dqn-model", $DqnModel,
        "--train-dataset", $TrainDataset,
        "--val-dataset", $ValDataset,
        "--holdout-dataset", $HoldoutDataset,
        "--seeds", $seed,
        "--episodes", "$EvalEpisodes",
        "--run-label", $RunLabel,
        "--output", $OutputJson
    )

    if ($IncludeFamilyHoldout -eq 1) {
        $CmdArgs += "--include-family-holdout"
    }
    if ($IncludeTemporalHoldout -eq 1) {
        $CmdArgs += "--include-temporal-holdout"
    }
    if ($NoNormalizedBurn -eq 1) {
        $CmdArgs += "--no-normalized-burn"
    }

    & uv @CmdArgs
}

Write-Host ""
Write-Host "Evaluation complete. Results are in '$OutputDir'."
