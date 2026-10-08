$ErrorActionPreference = 'Stop'
foreach ($taskStage in @('E', 'F')) {
    & docker compose --profile quality run --rm benchmark model_benchmarks/scripts/run_native_retrieval.py $taskStage --view question
    if ($LASTEXITCODE -ne 0) { throw "Native stage failed: $taskStage" }
}
& docker compose --profile quality run --rm benchmark model_benchmarks/scripts/select_and_calibrate.py
if ($LASTEXITCODE -ne 0) { throw 'Selection/calibration failed' }
