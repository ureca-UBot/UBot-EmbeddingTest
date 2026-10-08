$ErrorActionPreference = 'Stop'
foreach ($taskStage in @('A', 'B', 'C')) {
    & docker compose --profile quality run --rm benchmark model_benchmarks/scripts/run_retrieval_benchmark.py $taskStage --split holdout
    if ($LASTEXITCODE -ne 0) { throw "Holdout prerequisite failed: $taskStage" }
}
& docker compose --profile quality run --rm benchmark model_benchmarks/scripts/run_retrieval_benchmark.py D --reranker bge --view question --split holdout
if ($LASTEXITCODE -ne 0) { throw 'Frozen challenger holdout failed' }
& docker compose --profile quality run --rm benchmark model_benchmarks/scripts/evaluate_frozen_holdout.py
if ($LASTEXITCODE -ne 0) { throw 'Frozen policy evaluation failed' }
& docker compose --profile quality run --rm benchmark model_benchmarks/scripts/run_selected_regression.py
if ($LASTEXITCODE -ne 0) { throw 'Frozen regression failed' }
& docker compose --profile quality run --rm benchmark model_benchmarks/scripts/analyze_retrieval_failures.py
if ($LASTEXITCODE -ne 0) { throw 'Calibration failure analysis failed' }
