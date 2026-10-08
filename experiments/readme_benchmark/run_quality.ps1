$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskRunRoot = Join-Path $PSScriptRoot 'outputs/run-v1'
function Invoke-FreshStage {
    param([string]$Service, [string[]]$StageArgs, [string]$Log)
    & docker compose --profile quality run --rm $Service @StageArgs 2>&1 | Tee-Object -FilePath (Join-Path $taskRunRoot $Log)
    if ($LASTEXITCODE -ne 0) { throw "Fresh stage failed: $($StageArgs -join ' ')" }
    $taskActive = & docker ps --filter label=com.docker.compose.project=ubot-readme-benchmark --format '{{.Names}}'
    if ($taskActive) { throw "Container remained active between quality stages: $taskActive" }
}
New-Item -ItemType Directory -Path (Join-Path $taskRunRoot 'logs') -Force | Out-Null
Invoke-FreshStage 'validate' @('-m','unittest','discover','-s','tests','-v') 'logs/contracts.log'
foreach ($taskStage in @('A','B','C')) {
    $taskName = if ($taskStage -eq 'C') { 'C-M20' } else { $taskStage }
    if (-not (Test-Path -LiteralPath (Join-Path $taskRunRoot "calibration/$taskName/summary.json"))) {
        Invoke-FreshStage 'benchmark' @('scripts/run_retrieval_benchmark.py',$taskStage) "logs/$taskStage.log"
    }
}
foreach ($taskReranker in @('bge','jina')) {
    foreach ($taskView in @('question','question_answer')) {
        $taskName = "D-$taskReranker-$taskView-M20"
        if (-not (Test-Path -LiteralPath (Join-Path $taskRunRoot "calibration/$taskName/summary.json"))) {
            $taskService = if ($taskReranker -eq 'jina') { 'jina-benchmark' } else { 'benchmark' }
            Invoke-FreshStage $taskService @('scripts/run_retrieval_benchmark.py','D','--reranker',$taskReranker,'--view',$taskView) "logs/$taskName.log"
        }
    }
}
Invoke-FreshStage 'benchmark' @('scripts/decide_native.py') 'logs/native-decision.log'
$taskNative = Get-Content -LiteralPath (Join-Path $taskRunRoot 'native_decision.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($taskNative.execute_sparse_and_multivector) {
    foreach ($taskStage in @('E','F')) {
        $taskName = if ($taskStage -eq 'E') { 'E-dense-sparse-RRF-M20' } else { 'F-full' }
        if (-not (Test-Path -LiteralPath (Join-Path $taskRunRoot "calibration/$taskName/summary.json"))) {
            Invoke-FreshStage 'benchmark' @('scripts/run_native_retrieval.py',$taskStage) "logs/$taskStage.log"
        }
    }
}
Invoke-FreshStage 'benchmark' @('scripts/freeze_selection.py') 'logs/selection.log'
$taskSelection = Get-Content -LiteralPath (Join-Path $taskRunRoot 'selection.json') -Raw -Encoding utf8 | ConvertFrom-Json
$taskService = if ($taskSelection.selected.StartsWith('D-jina-')) { 'jina-benchmark' } else { 'benchmark' }
Invoke-FreshStage $taskService @('scripts/evaluate_selected.py','--mode','holdout') 'logs/holdout.log'
Invoke-FreshStage $taskService @('scripts/evaluate_selected.py','--mode','diagnostics') 'logs/diagnostics.log'
