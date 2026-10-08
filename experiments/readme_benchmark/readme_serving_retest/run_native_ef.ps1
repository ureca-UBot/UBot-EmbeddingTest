$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskNativeOutput=Join-Path $PSScriptRoot 'outputs/run-v1/native_ef'
New-Item -ItemType Directory -Force -Path $taskNativeOutput | Out-Null
function Invoke-NativeStage {
    param([string]$Service,[string[]]$StageArgs,[string]$Name)
    $ErrorActionPreference='Continue'
    & docker compose -f compose.yaml run --rm --no-deps $Service @StageArgs 2>&1 |
        Tee-Object -FilePath (Join-Path $taskNativeOutput "$Name.log")
    if ($LASTEXITCODE -ne 0) { throw "Native E/F stage failed: $Name" }
    $taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-readme-serving-retest --format '{{.Names}}')
    if ($taskActive.Count -gt 0) { throw "Previous stage must be stopped before next stage: $taskActive" }
}
$taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-readme-serving-retest --format '{{.Names}}')
if ($taskActive.Count -gt 0) { throw "Check the active benchmark container before native E/F: $taskActive" }
Invoke-NativeStage 'validate' @('scripts/test_native_ef.py') 'contracts'
if (-not (Test-Path -LiteralPath (Join-Path $taskNativeOutput 'execution.json'))) {
    Invoke-NativeStage 'worker' @('scripts/native_ef.py','scores') 'scores'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskNativeOutput 'language_baseline_execution.json'))) {
    Invoke-NativeStage 'worker' @('scripts/native_ef.py','languages') 'languages'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskNativeOutput 'completion_audit.json'))) {
    Invoke-NativeStage 'validate' @('scripts/native_ef.py','evaluate') 'evaluate'
}
$taskResult=Get-Content -Raw -LiteralPath (Join-Path $taskNativeOutput 'completion_audit.json') | ConvertFrom-Json
if (-not $taskResult.completed) { throw 'Native E/F completion audit did not pass' }
$taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-readme-serving-retest --format '{{.Names}}')
if ($taskActive.Count -gt 0) { throw 'Native E/F shutdown verification failed' }
@{verified_stopped=$true;active_benchmark_containers=$taskActive;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskNativeOutput 'final_shutdown.json') -Encoding utf8
Write-Output "Native E/F completed: $($taskResult.structures) structures; $($taskResult.candidate_log_rows) verified log rows."
