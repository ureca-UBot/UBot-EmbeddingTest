param([Parameter(Mandatory=$true)][string]$Engine,[Parameter(Mandatory=$true)][string]$Url)
$ErrorActionPreference = 'Stop'
$taskStart = [DateTimeOffset]::UtcNow
$taskWatch = [Diagnostics.Stopwatch]::StartNew()
$taskError = ''
$taskDir = Join-Path $PSScriptRoot "model_benchmarks/outputs/retrieval/run-20261006-v1/serving/$Engine"
New-Item -ItemType Directory -Force -Path $taskDir | Out-Null
while ($taskWatch.Elapsed.TotalSeconds -lt 240) {
    try {
        $taskResponse = Invoke-WebRequest -Uri $Url -TimeoutSec 3
        if ($taskResponse.StatusCode -eq 200) {
            @{ status='READY'; elapsed_from_probe_start_seconds=$taskWatch.Elapsed.TotalSeconds; started_at=$taskStart.ToString('o'); endpoint=$Url } |
                ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskDir 'readiness.json') -Encoding utf8
            Write-Output "$Engine ready in $($taskWatch.Elapsed.TotalSeconds) seconds"
            exit 0
        }
    } catch { $taskError = $_.Exception.Message }
    $taskState = & docker inspect "ubot-retrieval-benchmark-$Engine-1" --format '{{.State.Running}}' 2>$null
    if ($taskState -ne 'true') { break }
    Start-Sleep -Seconds 2
}
@{ status='FAILED_READY'; elapsed_from_probe_start_seconds=$taskWatch.Elapsed.TotalSeconds; error=$taskError } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskDir 'readiness.json') -Encoding utf8
exit 1
