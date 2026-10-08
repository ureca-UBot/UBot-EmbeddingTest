param([Parameter(Mandatory=$true)][string]$Engine)
$ErrorActionPreference = 'Stop'
$taskOutput = Join-Path $PSScriptRoot "model_benchmarks/outputs/retrieval/run-20261006-v1/serving/$Engine/docker-memory-samples.jsonl"
$taskContainer = "ubot-retrieval-benchmark-$Engine-1"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $taskOutput) | Out-Null
while ($true) {
    $taskState = & docker inspect $taskContainer --format '{{.State.Running}}' 2>$null
    if ($LASTEXITCODE -ne 0 -or $taskState -ne 'true') { break }
    $taskStats = & docker stats --no-stream --format '{{json .}}' $taskContainer
    if ($LASTEXITCODE -eq 0) {
        $taskRecord = @{ timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds(); stats = ($taskStats | ConvertFrom-Json) }
        $taskRecord | ConvertTo-Json -Compress -Depth 4 | Add-Content -LiteralPath $taskOutput -Encoding utf8
    }
    Start-Sleep -Seconds 1
}
