$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskActive = @(& docker ps --filter label=com.docker.compose.project=ubot-readme-benchmark --format '{{.Names}}')
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect active benchmark containers' }
$taskProbes = @(
    @{ Name='ollama'; Url='http://127.0.0.1:11435/api/tags' },
    @{ Name='tei'; Url='http://127.0.0.1:8081/health' },
    @{ Name='vllm'; Url='http://127.0.0.1:8001/health' }
)
$taskStates = @()
foreach ($taskProbe in $taskProbes) {
    $taskId = & docker compose --profile $taskProbe.Name ps -a -q $taskProbe.Name
    if ($LASTEXITCODE -ne 0 -or -not $taskId) { throw "Cannot inspect engine: $($taskProbe.Name)" }
    $taskState = (& docker inspect $taskId --format '{{json .State}}') | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Container inspection failed' }
    $taskHttpOpen = $false
    try { Invoke-WebRequest -Uri $taskProbe.Url -TimeoutSec 2 -ErrorAction Stop | Out-Null; $taskHttpOpen=$true } catch {}
    $taskStates += @{ engine=$taskProbe.Name; status=$taskState.Status; exit_code=$taskState.ExitCode; running=$taskState.Running; http_open=$taskHttpOpen }
}
$taskVerified = $taskActive.Count -eq 0 -and @($taskStates | Where-Object { $_.running -or $_.http_open }).Count -eq 0
$taskResult = @{
    timestamp=[DateTimeOffset]::UtcNow.ToString('o')
    verified_stopped=$taskVerified
    active_project_containers=$taskActive
    engines=$taskStates
    host_gpu_after_shutdown=(& nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits)
    gpu_memory_scope='Host total includes desktop; not expected to reach zero'
}
$taskResult | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath 'outputs/run-v1/final_shutdown.json' -Encoding utf8
if (-not $taskVerified) { throw 'Final shutdown verification failed' }
Write-Output 'Final verification: benchmark containers 0, engine processes stopped, HTTP ports closed'
