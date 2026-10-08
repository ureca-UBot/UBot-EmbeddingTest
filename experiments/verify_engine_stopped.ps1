param([Parameter(Mandatory=$true)][string]$Engine,[Parameter(Mandatory=$true)][string]$Url)
$ErrorActionPreference = 'Stop'
$taskDir = Join-Path $PSScriptRoot "model_benchmarks/outputs/retrieval/run-20261006-v1/serving/$Engine"
$taskRunning = & docker inspect "ubot-retrieval-benchmark-$Engine-1" --format '{{.State.Running}}'
$taskStatus = & docker inspect "ubot-retrieval-benchmark-$Engine-1" --format '{{.State.Status}}'
$taskResponding = $false
try { Invoke-WebRequest -Uri $Url -TimeoutSec 3 | Out-Null; $taskResponding = $true } catch { }
$taskProjectRunning = @(& docker ps --filter 'label=com.docker.compose.project=ubot-retrieval-benchmark' --format '{{.Names}}')
$taskGPU = & docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04@sha256:80d4d9ac041242f6ae5d05f9be262b3374e0e0b8bb5a49c6c3e94e192cde4a44 nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
@{ engine=$Engine; checked_at=[DateTimeOffset]::UtcNow.ToString('o'); container_running=($taskRunning -eq 'true');
   container_status=$taskStatus; endpoint_still_responding=$taskResponding; project_running_containers=$taskProjectRunning;
   gpu_memory_MiB_and_utilization_percent=$taskGPU; gpu_memory_scope='whole device including Windows display processes' } |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskDir 'shutdown-verification.json') -Encoding utf8
if ($taskRunning -eq 'true' -or $taskResponding) { throw "Engine has not stopped: $Engine" }
Write-Output "$Engine stopped; endpoint closed; GPU snapshot: $taskGPU"
