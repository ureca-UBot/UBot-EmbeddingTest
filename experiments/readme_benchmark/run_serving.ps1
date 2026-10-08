$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskRunRoot = Join-Path $PSScriptRoot 'outputs/run-v1'
if (-not (Test-Path -LiteralPath (Join-Path $taskRunRoot 'holdout/policy_results.json'))) { throw 'Complete frozen holdout first' }
New-Item -ItemType Directory -Path (Join-Path $taskRunRoot 'serving') -Force | Out-Null
$taskEngines = @(
    @{ Name='ollama'; HostUrl='http://127.0.0.1:11435'; ClientUrl='http://host.docker.internal:11435'; Health='/api/tags' },
    @{ Name='tei'; HostUrl='http://127.0.0.1:8081'; ClientUrl='http://host.docker.internal:8081'; Health='/health' },
    @{ Name='vllm'; HostUrl='http://127.0.0.1:8001'; ClientUrl='http://host.docker.internal:8001'; Health='/health' }
)
foreach ($taskEngine in $taskEngines) {
    $taskDir = Join-Path $taskRunRoot "serving/$($taskEngine.Name)"
    New-Item -ItemType Directory -Path $taskDir -Force | Out-Null
    $taskActive = & docker ps --filter label=com.docker.compose.project=ubot-readme-benchmark --format '{{.Names}}'
    if ($taskActive) { throw "A previous test container is still active: $taskActive" }
    $taskClock = [Diagnostics.Stopwatch]::StartNew()
    $taskReady=$false
    $taskMemoryJob=$null
    try {
        & docker compose --profile $taskEngine.Name up -d $taskEngine.Name
        if ($LASTEXITCODE -ne 0) { throw 'Failed to start serving engine' }
        $taskContainerId = & docker compose --profile $taskEngine.Name ps -a -q $taskEngine.Name
        $taskMemoryJob = Start-Job -ArgumentList $taskContainerId,(Join-Path $taskDir 'memory_samples.jsonl') -ScriptBlock {
            param($taskId,$taskMemoryPath)
            while ($true) {
                $taskRunning = & docker inspect $taskId --format '{{.State.Running}}' 2>$null
                if ($taskRunning -ne 'true') { break }
                $taskStats = & docker stats --no-stream --format '{{json .}}' $taskId 2>$null
                $taskGpu = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null
                @{timestamp=[DateTimeOffset]::UtcNow.ToString('o'); docker_stats=$taskStats; gpu_memory_and_utilization=$taskGpu} |
                    ConvertTo-Json -Compress | Add-Content -LiteralPath $taskMemoryPath -Encoding utf8
                Start-Sleep -Seconds 2
            }
        }
        while ($taskClock.Elapsed.TotalSeconds -lt 300) {
            try {
                Invoke-WebRequest -Uri ($taskEngine.HostUrl+$taskEngine.Health) -TimeoutSec 3 -ErrorAction Stop | Out-Null
                $taskReady=$true;break
            } catch { Start-Sleep -Seconds 2 }
        }
        if (-not $taskReady) { throw 'Health check timed out' }
        if ($taskEngine.Name -eq 'ollama') {
            $taskBody = '{"model":"bge-m3","input":["휴대폰 요금 납부일은 언제인가요?"],"truncate":false,"keep_alive":"30m"}'
            $taskResponse = Invoke-RestMethod -Uri ($taskEngine.HostUrl+'/api/embed') -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($taskBody)) -TimeoutSec 180
            if ($taskResponse.embeddings[0].Count -ne 1024) { throw 'Ollama embedding dimension probe failed' }
        }
        @{ engine=$taskEngine.Name; healthy=$true; cold_start_to_ready_seconds=$taskClock.Elapsed.TotalSeconds; timestamp=[DateTimeOffset]::UtcNow.ToString('o') } |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskDir 'readiness.json') -Encoding utf8
        Write-Output "Engine ready: $($taskEngine.Name)"
        & docker compose --profile serving run --rm client scripts/benchmark_serving_client.py --engine $taskEngine.Name --url $taskEngine.ClientUrl --seconds 10 --minimum-attempts 100 2>&1 |
            Tee-Object -FilePath (Join-Path $taskDir 'client.log')
        if ($LASTEXITCODE -ne 0) { throw "Serving measurement failed: $($taskEngine.Name)" }
    } finally {
        & docker compose --profile $taskEngine.Name logs --no-color $taskEngine.Name 2>&1 | Set-Content -LiteralPath (Join-Path $taskDir 'engine.log') -Encoding utf8
        & docker compose --profile $taskEngine.Name stop $taskEngine.Name
        if ($LASTEXITCODE -ne 0) { throw "Failed to stop engine: $($taskEngine.Name)" }
        if ($taskMemoryJob) {
            Wait-Job -Job $taskMemoryJob -Timeout 10 | Out-Null
            if ($taskMemoryJob.State -eq 'Running') { Stop-Job -Job $taskMemoryJob }
            Receive-Job -Job $taskMemoryJob | Out-Null
            Remove-Job -Job $taskMemoryJob
        }
        $taskId = & docker compose --profile $taskEngine.Name ps -a -q $taskEngine.Name
        $taskStateText = & docker inspect $taskId --format '{{json .State}}'
        $taskState = $taskStateText | ConvertFrom-Json
        $taskHttpOpen=$false
        try { Invoke-WebRequest -Uri ($taskEngine.HostUrl+$taskEngine.Health) -TimeoutSec 2 -ErrorAction Stop | Out-Null; $taskHttpOpen=$true } catch {}
        $taskActive = @(& docker ps --filter label=com.docker.compose.project=ubot-readme-benchmark --format '{{.Names}}')
        $taskVerified=(-not $taskState.Running) -and (-not $taskHttpOpen) -and $taskActive.Count -eq 0
        @{engine=$taskEngine.Name; verified_stopped=$taskVerified; container_state=$taskState; http_open=$taskHttpOpen; active_project_containers=$taskActive; timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
            ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskDir 'stop_verification.json') -Encoding utf8
        if (-not $taskVerified) { throw 'Engine stop verification failed; next engine must not start' }
        Write-Output "Engine stopped and verified: $($taskEngine.Name)"
    }
}
& docker compose --profile validation run --rm validate scripts/shortlist_engines.py
if ($LASTEXITCODE -ne 0) { throw 'Engine shortlist failed' }
