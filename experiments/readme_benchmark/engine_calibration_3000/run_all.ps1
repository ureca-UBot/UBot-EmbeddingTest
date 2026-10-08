$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskRoot=Join-Path $PSScriptRoot 'outputs/run-v1'
New-Item -ItemType Directory -Force -Path (Join-Path $taskRoot 'logs') | Out-Null
& docker compose --profile validation run --rm validate scripts/run_matrix.py --stage validate
if ($LASTEXITCODE -ne 0) { throw 'Full 3000-row input verification failed' }
function Invoke-MatrixStage {
    param([string]$Service,[string[]]$StageArgs,[string]$LogName)
    & docker compose --profile matrix run --rm $Service @StageArgs 2>&1 |
        Tee-Object -FilePath (Join-Path $taskRoot "logs/$LogName.log")
    if ($LASTEXITCODE -ne 0) { throw "Matrix stage failed: $LogName" }
}
$taskEngines=@(
    @{Name='ollama'; Host='http://127.0.0.1:11435'; Client='http://host.docker.internal:11435'; Health='/api/tags'},
    @{Name='vllm'; Host='http://127.0.0.1:8001'; Client='http://host.docker.internal:8001'; Health='/health'},
    @{Name='tei'; Host='http://127.0.0.1:8081'; Client='http://host.docker.internal:8081'; Health='/health'}
)
foreach ($taskEngine in $taskEngines) {
    $taskDir=Join-Path $taskRoot "engines/$($taskEngine.Name)"
    New-Item -ItemType Directory -Force -Path $taskDir | Out-Null
    if ((Test-Path -LiteralPath (Join-Path $taskDir 'selection.json')) -and (Test-Path -LiteralPath (Join-Path $taskDir 'stop_verification.json'))) { continue }
    foreach ($taskProject in @('ubot-readme-benchmark','ubot-engine-calibration-3000')) {
        $taskActive=@(& docker ps --filter "label=com.docker.compose.project=$taskProject" --format '{{.Names}}')
        if ($taskActive.Count -gt 0) { throw "Previous benchmark is still active: $taskProject" }
    }
    $taskClock=[Diagnostics.Stopwatch]::StartNew();$taskMemoryJob=$null
    try {
        & docker compose --profile $taskEngine.Name up -d $taskEngine.Name
        if ($LASTEXITCODE -ne 0) { throw 'Engine startup failed' }
        $taskId=& docker compose --profile $taskEngine.Name ps -a -q $taskEngine.Name
        $taskImage=& docker inspect $taskId --format '{{.Image}}'
        @{engine=$taskEngine.Name;image_id=$taskImage;source='engine_calibration_3000/compose.yaml'} |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskDir 'engine_identity.json') -Encoding utf8
        $taskMemoryJob=Start-Job -ArgumentList $taskId,(Join-Path $taskDir 'memory_samples.jsonl') -ScriptBlock {
            param($taskContainer,$taskPath)
            while ((& docker inspect $taskContainer --format '{{.State.Running}}' 2>$null) -eq 'true') {
                $taskStats=& docker stats --no-stream --format '{{json .}}' $taskContainer 2>$null
                $taskGpu=& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null
                @{timestamp=[DateTimeOffset]::UtcNow.ToString('o');docker_stats=$taskStats;host_gpu_memory_utilization=$taskGpu} |
                    ConvertTo-Json -Compress | Add-Content -LiteralPath $taskPath -Encoding utf8
                Start-Sleep -Seconds 2
            }
        }
        $taskReady=$false
        while ($taskClock.Elapsed.TotalSeconds -lt 300) {
            try { Invoke-WebRequest -Uri ($taskEngine.Host+$taskEngine.Health) -TimeoutSec 3 -ErrorAction Stop | Out-Null;$taskReady=$true;break }
            catch { Start-Sleep -Seconds 2 }
        }
        if (-not $taskReady) { throw 'Health readiness timed out' }
        @{engine=$taskEngine.Name;healthy=$true;ready_seconds=$taskClock.Elapsed.TotalSeconds;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskDir 'readiness.json') -Encoding utf8
        Write-Output "Engine ready: $($taskEngine.Name); full 3000-question matrix begins"
        if (-not (Test-Path -LiteralPath (Join-Path $taskDir 'C-M20/summary.json'))) {
            Invoke-MatrixStage 'worker' @('scripts/run_matrix.py','--stage','embeddings','--engine',$taskEngine.Name,'--url',$taskEngine.Client) "$($taskEngine.Name)-ABC"
        }
        foreach ($taskReranker in @('bge','jina')) {
            foreach ($taskView in @('question','question_answer')) {
                $taskName="D-$taskReranker-$taskView-M20"
                if (-not (Test-Path -LiteralPath (Join-Path $taskDir "$taskName/summary.json"))) {
                    $taskWorker=if ($taskReranker -eq 'jina') { 'jina-worker' } else { 'worker' }
                    Invoke-MatrixStage $taskWorker @('scripts/run_matrix.py','--stage','D','--engine',$taskEngine.Name,'--reranker',$taskReranker,'--view',$taskView) "$($taskEngine.Name)-$taskName"
                }
            }
        }
        if (-not (Test-Path -LiteralPath (Join-Path $taskDir 'F-dense-pool-M20/summary.json'))) {
            Invoke-MatrixStage 'worker' @('scripts/run_matrix.py','--stage','native','--engine',$taskEngine.Name) "$($taskEngine.Name)-EF"
        }
        Invoke-MatrixStage 'worker' @('scripts/run_matrix.py','--stage','finish','--engine',$taskEngine.Name) "$($taskEngine.Name)-finish"
    } finally {
        & docker compose --profile $taskEngine.Name logs --no-color $taskEngine.Name 2>&1 |
            Set-Content -LiteralPath (Join-Path $taskDir 'engine.log') -Encoding utf8
        & docker compose --profile $taskEngine.Name stop $taskEngine.Name
        if ($LASTEXITCODE -ne 0) { throw 'Engine shutdown failed' }
        if ($taskMemoryJob) {
            Wait-Job -Job $taskMemoryJob -Timeout 10 | Out-Null
            if ($taskMemoryJob.State -eq 'Running') { Stop-Job $taskMemoryJob }
            Receive-Job $taskMemoryJob | Out-Null;Remove-Job $taskMemoryJob
        }
        $taskId=& docker compose --profile $taskEngine.Name ps -a -q $taskEngine.Name
        $taskState=(& docker inspect $taskId --format '{{json .State}}') | ConvertFrom-Json
        $taskHttp=$false
        try { Invoke-WebRequest -Uri ($taskEngine.Host+$taskEngine.Health) -TimeoutSec 2 -ErrorAction Stop | Out-Null;$taskHttp=$true } catch {}
        $taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-engine-calibration-3000 --format '{{.Names}}')
        $taskVerified=(-not $taskState.Running) -and (-not $taskHttp) -and $taskActive.Count -eq 0
        @{engine=$taskEngine.Name;verified_stopped=$taskVerified;container_state=$taskState;http_open=$taskHttp;active_containers=$taskActive;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
            ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskDir 'stop_verification.json') -Encoding utf8
        if (-not $taskVerified) { throw 'Shutdown verification failed; next engine must not start' }
        Write-Output "Engine stopped and verified: $($taskEngine.Name)"
    }
}
& docker compose --profile validation run --rm validate scripts/write_comparison.py
if ($LASTEXITCODE -ne 0) { throw 'Comparison report failed' }
