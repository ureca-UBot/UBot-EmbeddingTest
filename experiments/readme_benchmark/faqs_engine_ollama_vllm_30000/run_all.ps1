$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskRoot=Join-Path $PSScriptRoot 'outputs/run-v1'
New-Item -ItemType Directory -Force -Path (Join-Path $taskRoot 'logs') | Out-Null
function Save-RunState {
    param([string]$State,[string]$Engine,[string]$Stage,[string]$Message='')
    @{status=$State;engine=$Engine;stage=$Stage;message=$Message;main_rows=30000;reference_rows=3000;engines=@('ollama','vllm');timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'run_status.json') -Encoding utf8
}
function Invoke-Stage {
    param([string]$Service,[string[]]$StageArgs,[string]$LogName)
    $ErrorActionPreference='Continue'
    Save-RunState 'running' $script:taskEngineName $LogName
    & docker compose --profile matrix run --rm $Service @StageArgs 2>&1 |
        Tee-Object -FilePath (Join-Path $taskRoot "logs/$LogName.log")
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $LogName" }
}
$script:taskEngineName='validation'
try {
    Save-RunState 'running' 'validation' 'contracts'
    & docker compose --profile validation run --rm validate scripts/test_contracts.py
    if ($LASTEXITCODE -ne 0) { throw 'New FAQ experiment contract tests failed' }
    & docker compose --profile validation run --rm validate scripts/run_experiment.py --stage validate
    if ($LASTEXITCODE -ne 0) { throw 'Frozen new input validation failed' }
    $taskEngines=@(
        @{Name='ollama';Host='http://127.0.0.1:11435';Client='http://host.docker.internal:11435';Health='/api/tags'},
        @{Name='vllm';Host='http://127.0.0.1:8001';Client='http://host.docker.internal:8001';Health='/health'}
    )
    foreach ($taskEngine in $taskEngines) {
        $script:taskEngineName=$taskEngine.Name
        $taskDir=Join-Path $taskRoot "engines/$($taskEngine.Name)"
        New-Item -ItemType Directory -Force -Path $taskDir | Out-Null
        if ((Test-Path -LiteralPath (Join-Path $taskDir 'selection.json')) -and (Test-Path -LiteralPath (Join-Path $taskDir 'stop_verification.json'))) { continue }
        foreach ($taskProject in @('ubot-readme-benchmark','ubot-engine-calibration-3000','ubot-engine-calibration-30000','ubot-faqs-engine-30000')) {
            $taskActive=@(& docker ps --filter "label=com.docker.compose.project=$taskProject" --format '{{.Names}}')
            if ($taskActive.Count -gt 0) { throw "A benchmark container is already active: $taskProject" }
        }
        if (-not (Test-Path -LiteralPath (Join-Path $taskDir 'C-M20/summary.json'))) {
            $taskClock=[Diagnostics.Stopwatch]::StartNew();$taskMemoryJob=$null
            try {
                Save-RunState 'running' $taskEngine.Name 'starting_dense_engine'
                & docker compose --profile $taskEngine.Name up -d $taskEngine.Name
                if ($LASTEXITCODE -ne 0) { throw 'Engine startup failed' }
                $taskId=& docker compose --profile $taskEngine.Name ps -a -q $taskEngine.Name
                $taskImage=& docker inspect $taskId --format '{{.Image}}'
                @{engine=$taskEngine.Name;image_id=$taskImage;dense_model='BAAI/bge-m3';input_limit=8192} |
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
                    try { Invoke-WebRequest -UseBasicParsing -Uri ($taskEngine.Host+$taskEngine.Health) -TimeoutSec 3 -ErrorAction Stop | Out-Null;$taskReady=$true;break }
                    catch { Start-Sleep -Seconds 2 }
                }
                if (-not $taskReady) { throw 'Engine health readiness timed out' }
                @{engine=$taskEngine.Name;healthy=$true;ready_seconds=$taskClock.Elapsed.TotalSeconds;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
                    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskDir 'readiness.json') -Encoding utf8
                Write-Output "Engine ready: $($taskEngine.Name); new FAQ 30000 main + 3000 reference HTTP queries"
                Invoke-Stage 'worker' @('scripts/run_experiment.py','--stage','embeddings','--engine',$taskEngine.Name,'--url',$taskEngine.Client) "$($taskEngine.Name)-ABC"
                $taskRuntimePath=if ($taskEngine.Name -eq 'ollama') { '/api/ps' } else { '/v1/models' }
                Invoke-RestMethod -Uri ($taskEngine.Host+$taskRuntimePath) -TimeoutSec 5 |
                    ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $taskDir 'serving_model_runtime.json') -Encoding utf8
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
                try { Invoke-WebRequest -UseBasicParsing -Uri ($taskEngine.Host+$taskEngine.Health) -TimeoutSec 2 -ErrorAction Stop | Out-Null;$taskHttp=$true } catch {}
                $taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-faqs-engine-30000 --format '{{.Names}}')
                $taskVerified=(-not $taskState.Running) -and (-not $taskHttp) -and $taskActive.Count -eq 0
                @{engine=$taskEngine.Name;verified_stopped=$taskVerified;container_state=$taskState;http_open=$taskHttp;active_containers=$taskActive;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
                    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskDir 'stop_verification.json') -Encoding utf8
                if (-not $taskVerified) { throw 'Engine shutdown verification failed' }
                Write-Output "Engine stopped and verified: $($taskEngine.Name)"
            }
        }
        foreach ($taskReranker in @('bge','jina')) {
            foreach ($taskView in @('question','question_answer')) {
                $taskName="D-$taskReranker-$taskView-M20"
                if (-not (Test-Path -LiteralPath (Join-Path $taskDir "$taskName/summary.json"))) {
                    $taskWorker=if ($taskReranker -eq 'jina') { 'jina-worker' } else { 'worker' }
                    Invoke-Stage $taskWorker @('scripts/run_experiment.py','--stage','D','--engine',$taskEngine.Name,'--reranker',$taskReranker,'--view',$taskView) "$($taskEngine.Name)-$taskName"
                }
            }
        }
        if (-not (Test-Path -LiteralPath (Join-Path $taskDir 'F-dense-pool-M20/summary.json'))) {
            Invoke-Stage 'worker' @('scripts/run_experiment.py','--stage','native','--engine',$taskEngine.Name) "$($taskEngine.Name)-EF"
        }
        Invoke-Stage 'worker' @('scripts/run_experiment.py','--stage','finish','--engine',$taskEngine.Name) "$($taskEngine.Name)-finish"
    }
    Invoke-Stage 'validate' @('scripts/compare_engines.py') 'comparison'
    Save-RunState 'complete' 'ollama+vllm' 'all_stages_complete'
} catch {
    Save-RunState 'failed' $script:taskEngineName 'pipeline_error' $_.Exception.Message
    throw
}
