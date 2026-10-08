$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskOutput=Join-Path $PSScriptRoot 'outputs/run-v1'
New-Item -ItemType Directory -Force -Path (Join-Path $taskOutput 'logs') | Out-Null
function Save-TaskState {
    param([string]$State,[string]$Engine,[string]$Stage,[string]$Message='')
    @{status=$State;engine=$Engine;stage=$Stage;message=$Message;main_rows=5000;engines=@('ollama','vllm');timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'run_status.json') -Encoding utf8
}
function Invoke-TaskStage {
    param([string]$Service,[string[]]$StageArgs,[string]$LogName)
    $ErrorActionPreference='Continue'
    Save-TaskState 'running' $script:taskEngine $LogName
    & docker compose --profile matrix run --rm $Service @StageArgs 2>&1 |
        Tee-Object -FilePath (Join-Path $taskOutput "logs/$LogName.log")
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $LogName" }
}
$script:taskEngine='validation'
try {
    Invoke-TaskStage 'validate' @('scripts/test_contracts.py') 'contracts'
    Invoke-TaskStage 'validate' @('scripts/run.py','--stage','validate') 'input_validation'
    $taskEngines=@(
        @{Name='ollama';Host='http://127.0.0.1:11435';Client='http://host.docker.internal:11435';Health='/api/tags'},
        @{Name='vllm';Host='http://127.0.0.1:8001';Client='http://host.docker.internal:8001';Health='/health'}
    )
    foreach ($taskEntry in $taskEngines) {
        $script:taskEngine=$taskEntry.Name
        $taskDir=Join-Path $taskOutput "engines/$($taskEntry.Name)"
        New-Item -ItemType Directory -Force -Path $taskDir | Out-Null
        if (-not (Test-Path -LiteralPath (Join-Path $taskDir 'serving_load.json'))) {
            $taskPrior=@(& docker ps --format '{{.Names}}')
            if ($taskPrior.Count -gt 0) { throw "An active container must be checked before startup: $taskPrior" }
            $taskTimer=[Diagnostics.Stopwatch]::StartNew();$taskJob=$null
            try {
                Save-TaskState 'running' $taskEntry.Name 'startup'
                & docker compose --profile $taskEntry.Name up -d $taskEntry.Name
                if ($LASTEXITCODE -ne 0) { throw 'Startup failed' }
                $taskContainer=& docker compose --profile $taskEntry.Name ps -a -q $taskEntry.Name
                $taskImage=& docker inspect $taskContainer --format '{{.Image}}'
                @{engine=$taskEntry.Name;container_id=$taskContainer;image_id=$taskImage;dense_model='BAAI/bge-m3';input_limit=8192} |
                    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskDir 'engine_identity.json') -Encoding utf8
                $taskJob=Start-Job -ArgumentList $taskContainer,(Join-Path $taskDir 'memory_samples.jsonl') -ScriptBlock {
                    param($taskContainerId,$taskSamplePath)
                    while ((& docker inspect $taskContainerId --format '{{.State.Running}}' 2>$null) -eq 'true') {
                        $taskStats=& docker stats --no-stream --format '{{json .}}' $taskContainerId 2>$null
                        $taskGpu=& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null
                        @{timestamp=[DateTimeOffset]::UtcNow.ToString('o');docker_stats=$taskStats;host_gpu_memory_utilization=$taskGpu} |
                            ConvertTo-Json -Compress | Add-Content -LiteralPath $taskSamplePath -Encoding utf8
                        Start-Sleep -Seconds 2
                    }
                }
                $taskHealthy=$false
                while ($taskTimer.Elapsed.TotalSeconds -lt 300) {
                    try { Invoke-WebRequest -UseBasicParsing -Uri ($taskEntry.Host+$taskEntry.Health) -TimeoutSec 3 -ErrorAction Stop | Out-Null;$taskHealthy=$true;break }
                    catch { Start-Sleep -Seconds 2 }
                }
                if (-not $taskHealthy) { throw 'Health readiness timed out' }
                @{healthy=$true;ready_seconds=$taskTimer.Elapsed.TotalSeconds;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
                    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskDir 'readiness.json') -Encoding utf8
                if (-not (Test-Path -LiteralPath (Join-Path $taskDir 'C-union-K20/summary.json'))) {
                    Invoke-TaskStage 'worker' @('scripts/run.py','--stage','embeddings','--engine',$taskEntry.Name,'--url',$taskEntry.Client) "$($taskEntry.Name)-ABC"
                }
                Invoke-TaskStage 'worker' @('scripts/run.py','--stage','serving','--engine',$taskEntry.Name,'--url',$taskEntry.Client) "$($taskEntry.Name)-serving"
                $taskModelPath=if ($taskEntry.Name -eq 'ollama') { '/api/ps' } else { '/v1/models' }
                Invoke-RestMethod -Uri ($taskEntry.Host+$taskModelPath) -TimeoutSec 5 |
                    ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $taskDir 'serving_model_runtime.json') -Encoding utf8
            } finally {
                $ErrorActionPreference='Continue'
                & docker compose --profile $taskEntry.Name logs --no-color $taskEntry.Name 2>&1 |
                    Set-Content -LiteralPath (Join-Path $taskDir 'engine.log') -Encoding utf8
                & docker compose --profile $taskEntry.Name stop $taskEntry.Name
                if ($LASTEXITCODE -ne 0) { throw 'Engine shutdown failed' }
                if ($taskJob) {
                    Wait-Job $taskJob -Timeout 10 | Out-Null
                    if ($taskJob.State -eq 'Running') { Stop-Job $taskJob }
                    Receive-Job $taskJob | Out-Null;Remove-Job $taskJob
                }
                $taskContainer=& docker compose --profile $taskEntry.Name ps -a -q $taskEntry.Name
                $taskState=(& docker inspect $taskContainer --format '{{json .State}}') | ConvertFrom-Json
                $taskHttp=$false
                try { Invoke-WebRequest -UseBasicParsing -Uri ($taskEntry.Host+$taskEntry.Health) -TimeoutSec 2 -ErrorAction Stop | Out-Null;$taskHttp=$true } catch {}
                $taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-readme-serving-retest --format '{{.Names}}')
                $taskStopped=(-not $taskState.Running) -and (-not $taskHttp) -and $taskActive.Count -eq 0
                @{verified_stopped=$taskStopped;container_state=$taskState;http_open=$taskHttp;active_containers=$taskActive;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
                    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskDir 'stop_verification.json') -Encoding utf8
                if (-not $taskStopped) { throw 'Shutdown verification failed' }
                Write-Output "Engine stopped and verified: $($taskEntry.Name)"
            }
        }
        foreach ($taskReranker in @('bge','jina')) {
            foreach ($taskView in @('question','question_answer')) {
                $taskName="D-$taskReranker-$taskView"
                if (-not (Test-Path -LiteralPath (Join-Path $taskDir "$taskName/candidate_log_schema.json"))) {
                    $taskWorker=if ($taskReranker -eq 'jina') { 'jina-worker' } else { 'worker' }
                    Invoke-TaskStage $taskWorker @('scripts/run.py','--stage','D','--engine',$taskEntry.Name,'--reranker',$taskReranker,'--view',$taskView) "$($taskEntry.Name)-$taskName"
                }
            }
        }
    }
    $script:taskEngine='ollama+vllm'
    Invoke-TaskStage 'validate' @('scripts/compare.py') 'comparison'
    Save-TaskState 'complete' 'ollama+vllm' 'all_stages_complete'
} catch {
    Save-TaskState 'failed' $script:taskEngine 'pipeline_error' $_.Exception.Message
    throw
}
