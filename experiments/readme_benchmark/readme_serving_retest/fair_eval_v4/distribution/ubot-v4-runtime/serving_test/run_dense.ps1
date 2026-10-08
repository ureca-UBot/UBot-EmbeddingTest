param([string[]]$Datasets=@('existing','kt','skt','lgu'))
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskV4Output=Join-Path $PSScriptRoot 'outputs/run-v4'
New-Item -ItemType Directory -Force -Path (Join-Path $taskV4Output 'logs') | Out-Null
function Save-V4State {
 param([string]$State,[string]$Dataset,[string]$Engine,[string]$Stage,[string]$Message='')
 @{status=$State;dataset=$Dataset;engine=$Engine;stage=$Stage;message=$Message;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskV4Output 'dense_status.json') -Encoding utf8
}
function Invoke-V4Stage {
 param([string]$Dataset,[string]$Engine,[string]$Url,[string]$Stage)
 Save-V4State 'running' $Dataset $Engine $Stage
 $ErrorActionPreference='Continue'
 & docker compose run --rm --no-deps worker run_v4.py $Stage --dataset $Dataset --engine $Engine --url $Url 2>&1 |
  Tee-Object -FilePath (Join-Path $taskV4Output "logs/$Dataset-$Engine-$Stage.log")
 if ($LASTEXITCODE -ne 0) { throw "Failed $Dataset $Engine $Stage" }
}
$taskV4Engines=@(
 @{Name='ollama';Host='http://127.0.0.1:11436';Client='http://host.docker.internal:11436';Health='/api/tags'},
 @{Name='vllm';Host='http://127.0.0.1:8002';Client='http://host.docker.internal:8002';Health='/health'}
)
try {
 foreach ($taskV4Dataset in $Datasets) {
  if ($taskV4Dataset -notin @('existing','kt','skt','lgu')) { throw 'Invalid dataset scope' }
  foreach ($taskV4Entry in $taskV4Engines) {
   $taskV4Dir=Join-Path $taskV4Output "$taskV4Dataset/$($taskV4Entry.Name)"
   New-Item -ItemType Directory -Force -Path $taskV4Dir | Out-Null
   if (Test-Path -LiteralPath (Join-Path $taskV4Dir 'dense_complete.json')) { continue }
   $taskV4Active=@(& docker ps --filter label=com.docker.compose.project=ubot-v4-runtime --format '{{.Names}}')
   if ($taskV4Active.Count -gt 0) { throw "Previous v4 container still active: $taskV4Active" }
   $taskV4Timer=[Diagnostics.Stopwatch]::StartNew();$taskV4Job=$null;$taskV4Ok=$false
   try {
    Save-V4State 'running' $taskV4Dataset $taskV4Entry.Name 'startup'
    & docker compose up -d $taskV4Entry.Name
    if ($LASTEXITCODE -ne 0) { throw 'Engine startup failed' }
    $taskV4Container=& docker compose ps -a -q $taskV4Entry.Name
    $taskV4Image=& docker inspect $taskV4Container --format '{{.Image}}'
    @{dataset_id=$taskV4Dataset;engine=$taskV4Entry.Name;container_id=$taskV4Container;image_id=$taskV4Image;started=[DateTimeOffset]::UtcNow.ToString('o')} |
     ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskV4Dir 'engine_identity.json') -Encoding utf8
    $taskV4Job=Start-Job -ArgumentList $taskV4Container,(Join-Path $taskV4Dir 'memory_samples.jsonl') -ScriptBlock {
     param($taskMonitorContainer,$taskMonitorFile)
     while ((& docker inspect $taskMonitorContainer --format '{{.State.Running}}' 2>$null) -eq 'true') {
      $taskMonitorStats=& docker stats --no-stream --format '{{json .}}' $taskMonitorContainer 2>$null
      $taskMonitorGpu=& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null
      @{timestamp=[DateTimeOffset]::UtcNow.ToString('o');docker_stats=$taskMonitorStats;gpu=$taskMonitorGpu} |
       ConvertTo-Json -Compress | Add-Content -LiteralPath $taskMonitorFile -Encoding utf8
      Start-Sleep -Seconds 5
     }
    }
    $taskV4Healthy=$false
    while ($taskV4Timer.Elapsed.TotalSeconds -lt 420) {
     try { Invoke-WebRequest -UseBasicParsing -Uri ($taskV4Entry.Host+$taskV4Entry.Health) -TimeoutSec 3 -ErrorAction Stop | Out-Null;$taskV4Healthy=$true;break }
     catch { Start-Sleep -Seconds 2 }
    }
    if (-not $taskV4Healthy) { throw 'Engine health timed out' }
    @{healthy=$true;ready_seconds=$taskV4Timer.Elapsed.TotalSeconds;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
     ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskV4Dir 'readiness.json') -Encoding utf8
    if ($taskV4Entry.Name -eq 'ollama') {
     Invoke-RestMethod -Method Post -Uri ($taskV4Entry.Host+'/api/show') -ContentType 'application/json' -Body '{"model":"bge-m3"}' |
      ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $taskV4Dir 'model_details.json') -Encoding utf8
    }
    Invoke-V4Stage $taskV4Dataset $taskV4Entry.Name $taskV4Entry.Client 'embeddings'
    Invoke-V4Stage $taskV4Dataset $taskV4Entry.Name $taskV4Entry.Client 'serving'
    Invoke-V4Stage $taskV4Dataset $taskV4Entry.Name $taskV4Entry.Client 'attacks'
    $taskV4Runtime=if ($taskV4Entry.Name -eq 'ollama') { '/api/ps' } else { '/v1/models' }
    Invoke-RestMethod -Uri ($taskV4Entry.Host+$taskV4Runtime) -TimeoutSec 5 |
     ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $taskV4Dir 'model_runtime.json') -Encoding utf8
    $taskV4Ok=$true
   } finally {
    $ErrorActionPreference='Continue'
    & docker compose logs --no-color $taskV4Entry.Name 2>&1 | Set-Content -LiteralPath (Join-Path $taskV4Dir 'engine.log') -Encoding utf8
    & docker compose stop $taskV4Entry.Name
    if ($LASTEXITCODE -ne 0) { throw 'Engine shutdown failed' }
    if ($taskV4Job) {
     Wait-Job $taskV4Job -Timeout 10 | Out-Null
     if ($taskV4Job.State -eq 'Running') { Stop-Job $taskV4Job }
     Receive-Job $taskV4Job | Out-Null;Remove-Job $taskV4Job
    }
    $taskV4Container=& docker compose ps -a -q $taskV4Entry.Name
    $taskV4State=(& docker inspect $taskV4Container --format '{{json .State}}') | ConvertFrom-Json
    $taskV4Http=$false
    try { Invoke-WebRequest -UseBasicParsing -Uri ($taskV4Entry.Host+$taskV4Entry.Health) -TimeoutSec 2 -ErrorAction Stop | Out-Null;$taskV4Http=$true } catch {}
    $taskV4Active=@(& docker ps --filter label=com.docker.compose.project=ubot-v4-runtime --format '{{.Names}}')
    $taskV4Stopped=(-not $taskV4State.Running) -and (-not $taskV4Http) -and $taskV4Active.Count -eq 0
    @{verified_stopped=$taskV4Stopped;container_state=$taskV4State;http_open=$taskV4Http;active_containers=$taskV4Active;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
     ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskV4Dir 'stop_verification.json') -Encoding utf8
    if (-not $taskV4Stopped) { throw 'Shutdown verification failed' }
    Write-Output "STOP VERIFIED: $taskV4Dataset $($taskV4Entry.Name)"
   }
   if ($taskV4Ok) {
    @{complete=$true;dataset_id=$taskV4Dataset;engine=$taskV4Entry.Name;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
     ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskV4Dir 'dense_complete.json') -Encoding utf8
   }
  }
 }
 Save-V4State 'complete' 'all_four' 'ollama+vllm' 'eight_runs_stopped'
} catch {
 Save-V4State 'failed' $taskV4Dataset $taskV4Entry.Name 'pipeline_error' $_.Exception.Message
 throw
}
