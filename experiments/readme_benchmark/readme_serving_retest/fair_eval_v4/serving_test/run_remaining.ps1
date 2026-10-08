$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskRemainingOutput=Join-Path $PSScriptRoot 'outputs/run-v4'
function Invoke-RemainingStage {
 param([string]$Service,[string[]]$Arguments,[string]$Stage)
 $taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-fair-v4 --format '{{.Names}}')
 if ($taskActive.Count -gt 0) { throw "Previous benchmark container still active: $taskActive" }
 @{status='running';stage=$Stage;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRemainingOutput 'remaining_status.json') -Encoding utf8
 $ErrorActionPreference='Continue'
 & docker compose run --rm --no-deps $Service @Arguments 2>&1 |
  Tee-Object -FilePath (Join-Path $taskRemainingOutput "logs/$Stage.log")
 if ($LASTEXITCODE -ne 0) { throw "Failed: $Stage" }
 $taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-fair-v4 --format '{{.Names}}')
 if ($taskActive.Count -gt 0) { throw "Worker remained active: $taskActive" }
}
try {
 @{status='waiting';stage='dense';timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRemainingOutput 'remaining_status.json') -Encoding utf8
 while ($true) {
  $taskDense=Get-Content -LiteralPath (Join-Path $taskRemainingOutput 'dense_status.json') -Raw | ConvertFrom-Json
  if ($taskDense.status -eq 'complete') { break }
  if ($taskDense.status -eq 'failed') { throw 'Dense stage failed; fix before continuing' }
  Start-Sleep -Seconds 5
 }
 Invoke-RemainingStage 'validate' @('test_native_math.py') 'native-math-contracts'
 Invoke-RemainingStage 'worker' @('run_rerank.py','bge') 'rerank-bge'
 Invoke-RemainingStage 'jina-worker' @('run_rerank.py','jina') 'rerank-jina'
 Invoke-RemainingStage 'worker' @('run_native.py') 'native-sparse-multivector'
 Invoke-RemainingStage 'validate' @('compare_v4.py') 'comparison-audit'
 $taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-fair-v4 --format '{{.Names}}')
 $taskPorts=@()
 foreach ($taskUrl in @('http://127.0.0.1:11436/api/tags','http://127.0.0.1:8002/health')) {
  try { Invoke-WebRequest -UseBasicParsing -Uri $taskUrl -TimeoutSec 2 -ErrorAction Stop | Out-Null;$taskPorts+=$taskUrl } catch {}
 }
 $taskStopped=$taskActive.Count -eq 0 -and $taskPorts.Count -eq 0
 @{verified_stopped=$taskStopped;active_containers=$taskActive;responding_urls=$taskPorts;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRemainingOutput 'final_shutdown.json') -Encoding utf8
 if (-not $taskStopped) { throw 'Final shutdown failed' }
 @{status='complete';stage='all_stages';timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRemainingOutput 'remaining_status.json') -Encoding utf8
} catch {
 @{status='failed';message=$_.Exception.Message;timestamp=[DateTimeOffset]::UtcNow.ToString('o')} |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRemainingOutput 'remaining_status.json') -Encoding utf8
 throw
}
