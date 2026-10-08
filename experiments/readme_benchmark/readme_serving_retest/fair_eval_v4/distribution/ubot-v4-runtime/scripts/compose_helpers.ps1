$ErrorActionPreference='Stop'
$taskRuntimeRoot=Split-Path -Parent $PSScriptRoot
$taskRuntimeCompose=Join-Path $taskRuntimeRoot 'serving_test/compose.yaml'
$taskRuntimeEnv=Join-Path $taskRuntimeRoot 'serving_test/.env'
if (-not (Test-Path -LiteralPath $taskRuntimeEnv)) {
 Copy-Item -LiteralPath (Join-Path $taskRuntimeRoot 'serving_test/.env.example') -Destination $taskRuntimeEnv
}
function Invoke-DockerChecked {
 & docker @args
 if ($LASTEXITCODE -ne 0) { throw "Docker command failed (exit $LASTEXITCODE): $($args -join ' ')" }
}
function Invoke-ComposeChecked {
 Invoke-DockerChecked compose --env-file $taskRuntimeEnv -f $taskRuntimeCompose @args
}
function Wait-OllamaReady {
 $taskDeadline=[DateTimeOffset]::UtcNow.AddSeconds(180)
 while ([DateTimeOffset]::UtcNow -lt $taskDeadline) {
  try { Invoke-RestMethod -Uri 'http://127.0.0.1:11436/api/tags' -TimeoutSec 3 | Out-Null; return } catch { Start-Sleep -Seconds 2 }
 }
 throw 'Ollama readiness timed out'
}
