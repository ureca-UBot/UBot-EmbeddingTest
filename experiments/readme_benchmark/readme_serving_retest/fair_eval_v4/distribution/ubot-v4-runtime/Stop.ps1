. (Join-Path $PSScriptRoot 'scripts/compose_helpers.ps1')
Invoke-ComposeChecked stop
$taskOneOff=@(& docker ps --filter label=com.docker.compose.project=ubot-v4-runtime --format '{{.ID}}')
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect remaining v4 containers' }
if ($taskOneOff.Count -gt 0) { Invoke-DockerChecked stop @taskOneOff }
$taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-v4-runtime --format '{{.Names}}')
if ($LASTEXITCODE -ne 0 -or $taskActive.Count -gt 0) { throw 'Some v4 runtime containers are still running' }
foreach ($taskUrl in @('http://127.0.0.1:11436/api/tags','http://127.0.0.1:8002/health')) {
 $taskResponded=$false
 try { Invoke-WebRequest -UseBasicParsing -Uri $taskUrl -TimeoutSec 2 | Out-Null;$taskResponded=$true } catch {}
 if ($taskResponded) { throw "Endpoint still responds: $taskUrl" }
}
Write-Output 'Stopped. Model volumes and benchmark outputs are preserved.'
