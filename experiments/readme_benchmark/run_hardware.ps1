$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath 'outputs/run-v1/serving/shortlist.json')) { throw 'Shortlist serving engines first' }
foreach ($taskDevice in @('cpu','cuda')) {
    $taskService = if ($taskDevice -eq 'cpu') { 'hardware-cpu' } else { 'benchmark' }
    & docker compose --profile hardware run --rm $taskService scripts/benchmark_cpu_gpu.py --device $taskDevice --requests 30 2>&1 |
        Tee-Object -FilePath "outputs/run-v1/logs/hardware-$taskDevice.log"
    if ($LASTEXITCODE -ne 0) { throw "Hardware probe failed: $taskDevice" }
    $taskActive = & docker ps --filter label=com.docker.compose.project=ubot-readme-benchmark --format '{{.Names}}'
    if ($taskActive) { throw 'Hardware container did not exit' }
}
& docker compose --profile validation run --rm validate scripts/combine_hardware.py
if ($LASTEXITCODE -ne 0) { throw 'Hardware parity check failed' }
