param(
 [Parameter(Mandatory=$true)][string]$DatasetZip,
 [switch]$ReuseLocalImages,
 [switch]$SkipModelDownload
)
. (Join-Path $PSScriptRoot 'scripts/compose_helpers.ps1')
$taskInputZip=(Resolve-Path -LiteralPath $DatasetZip).Path
$taskDataDir=Join-Path $PSScriptRoot 'datasets'
New-Item -ItemType Directory -Force -Path $taskDataDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'serving_test/outputs') | Out-Null
Invoke-DockerChecked info --format '{{.OSType}}'
Invoke-ComposeChecked config --quiet
if ($ReuseLocalImages) {
 $taskLock=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'environment/runtime-lock.json') -Raw | ConvertFrom-Json
 foreach ($taskPair in @(@('ubot-retrieval-benchmark:run-v1','ubot-v4-benchmark:runtime'),@('ubot-retrieval-jina:run-v1','ubot-v4-jina:runtime'))) {
  $taskId=& docker image inspect $taskPair[0] --format '{{.Id}}'
  if ($LASTEXITCODE -ne 0 -or $taskId -ne $taskLock.original_images.($taskPair[0])) { throw "Measured image differs: $($taskPair[0])" }
  Invoke-DockerChecked tag $taskPair[0] $taskPair[1]
 }
} else {
 Invoke-ComposeChecked build worker
 Invoke-ComposeChecked build jina-worker
 Invoke-ComposeChecked pull ollama vllm
}
Invoke-ComposeChecked run --rm --no-deps validate /workspace/scripts/verify_environment.py --kind benchmark
Invoke-ComposeChecked run --rm --no-deps jina-worker /workspace/scripts/verify_environment.py --kind jina
Invoke-ComposeChecked run --rm --no-deps --volume "${taskInputZip}:/inputs/questions.zip:ro" --volume "${taskDataDir}:/data:rw" validate /workspace/scripts/import_dataset.py --source /inputs/questions.zip --destination /data
if ($SkipModelDownload) {
 Invoke-ComposeChecked run --rm --no-deps validate /workspace/scripts/prepare_models.py --verify-only
} else {
 Invoke-ComposeChecked run --rm --no-deps model-setup
}
$taskStartedOllama=$false
try {
 $taskActive=@(& docker ps --filter label=com.docker.compose.project=ubot-v4-runtime --format '{{.Names}}')
 if ($LASTEXITCODE -ne 0 -or $taskActive.Count -gt 0) { throw 'A v4 runtime container is already running' }
 Invoke-ComposeChecked up -d ollama
 $taskStartedOllama=$true
 Wait-OllamaReady
 if (-not $SkipModelDownload) { Invoke-ComposeChecked exec -T ollama ollama pull bge-m3:latest }
 $taskTags=Invoke-RestMethod -Uri 'http://127.0.0.1:11436/api/tags' -TimeoutSec 10
 $taskModel=$taskTags.models | Where-Object { $_.name -eq 'bge-m3:latest' }
 $taskLock=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'environment/runtime-lock.json') -Raw | ConvertFrom-Json
 if ($taskModel.digest -ne $taskLock.ollama_model.manifest_digest) { throw 'Ollama bge-m3 tag differs from the measured digest. Use the original model cache.' }
 $taskDetails=Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:11436/api/show' -ContentType 'application/json' -Body '{"model":"bge-m3"}'
 if ($taskDetails.details.quantization_level -ne 'F16' -or $taskDetails.modelfile -notmatch $taskLock.ollama_model.gguf_sha256) { throw 'Ollama GGUF weights/precision mismatch' }
} finally {
 if ($taskStartedOllama) { Invoke-ComposeChecked stop ollama }
}
Invoke-ComposeChecked run --rm --no-deps validate test_runner.py
Invoke-ComposeChecked run --rm --no-deps validate test_native_math.py
Invoke-ComposeChecked run --rm --no-deps validate run_v4.py preflight
Write-Output 'v4 preparation complete. Run .\Run.ps1 to start the benchmark.'
