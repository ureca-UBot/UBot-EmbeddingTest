param([ValidateSet('All','Dense','Auxiliary','Validate')][string]$Stage='All')
. (Join-Path $PSScriptRoot 'scripts/compose_helpers.ps1')
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'serving_test/outputs') | Out-Null
Invoke-ComposeChecked config --quiet
Invoke-ComposeChecked run --rm --no-deps validate test_runner.py
Invoke-ComposeChecked run --rm --no-deps validate run_v4.py preflight
if ($Stage -eq 'Validate') { return }
Push-Location -LiteralPath (Join-Path $PSScriptRoot 'serving_test')
try {
 if ($Stage -in @('All','Dense')) { & .\run_dense.ps1 }
 if ($Stage -in @('All','Auxiliary')) { & .\run_remaining.ps1 }
} finally { Pop-Location }
