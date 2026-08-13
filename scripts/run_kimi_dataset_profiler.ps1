param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [Parameter(Mandatory = $true)]
    [string]$ValidationPath
)

$ErrorActionPreference = "Stop"
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$commandPath = Join-Path $workspaceRoot ".venv\Scripts\bci-profile-dataset.exe"
if (-not (Test-Path -LiteralPath $commandPath)) {
    throw "Dataset profiler command not found: $commandPath"
}

$secureKey = Read-Host "Kimi API Key (input is hidden)" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw "Kimi API Key cannot be empty."
    }
    $env:MOONSHOT_API_KEY = $plainKey
    $runId = "dataset-profile-kimi-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    $runDir = Join-Path $workspaceRoot ("artifacts\runs\" + $runId)
    & $commandPath `
        --dataset-id auto `
        --dataset-root (Join-Path $workspaceRoot $DatasetRoot) `
        --validation (Join-Path $workspaceRoot $ValidationPath) `
        --provider kimi `
        --model kimi-k2.7-code `
        --max-output-tokens 16384 `
        --run-id $runId `
        --run-dir $runDir
    exit $LASTEXITCODE
}
finally {
    Remove-Item Env:MOONSHOT_API_KEY -ErrorAction SilentlyContinue
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
    Remove-Variable plainKey -ErrorAction SilentlyContinue
}
