param(
    [Parameter(Mandatory = $true)]
    [string]$SearchSpacePath,
    [string]$EvidenceDb,
    [string]$EvidenceRunId
)

$ErrorActionPreference = "Stop"
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$commandPath = Join-Path $workspaceRoot ".venv\Scripts\bci-scout-literature.exe"
if (-not (Test-Path -LiteralPath $commandPath)) {
    throw "Literature scout command not found: $commandPath"
}
$resolvedSearchSpace = Join-Path $workspaceRoot $SearchSpacePath
if (-not (Test-Path -LiteralPath $resolvedSearchSpace)) {
    throw "Search-space draft not found: $resolvedSearchSpace"
}
$contract = Get-Content -Raw -LiteralPath $resolvedSearchSpace | ConvertFrom-Json
$datasetId = [string]$contract.dataset_id
if ([string]::IsNullOrWhiteSpace($datasetId)) {
    throw "Search-space draft has no dataset_id: $resolvedSearchSpace"
}
$safeDatasetId = $datasetId -replace '[^A-Za-z0-9._-]', '_'

$secureKey = Read-Host "Kimi API Key (input is hidden)" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw "Kimi API Key cannot be empty."
    }
    $env:MOONSHOT_API_KEY = $plainKey
    $runId = $safeDatasetId + "-literature-kimi-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    $runDir = Join-Path $workspaceRoot ("artifacts\runs\" + $runId)
    $commandArguments = @(
        "--search-space", $resolvedSearchSpace,
        "--provider", "kimi",
        "--model", "kimi-k2.7-code",
        "--max-output-tokens", "16384",
        "--provider-timeout-seconds", "240",
        "--run-id", $runId,
        "--run-dir", $runDir
    )
    if (-not [string]::IsNullOrWhiteSpace($EvidenceDb)) {
        if ([string]::IsNullOrWhiteSpace($EvidenceRunId)) {
            throw "EvidenceRunId is required when EvidenceDb is supplied."
        }
        $resolvedEvidenceDb = Join-Path $workspaceRoot $EvidenceDb
        if (-not (Test-Path -LiteralPath $resolvedEvidenceDb)) {
            throw "Evidence database not found: $resolvedEvidenceDb"
        }
        $commandArguments += @(
            "--evidence-db", $resolvedEvidenceDb,
            "--evidence-run-id", $EvidenceRunId
        )
    }
    & $commandPath @commandArguments
    exit $LASTEXITCODE
}
finally {
    Remove-Item Env:MOONSHOT_API_KEY -ErrorAction SilentlyContinue
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
    Remove-Variable plainKey -ErrorAction SilentlyContinue
}
