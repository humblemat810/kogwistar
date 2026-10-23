param(
    [string]$OutputRoot = ".tmp/mixed-runtime-matrix-20261022",
    [int]$CpuLimit = 4,
    [string]$MemoryLimit = "8g",
    [int]$Samples = 5
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dockerfile = Join-Path $PSScriptRoot "runtime_matrix/Dockerfile"
$runtimeMatrixContext = Join-Path $PSScriptRoot "runtime_matrix"
$output = Join-Path $repo $OutputRoot
New-Item -ItemType Directory -Force -Path $output | Out-Null
$memoryMatch = [regex]::Match($MemoryLimit, '^(\d+)([kKmMgG])$')
if (-not $memoryMatch.Success) { throw "MemoryLimit must be an integer followed by k, m, or g" }
$memoryMultiplier = switch ($memoryMatch.Groups[2].Value.ToLowerInvariant()) {
    "k" { 1024 }
    "m" { 1024 * 1024 }
    "g" { 1024 * 1024 * 1024 }
}
$memoryBytes = [int64]$memoryMatch.Groups[1].Value * $memoryMultiplier
$sourceStage = Join-Path $output "source-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force -Path (Join-Path $sourceStage "scripts") | Out-Null
$packageSource = Join-Path $repo "kogwistar"
if (-not (Test-Path -LiteralPath (Join-Path $packageSource "__init__.py"))) {
    throw "Kogwistar import package was not found at $packageSource"
}
Copy-Item -Recurse -LiteralPath $packageSource -Destination (Join-Path $sourceStage "kogwistar")
foreach ($scriptName in @(
    "mixed_concurrent_benchmark.py",
    "mixed_workload.py",
    "mixed_backend_benchmark.py",
    "mixed_runtime_container_benchmark.py"
)) {
    Copy-Item (Join-Path $PSScriptRoot $scriptName) (Join-Path $sourceStage "scripts/$scriptName")
}
$stagedRust = Join-Path $sourceStage "rust"
New-Item -ItemType Directory -Force -Path $stagedRust | Out-Null
foreach ($rustFile in @("Cargo.toml", "Cargo.lock", "deny.toml", "rust-toolchain.toml")) {
    $rustFilePath = Join-Path (Join-Path $repo "rust") $rustFile
    if (Test-Path -LiteralPath $rustFilePath) {
        Copy-Item -LiteralPath $rustFilePath -Destination $stagedRust
    }
}
Copy-Item -Recurse -LiteralPath (Join-Path $repo "rust/crates") -Destination (Join-Path $stagedRust "crates")
foreach ($rootFile in @("pyproject.toml", "Cargo.lock", "README.md", "LICENSE")) {
    $rootFilePath = Join-Path $repo $rootFile
    if (Test-Path -LiteralPath $rootFilePath) {
        Copy-Item -LiteralPath $rootFilePath -Destination $sourceStage
    }
}
$requirementsSource = Join-Path $runtimeMatrixContext "requirements.txt"
Copy-Item -LiteralPath $requirementsSource -Destination (Join-Path $sourceStage "requirements.txt")
$contractSource = Join-Path $repo "contracts"
if (Test-Path $contractSource) {
    Copy-Item -Recurse $contractSource (Join-Path $sourceStage "contracts")
}
$resolvedSource = (Resolve-Path $sourceStage).Path
$sourceFiles = @(
    Get-ChildItem $sourceStage -Recurse -File |
        Where-Object { $_.Extension -ne ".pyc" -and $_.FullName -notmatch '[\\/]__pycache__[\\/]' } |
        Sort-Object FullName |
        ForEach-Object {
            [pscustomobject]@{
                path = $_.FullName.Substring($sourceStage.Length + 1).Replace('\', '/')
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        }
)
$sourceManifestJson = ConvertTo-Json -InputObject $sourceFiles -Depth 4 -Compress
$sourceManifestPath = Join-Path $output "runtime-source-manifest.json"
Set-Content -LiteralPath $sourceManifestPath -Value $sourceManifestJson -Encoding utf8
$shaProvider = [System.Security.Cryptography.SHA256]::Create()
try {
    $sourceManifestDigest = [BitConverter]::ToString(
        $shaProvider.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($sourceManifestJson))
    ).Replace('-', '').ToLowerInvariant()
} finally {
    $shaProvider.Dispose()
}
$manifestPath = Join-Path $output "runtime-matrix-manifest.json"
$manifest = @{
    schema = "kogwistar-fixed-resource-runtime-matrix/v1"
    created_utc = [DateTime]::UtcNow.ToString("o")
    host = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    docker_version = (docker version --format '{{.Server.Version}}' | Out-String).Trim()
    git_revision = (git -C $repo rev-parse HEAD | Out-String).Trim()
    git_status = @(git -C $repo status --short)
    dockerfile_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $dockerfile).Hash.ToLowerInvariant()
    requirements_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $runtimeMatrixContext "requirements.txt")).Hash.ToLowerInvariant()
    source_manifest_sha256 = $sourceManifestDigest
    source_manifest_path = "runtime-source-manifest.json"
    limits = @{ cpus = $CpuLimit; memory = $MemoryLimit; memory_swap = $MemoryLimit; network = "none" }
    runtimes = @()
}

$matrix = @(
    @{ Name = "cpython314-gil"; Target = "cpython314-gil"; Items = @(1000, 2000) },
    @{ Name = "cpython315-gil"; Target = "cpython315-gil"; Items = @(1000, 2000) },
    @{ Name = "cpython315-nogil"; Target = "cpython315-nogil"; Items = @(1000, 2000) },
    @{ Name = "cpython315-nogil-rust"; Target = "cpython315-nogil-rust"; Authority = "rust"; Items = @(1000, 2000) },
    @{ Name = "pypy311"; Target = "pypy311"; Items = @(1000, 2000) }
)

foreach ($entry in $matrix) {
    $tag = "kogwistar-mixed-runtime:$($entry.Name)"
    Write-Host "Building $tag"
    docker build --file $dockerfile --target $entry.Target --tag $tag $resolvedSource
    if ($LASTEXITCODE -ne 0) { throw "Docker build failed for $tag" }
    $imageInfo = (docker image inspect $tag | ConvertFrom-Json)[0]
    $runtimeRecord = @{
        name = $entry.Name
        target = $entry.Target
        image_id = $imageInfo.Id
        repo_digests = @($imageInfo.RepoDigests)
        created = $imageInfo.Created
        status = "built"
    }
    $manifest.runtimes += $runtimeRecord
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $manifestPath

    $runtimeOutput = Join-Path $output $entry.Name
    New-Item -ItemType Directory -Force -Path $runtimeOutput | Out-Null
    $resolvedOutput = (Resolve-Path $runtimeOutput).Path
    $itemArgs = @()
    foreach ($size in $entry.Items) { $itemArgs += @("--items", "$size") }
    Write-Host "Running $($entry.Name) with --cpus=$CpuLimit --memory=$MemoryLimit --memory-swap=$MemoryLimit"
    $authority = if ($entry.Authority) { $entry.Authority } else { "python" }
    docker run --rm --network none --cpus $CpuLimit --memory $MemoryLimit --memory-swap $MemoryLimit `
        --mount "type=bind,source=$resolvedOutput,target=/results" `
        $tag --output /results --samples $Samples --expected-runtime $entry.Name --authority $authority `
        --expected-cpus $CpuLimit --expected-memory-bytes $memoryBytes @itemArgs
    if ($LASTEXITCODE -ne 0) { throw "Runtime verification/benchmark failed for $($entry.Name)" }
    $runtimeResult = Get-Content -LiteralPath (Join-Path $runtimeOutput "runtime-report.json") -Raw | ConvertFrom-Json
    $runtimeRecord.status = "complete"
    $runtimeRecord.runtime_verified = [bool]$runtimeResult.runtime_verified
    $runtimeRecord.limits_verified = [bool]$runtimeResult.limits_verified
    $runtimeRecord.packages = $runtimeResult.runtime.packages
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $manifestPath
}

$verifier = Join-Path $PSScriptRoot "verify_mixed_runtime_matrix.py"
python $verifier --root $output
if ($LASTEXITCODE -ne 0) { throw "Runtime matrix parity verification failed; see runtime-matrix-verification.json" }
$manifest.status = "complete"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $manifestPath
