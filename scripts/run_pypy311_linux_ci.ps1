[CmdletBinding()]
param(
    [string]$Image = "kogwistar-pypy311-ci:local",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
$pypyUrl = "https://downloads.python.org/pypy/pypy3.11-v7.3.20-linux64.tar.bz2"
$pypySha256 = "1410db3a7ae47603e2b7cbfd7ff6390b891b2e041c9eb4f1599f333677bccb3e"

if (-not $SkipBuild) {
    docker build --file (Join-Path $root "Dockerfile.pypy-ci") `
        --build-arg "PYPY_URL=$pypyUrl" `
        --build-arg "PYPY_SHA256=$pypySha256" `
        --build-arg "PYPY_VERSION=3.11" `
        --tag $Image $root
    if ($LASTEXITCODE -ne 0) { throw "PyPy test image build failed with exit code $LASTEXITCODE" }
}

docker run --rm -i `
    --env KOGWISTAR_IMPL_MODE=python `
    --env KOGWISTAR_LOG_LEVEL=WARNING `
    --env LOG_LEVEL=WARNING `
    --volume "${root}:/workspace" `
    --workdir /workspace `
    --entrypoint pypy3 $Image `
    scripts/run_pypy311_ci.py --venv /tmp/pypy311-venv
if ($LASTEXITCODE -ne 0) { throw "Kogwistar PyPy 3.11 Linux CI profile failed with exit code $LASTEXITCODE" }
