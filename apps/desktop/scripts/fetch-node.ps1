# Fetch and verify the private Node runtime the Windows desktop bundle ships (win-x64).
#
# The PowerShell twin of fetch-node.sh: same pinned version, same SHA-256 pin, same refusal to
# install anything whose digest does not match. Not committed output: node.exe is a ~90 MB binary.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File apps\desktop\scripts\fetch-node.ps1
#   ... -File fetch-node.ps1 -Check                  # verify what is already installed
#   ... -File fetch-node.ps1 -FromFile node.zip      # verify and install a zip you already have (offline)
#
# The Node version must satisfy the App Host's own engines.node (>= 24.12.0).
[CmdletBinding()]
param(
    [switch]$Check,
    [string]$FromFile = '',
    [string]$Dest = ''
)
$ErrorActionPreference = 'Stop'

$NodeVersion = if ($env:STRATA_NODE_VERSION) { $env:STRATA_NODE_VERSION } else { '24.12.0' }
$Target = 'win-x64'
# SHA-256 of node-v24.12.0-win-x64.zip from nodejs.org's SHASUMS256.txt; keep equal to fetch-node.sh.
$Expected = '9c125f61ae947b52e779095830f9cac267846a043ef7192183c84016aaad2812'
if (-not $Dest) { $Dest = Join-Path $PSScriptRoot '..\src-tauri\resources\node' }
$Installed = Join-Path $Dest 'node.exe'

function Fail([string]$Message) { [Console]::Error.WriteLine("fetch-node: $Message"); exit 1 }

if ($Check) {
    if (-not (Test-Path -LiteralPath $Installed)) { Fail "not installed at $Installed" }
    $got = (& $Installed --version).Trim()
    Write-Output "fetch-node: $Installed is $got"
    if ($got -ne "v$NodeVersion") { Fail "expected v$NodeVersion" }
    exit 0
}

$Archive = "node-v$NodeVersion-$Target.zip"
$Work = Join-Path ([IO.Path]::GetTempPath()) ("strata-node-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Work | Out-Null
try {
    $Zip = Join-Path $Work $Archive
    if ($FromFile) {
        Copy-Item -LiteralPath $FromFile -Destination $Zip
    } else {
        $Url = "https://nodejs.org/dist/v$NodeVersion/$Archive"
        Write-Output "fetch-node: $Url"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Zip
    }
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Zip).Hash.ToLowerInvariant()
    if ($NodeVersion -ne '24.12.0') {
        # another version has no recorded digest: say so rather than implying a verification
        [Console]::Error.WriteLine("fetch-node: WARNING - no recorded digest for v$NodeVersion; this download is NOT authenticated.")
        [Console]::Error.WriteLine("fetch-node: computed sha256 $Actual - record it in this script and fetch-node.sh.")
    } elseif ($Actual -ne $Expected) {
        Fail "REFUSING - $Archive hashes $Actual, expected $Expected"
    } else {
        Write-Output 'fetch-node: sha256 verified against the recorded digest'
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zipFile = [IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        $entry = $zipFile.GetEntry("node-v$NodeVersion-$Target/node.exe")
        if (-not $entry) { Fail "node.exe is not inside $Archive" }
        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $Installed, $true)
    } finally { $zipFile.Dispose() }
    Write-Output "fetch-node: installed at $Installed"
} finally {
    Remove-Item -LiteralPath $Work -Recurse -Force -ErrorAction SilentlyContinue
}
