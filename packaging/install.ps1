# Strata Terminal: one-line installer for Windows (PowerShell 5.1 or newer).
#
#     irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
#
# What it does, in order (and nothing else):
#   1. finds out which computer this is (Windows, processor, graphics card);
#   2. downloads the Terminal archive named in the release folder's install-manifest.txt and checks its SHA-256 against it;
#   3. unpacks it into  %LOCALAPPDATA%\Strata\terminal\<version>  (no administrator rights);
#   4. checks the release's signed feed with the Terminal it just unpacked (`strata update --check`); if that fails it removes what it
#      unpacked and stops;
#   5. adds  %LOCALAPPDATA%\Strata\terminal\current\bin  to your user PATH and prints:  Installed. Run: strata launch
#
# The engine and the model are not part of this: `strata launch` fetches the engine for your graphics card the first time you choose
# a model, from the same release folder, signed.
#
# <feed> is where the release folder is hosted: GitHub Releases of Niko1221/Strata (the "latest" release's files), built in as the default.
# Set $env:STRATA_FEED to another folder (a URL) to use that instead, for example a local test feed:  $env:STRATA_FEED='http://127.0.0.1:8099'; irm http://127.0.0.1:8099/install.ps1 | iex
# Optional: $env:STRATA_DATA_DIR (default %LOCALAPPDATA%\Strata), $env:STRATA_INSTALL_TEST_TRUST='1' (a test feed: trust only its
# test-trust.json for step 4), $env:STRATA_NO_PATH='1' (do not change the user PATH).
& {   # a child scope: `irm | iex` runs in the caller's scope, and this script's variables must not overwrite the caller's
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Die([string]$msg) { [Console]::Error.WriteLine("strata installer: $msg"); throw "strata installer stopped" }
# older Windows 10 builds offer only TLS 1.0/1.1 to Windows PowerShell 5.1, and GitHub accepts only 1.2; a company proxy may want the Windows sign-in
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }
try { if ([Net.WebRequest]::DefaultWebProxy) { [Net.WebRequest]::DefaultWebProxy.Credentials = [Net.CredentialCache]::DefaultNetworkCredentials } } catch { }

$feed = if ($env:STRATA_FEED) { $env:STRATA_FEED } else { 'https://github.com/Niko1221/Strata/releases/latest/download' }
$feed = $feed.TrimEnd('/')
if (-not $feed) {
    Die 'this installer does not know where the Strata release folder is. Set $env:STRATA_FEED to its address and run it again, for example: $env:STRATA_FEED=''https://example.org/strata'''
}
if ([Environment]::OSVersion.Platform -ne 'Win32NT') { Die 'this is the Windows installer; on Linux or macOS use install.sh.' }
$arch = $env:PROCESSOR_ARCHITEW6432; if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
if ($arch -ne 'AMD64') { Die "this computer ($arch) is a Windows on ARM PC. Strata needs an NVIDIA or AMD graphics card and has no engine for ARM processors, so it could not run a model here. Run this installer on a Windows PC with an Intel or AMD processor (x64) and one of those cards." }
$platform = 'windows-x86_64'
$gpu = ''
# a discrete card is named before a built-in one (a Ryzen's "AMD Radeon(TM) Graphics" is listed first on many PCs); this line is only for the person to read
try { $cards = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Basic|Remote' }); $gpu = ($cards | Sort-Object @{ Expression = { if ($_.Name -match 'NVIDIA|GeForce|RTX|GTX|Quadro|Radeon (RX|PRO|AI)|Arc|Instinct') { 0 } else { 1 } } } | Select-Object -First 1 -ExpandProperty Name) } catch { }
Write-Host 'Strata Terminal installer'
Write-Host "  This computer: Windows $arch ($platform)$(if ($gpu) { ", graphics: $gpu" })"
Write-Host "  Release folder: $feed"

if (-not $env:STRATA_DATA_DIR -and -not $env:LOCALAPPDATA) { Die 'Windows did not say where your user folder is. Set $env:STRATA_DATA_DIR to a folder you own, for example $env:STRATA_DATA_DIR=''C:\Strata'', and run this again.' }
$data = if ($env:STRATA_DATA_DIR) { $env:STRATA_DATA_DIR } else { Join-Path $env:LOCALAPPDATA 'Strata' }
$work = Join-Path ([IO.Path]::GetTempPath()) ("strata-install-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $work | Out-Null
try {
    try { Invoke-WebRequest -UseBasicParsing -Uri "$feed/install-manifest.txt" -OutFile (Join-Path $work 'manifest.txt') }
    catch { Die "could not read $feed/install-manifest.txt ($($_.Exception.Message)). Check the address and your internet connection; behind a company proxy or firewall, ask for the release folder to be allowed." }
    $line = Get-Content (Join-Path $work 'manifest.txt') | Where-Object { $_ -match "^$platform\s" } | Select-Object -First 1
    if (-not $line) { Die "this release has no Terminal for $platform." }
    $f = -split $line
    $version, $rel, $want, $size = $f[1], $f[2], $f[3].ToLower(), $f[4]
    Write-Host "  Terminal $version ($([math]::Round([double]$size / 1MB)) MB)"
    Write-Host 'Downloading...'
    $zip = Join-Path $work 'terminal.zip'
    try { Invoke-WebRequest -UseBasicParsing -Uri "$feed/$rel" -OutFile $zip }
    catch { Die "the download failed ($($_.Exception.Message)). Check your internet connection and run this again; nothing was installed." }
    $got = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    if ($got -ne $want) { Die "the download does not match the release's checksum (got $got, wanted $want). It was not installed; try again." }
    Write-Host '  Checksum ok.'

    $dest = Join-Path $data "terminal\$version"
    $incoming = "$dest.incoming"
    New-Item -ItemType Directory -Force -Path (Join-Path $data 'terminal') | Out-Null
    if (Test-Path $incoming) { Remove-Item -Recurse -Force $incoming }
    $tooLong = 'If it says a path is too long, set $env:STRATA_DATA_DIR to a short folder such as C:\Strata and run this again; otherwise check the free disk space.'
    try {
        # tar.exe (Windows 10 1803 and later) handles long paths; Expand-Archive in Windows PowerShell 5.1 does not.
        # tar.exe receives its arguments through the ANSI/OEM code page, so a folder with an accented letter in its name turns into one with a question mark and the
        # unpack fails (A04). It is therefore started with ASCII-only, relative arguments from inside the data folder (PowerShell's own cmdlets are Unicode-safe),
        # unpacks into a folder with an ASCII name, and that folder is renamed to $incoming.
        # Windows' own bsdtar first: a GNU tar from Git for Windows or MSYS earlier on the PATH cannot read a zip
        $tar = Get-Command (Join-Path $env:SystemRoot 'System32\tar.exe') -ErrorAction SilentlyContinue
        if (-not $tar) { $tar = Get-Command tar.exe -ErrorAction SilentlyContinue }
        if ($tar) {
            $tag = 'u' + [Guid]::NewGuid().ToString('N').Substring(0, 6)
            $terminalDir = Join-Path $data 'terminal'
            Push-Location -LiteralPath $terminalDir
            try {
                Copy-Item -LiteralPath $zip -Destination "$tag.zip"
                New-Item -ItemType Directory -Path $tag | Out-Null
                $tarOut = & $tar.Source -xf "$tag.zip" -C $tag 2>&1 | Out-String
                if ($LASTEXITCODE -ne 0) { throw $tarOut.Trim() }
            }
            finally {
                Remove-Item -Force -ErrorAction SilentlyContinue -LiteralPath "$tag.zip"
                Pop-Location
            }
            Move-Item -LiteralPath (Join-Path $terminalDir $tag) -Destination $incoming
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -LiteralPath (Join-Path $terminalDir $tag)
        } else { Expand-Archive -LiteralPath $zip -DestinationPath $incoming -Force }
    }
    catch { Remove-Item -Recurse -Force $incoming -ErrorAction SilentlyContinue; Die "the archive could not be unpacked ($($_.Exception.Message)). $tooLong Nothing was installed." }
    # the archive of a release folder is rooted at the bundle itself (control package) or in one folder (terminal archive)
    if (-not (Test-Path (Join-Path $incoming 'bin\strata.cmd'))) {
        $inner = Get-ChildItem $incoming -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'bin\strata.cmd') } | Select-Object -First 1
        if (-not $inner) { Remove-Item -Recurse -Force $incoming; Die 'the archive does not look like a Strata Terminal.' }
        $moved = "$dest.moved"; Move-Item $inner.FullName $moved; Remove-Item -Recurse -Force $incoming; Move-Item $moved $incoming
    }

    if ($env:STRATA_INSTALL_TEST_TRUST -eq '1') {
        $trust = Join-Path $work 'test-trust.json'
        try { Invoke-WebRequest -UseBasicParsing -Uri "$feed/test-trust.json" -OutFile $trust } catch { Remove-Item -Recurse -Force $incoming; Die 'the test feed has no test-trust.json' }
        $env:STRATA_UPDATE_TRUSTED_KEYS = $trust
    }
    Write-Host "Checking the release's signature..."
    $strata = Join-Path $incoming 'bin\strata.cmd'
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'      # a native program's stderr note must not stop the script
    $out = & $strata update --check --no-engine --feed $feed 2>&1 | Out-String
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($code -ne 0) {
        ($out -split "`r?`n" | Where-Object { $_.Trim() } | ForEach-Object { "    $_" }) | ForEach-Object { [Console]::Error.WriteLine($_) }
        Remove-Item -Recurse -Force $incoming
        Die "the release folder's signed feed could not be verified, so nothing was installed. The reason is printed above; if it names a newer feed already accepted here, use the newest release folder, or follow the step it gives. If you trust this source and see no such reason, ask its owner to publish a signed feed."
    }
    Write-Host '  Signature ok.'

    if (Test-Path $dest) {
        try { Remove-Item -Recurse -Force $dest }
        catch { Remove-Item -Recurse -Force $incoming -ErrorAction SilentlyContinue; Die "Strata $version is already installed and a program is using it (a Strata window or server is running). Close it (run: strata launch --stop), then run this again. Nothing was changed." }
    }
    Move-Item $incoming $dest
    $current = Join-Path $data 'terminal\current'
    if (Test-Path $current) { cmd /c rmdir "$current" | Out-Null }
    cmd /c mklink /J "$current" "$dest" | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $current 'bin\strata.cmd'))) {
        # a drive that cannot hold a junction (not NTFS) or a policy that forbids it: a small launcher in its place does the same job
        try {
            if (Test-Path $current) { Remove-Item -Recurse -Force $current -ErrorAction SilentlyContinue }
            New-Item -ItemType Directory -Force -Path (Join-Path $current 'bin') | Out-Null
            Set-Content -Encoding Ascii -Path (Join-Path $current 'bin\strata.cmd') -Value @('@echo off', ('"' + (Join-Path $dest 'bin\strata.cmd') + '" %*'))
            Set-Content -Encoding Ascii -Path (Join-Path $current 'current.txt') -Value $dest
            Write-Host '  (this drive could not make a link, so a small launcher was used instead)'
        }
        catch { Die "Strata was unpacked but its start command could not be set up ($($_.Exception.Message)). Run it directly: $dest\bin\strata.cmd" }
    }
    $bin = Join-Path $current 'bin'
    if ($env:STRATA_NO_PATH -ne '1') {
        $user = [Environment]::GetEnvironmentVariable('Path', 'User')
        if (-not $user) { $user = '' }
        if (($user -split ';') -notcontains $bin) {
            [Environment]::SetEnvironmentVariable('Path', ($user.TrimEnd(';') + ';' + $bin).TrimStart(';'), 'User')
        }
        $env:Path = "$env:Path;$bin"
    }
    Write-Host 'Installed. Run: strata launch'
    if ($env:STRATA_NO_PATH -eq '1') { Write-Host "  (the program is $bin\strata.cmd)" }
    else { Write-Host '  (open a new terminal window if the command is not found yet)' }
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
}
