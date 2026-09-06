# AvatarCam installer for Windows (native ARM64 or x64).
# Run from PowerShell:  Set-ExecutionPolicy -Scope Process Bypass; .\install.ps1
#
# What it does: checks Python, creates .venv, installs the right requirements file for
# your CPU architecture, downloads and verifies the models, and creates run shortcuts.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$arch = $env:PROCESSOR_ARCHITECTURE
Write-Host "Processor architecture: $arch"

# --- find a Python interpreter ------------------------------------------------------
$py = $null
foreach ($candidate in @("py -3.12", "py -3.11", "python3.12", "python", "py -3")) {
    try {
        $ver = & cmd /c "$candidate -c ""import sys,platform;print(sys.version_info[0],sys.version_info[1],platform.machine())""" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver) { $py = $candidate; break }
    } catch {}
}
if (-not $py) {
    Write-Error "Python 3.11+ not found. Install Python 3.12 (choose the ARM64 installer on a Snapdragon PC) from https://www.python.org/downloads/windows/ and re-run."
}
$parts = $ver.Trim().Split(" ")
$major = [int]$parts[0]; $minor = [int]$parts[1]; $pyArch = $parts[2]
Write-Host "Using $py (Python $major.$minor, $pyArch)"

if ($arch -eq "ARM64" -and $pyArch -ne "ARM64") {
    Write-Warning "This is an ARM64 PC but the Python found is $pyArch (emulated). The native stack needs the ARM64 Python build."
    Write-Warning "Continuing with the x64 requirements (DirectML). Install Python 3.12 ARM64 for NPU support."
}

$native = ($arch -eq "ARM64" -and $pyArch -eq "ARM64")
if ($native -and $minor -ne 12) {
    Write-Error "The Windows ARM64 OpenCV wheel exists only for Python 3.12. Install Python 3.12 ARM64 and re-run."
}

# --- virtual environment --------------------------------------------------------------
if (-not (Test-Path ".venv")) {
    & cmd /c "$py -m venv .venv"
}
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip wheel | Out-Null

$req = if ($native) { "requirements-arm64.txt" } else { "requirements-x64.txt" }
Write-Host "Installing $req ..."
& $venvPy -m pip install -r $req
& $venvPy -m pip install -e . --no-deps

# --- models -----------------------------------------------------------------------------
Write-Host "Downloading and verifying models (about 1 GB, once) ..."
& $venvPy -c "from avatarcam.config import Config; from avatarcam.models.registry import ensure_all, MODELS; ensure_all(list(MODELS), Config.load().models_dir())"

# --- shortcuts ----------------------------------------------------------------------------
$runBat = @"
@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m avatarcam.app %*
"@
Set-Content -Path (Join-Path $root "run.bat") -Value $runBat -Encoding ASCII
$benchBat = @"
@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m avatarcam.bench %*
pause
"@
Set-Content -Path (Join-Path $root "run_bench.bat") -Value $benchBat -Encoding ASCII
$prepBat = @"
@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m avatarcam.tools.prepare_avatar %*
"@
Set-Content -Path (Join-Path $root "prepare_avatar.bat") -Value $prepBat -Encoding ASCII

try {
    $shell = New-Object -ComObject WScript.Shell
    $desktop = [Environment]::GetFolderPath("Desktop")
    $lnk = $shell.CreateShortcut((Join-Path $desktop "AvatarCam.lnk"))
    $lnk.TargetPath = Join-Path $root "run.bat"
    $lnk.WorkingDirectory = $root
    $lnk.Save()
    Write-Host "Desktop shortcut created."
} catch { Write-Warning "Could not create a desktop shortcut: $_" }

& $venvPy -c "from avatarcam.runtime import describe; print(describe())"
Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. run_bench.bat                      (measure which accelerator works on this PC)"
Write-Host "  2. prepare_avatar.bat --out me.npz ref1.png ref2.png ..."
Write-Host "  3. run.bat --profile me.npz"
