# simple-video-transcriber — setup_autostart.ps1
# Registers the tray controller in Windows Startup (HKCU Run registry key).
# Run once with: powershell -ExecutionPolicy Bypass -File setup_autostart.ps1

param(
    [switch]$Uninstall,
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"

$regPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$regName = "SimpleVideoTranscriber"

# 1. Always unregister any legacy Scheduled Task if present
try {
    $existingTask = Get-ScheduledTask -TaskName $regName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Unregister-ScheduledTask -TaskName $regName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "[OK] Cleaned up legacy Windows Scheduled Task '$regName'."
    }
} catch {}

if ($Uninstall) {
    Remove-ItemProperty -Path $regPath -Name $regName -ErrorAction SilentlyContinue
    Write-Host "[OK] Autostart registry entry removed. SimpleVideoTranscriber will not start at login."
    exit 0
}

# 2. Locate pythonw
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { python -c "import sys; print(sys.executable)" }
$pythonw   = $pythonExe -replace "python\.exe$", "pythonw.exe"
$script    = Join-Path $PSScriptRoot "tray_app.py"

if (-not (Test-Path $pythonw)) {
    Write-Warning "pythonw.exe not found at $pythonw — using python.exe"
    $pythonw = $pythonExe
}

# 3. Register in HKCU Run key
$cmd = "`"$pythonw`" `"$script`""
Set-ItemProperty -Path $regPath -Name $regName -Value $cmd

Write-Host ""
Write-Host "==================================================="
Write-Host " Autostart Registered via Windows Run Key (HKCU)"
Write-Host "==================================================="
Write-Host "Command: $cmd"
Write-Host ""
Write-Host "[OK] SimpleVideoTranscriber will automatically start when you log in to Windows."
Write-Host "     (Runs via Explorer desktop session, immune to laptop battery policies)"
Write-Host ""

if ($StartNow) {
    Start-Process -FilePath $pythonw -ArgumentList "`"$script`""
    Write-Host "[OK] Launched SimpleVideoTranscriber in the background."
} else {
    Write-Host "To launch right now, run:"
    Write-Host "  Start-Process `"$pythonw`" -ArgumentList `"`"$script`"`""
}
Write-Host "To uninstall autostart, run:"
Write-Host "  powershell -ExecutionPolicy Bypass -File setup_autostart.ps1 -Uninstall"
Write-Host ""
