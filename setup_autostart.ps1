# simple-video-transcriber — setup_autostart.ps1
# Registers the tray controller as a Windows Task Scheduler task that starts at login.
# Run once with: powershell -ExecutionPolicy Bypass -File setup_autostart.ps1

$ErrorActionPreference = "Stop"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { python -c "import sys; print(sys.executable)" }
$pythonw   = $pythonExe -replace "python\.exe$", "pythonw.exe"
$script    = Join-Path $PSScriptRoot "tray_app.py"

if (-not (Test-Path $pythonw)) {
    Write-Warning "pythonw.exe not found at $pythonw — using python.exe (window will appear)"
    $pythonw = $pythonExe
}

$action   = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$script`""
$trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit 0 `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName    "SimpleVideoTranscriber" `
    -Action      $action `
    -Trigger     $trigger `
    -Settings    $settings `
    -Description "Watch the OBS recording folder and auto-transcribe new videos" `
    -Force | Out-Null

Write-Host ""
Write-Host "Task registered: SimpleVideoTranscriber"
Write-Host "The watcher will start automatically on next login."
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Start now:   Start-ScheduledTask  -TaskName SimpleVideoTranscriber"
Write-Host "  Stop:        Stop-ScheduledTask   -TaskName SimpleVideoTranscriber"
Write-Host "  Disable:     Disable-ScheduledTask -TaskName SimpleVideoTranscriber"
Write-Host "  Uninstall:   Unregister-ScheduledTask -TaskName SimpleVideoTranscriber"
Write-Host "  View log:    Get-Content `"$env:LOCALAPPDATA\SimpleVideoTranscriber\logs\app.log`" -Tail 30"
