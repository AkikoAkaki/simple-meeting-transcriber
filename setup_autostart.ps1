# meeting-transcriber — setup_autostart.ps1
# Registers watch.py as a Windows Task Scheduler task that starts at login.
# Run once with: powershell -ExecutionPolicy Bypass -File setup_autostart.ps1

$ErrorActionPreference = "Stop"

$pythonExe = python -c "import sys; print(sys.executable)"
$pythonw   = $pythonExe -replace "python\.exe$", "pythonw.exe"
$script    = Join-Path $PSScriptRoot "watch.py"

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
    -TaskName    "MeetingTranscriber-Watcher" `
    -Action      $action `
    -Trigger     $trigger `
    -Settings    $settings `
    -Description "Auto-transcribe new videos dropped into the inbox folder" `
    -Force | Out-Null

Write-Host ""
Write-Host "Task registered: MeetingTranscriber-Watcher"
Write-Host "The watcher will start automatically on next login."
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Start now:   Start-ScheduledTask  -TaskName MeetingTranscriber-Watcher"
Write-Host "  Stop:        Stop-ScheduledTask   -TaskName MeetingTranscriber-Watcher"
Write-Host "  Disable:     Disable-ScheduledTask -TaskName MeetingTranscriber-Watcher"
Write-Host "  Uninstall:   Unregister-ScheduledTask -TaskName MeetingTranscriber-Watcher"
Write-Host "  View log:    Get-Content watch.log -Tail 30"
