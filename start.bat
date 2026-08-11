@echo off
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    "%~dp0.venv\Scripts\pythonw.exe" "%~dp0tray_app.py"
) else (
    pythonw "%~dp0tray_app.py"
)
