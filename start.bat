@echo off
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0tray_app.py"
) else if exist "%USERPROFILE%\.pyenv\pyenv-win\versions\3.12.6\pythonw.exe" (
    start "" "%USERPROFILE%\.pyenv\pyenv-win\versions\3.12.6\pythonw.exe" "%~dp0tray_app.py"
) else (
    start "" pythonw "%~dp0tray_app.py"
)
