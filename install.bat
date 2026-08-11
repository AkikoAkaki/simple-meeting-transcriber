@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===================================================
echo  Simple Video Transcriber - Installation Script
echo ===================================================
echo.

:: 1. Check Python
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and check "Add Python to PATH" during installation.
    goto :FAIL
)

:: 2. Check ffmpeg
echo Checking ffmpeg installation...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ffmpeg was not found in your PATH.
    echo Transcription requires ffmpeg.
    echo Attempting to check if winget is available to install ffmpeg...
    winget --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] winget is available. Installing ffmpeg via winget...
        winget install Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements
        if !errorlevel! neq 0 (
            echo [ERROR] winget installation failed. Please install ffmpeg manually: https://ffmpeg.org/download.html
            goto :FAIL
        ) else (
            echo [INFO] winget finished installing ffmpeg.
        )
    ) else (
        echo [ERROR] winget is unavailable. Please install ffmpeg manually: https://ffmpeg.org/download.html
        goto :FAIL
    )
)
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] ffmpeg is still unavailable in this terminal.
    echo Close this window, open a new one, and run install.bat again.
    goto :FAIL
)
echo [OK] ffmpeg is installed.

:: 3. Create Virtual Environment
echo.
echo Creating Python virtual environment (.venv)...
if exist ".venv" (
    echo [INFO] .venv folder already exists, skipping creation.
) else (
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        goto :FAIL
    )
    echo [OK] Virtual environment created.
)

:: 4. Install Dependencies
echo.
echo Installing dependencies from requirements.txt...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    goto :FAIL
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    goto :FAIL
)
echo [OK] All dependencies installed.

echo.
echo ===================================================
echo  Installation Completed Successfully!
echo ===================================================
echo.
echo To start the application in the background:
echo   Double-click: start.bat
echo.
echo To configure Windows auto-start at login:
echo   Run: powershell -ExecutionPolicy Bypass -File setup_autostart.ps1
echo.
pause
exit /b 0

:FAIL
echo.
echo [FATAL] Installation failed. Please check the error messages above.
echo.
pause
exit /b 1
