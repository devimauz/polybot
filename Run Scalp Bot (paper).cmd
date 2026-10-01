@echo off
cd /d "%~dp0"
set "PY=C:\Users\user\AppData\Local\hermes\installs\0d1790b230609f8c\environments\bb25ab9dd7a34e3599ece40a7efc18d9\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Python launcher not found: %PY%
    echo Fallback: try your default python.
    python scalp_bot.py --paper
    pause
    exit /b
)
"%PY%" scalp_bot.py --paper
pause
