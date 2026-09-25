@echo off
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "Price Log Viewer" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0price_log_viewer.py"
) else (
    python "%~dp0price_log_viewer.py"
)
