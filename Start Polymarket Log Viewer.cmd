@echo off
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "Polymarket Log Viewer" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0polymarket_log_viewer.py" "%~dp0price_logs\polymarket"
) else (
    python "%~dp0polymarket_log_viewer.py" "%~dp0price_logs\polymarket"
)
