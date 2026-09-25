@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0btc_analyzer.py"
if errorlevel 1 pause
