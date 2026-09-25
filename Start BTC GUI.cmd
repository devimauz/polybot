@echo off
cd /d "%~dp0"
start "BTC Desk" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0bot_gui.py"
