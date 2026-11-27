@echo off
REM Run the BTC 5-minute endgame-favourite bot in PAPER mode (polysimulator).
REM Strategy validated by btc5m_endgame_backtest.py -- see ENDGAME_EDGE.md
setlocal
cd /d "%~dp0"

set "PY=%LOCALAPPDATA%\hermes\installs\0d1790b230609f8c\environments\bb25ab9dd7a34e3599ece40a7efc18d9\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Starting BTC 5m endgame-favourite bot (PAPER / polysimulator)...
"%PY%" btc5m_endgame_bot.py --paper %*
endlocal
