@echo off
rem Start the training console. Opens the browser automatically.
cd /d "%~dp0.."
start "" http://127.0.0.1:8770/
python console\server.py --port 8770
