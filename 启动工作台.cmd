@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Core environment is missing. Open this directory in PowerShell and run:
    echo python -m venv .venv
    echo .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
start "" /b ".venv\Scripts\pythonw.exe" -m sim_app.workbench
