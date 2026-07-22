@echo off
echo Starting Aplo Backend...
start "Aplo Backend" cmd /c "python app.py"

cd /d "%~dp0aplo-dashboard"
echo Starting Aplo Frontend...
start "Aplo Frontend" cmd /c "npm run dev"

echo Both services started. Close the windows to stop.
