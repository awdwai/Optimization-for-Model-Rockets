@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python was not found on PATH.
  echo Install Python 3 from https://www.python.org/downloads/ and check "Add Python to PATH".
  pause
  exit /b 1
)

echo Installing dependencies from requirements.txt ...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: pip install failed.
  pause
  exit /b 1
)

echo.
echo Starting Streamlit demo UI (ui/app.py) ...
echo Close the browser tab or press Ctrl+C in this window to stop.
echo.
python -m streamlit run ui/app.py
if errorlevel 1 (
  echo.
  echo ERROR: Streamlit failed to start. Is streamlit installed?
  pause
  exit /b 1
)

endlocal
