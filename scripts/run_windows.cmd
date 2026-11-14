@echo off
setlocal
cd /d "%~dp0.."
py -3.12 -c "import sys" >nul 2>&1
if errorlevel 1 goto try311
set "PROJECT_PY=py -3.12"
goto setup
:try311
py -3.11 -c "import sys" >nul 2>&1
if errorlevel 1 goto no_python
set "PROJECT_PY=py -3.11"
:setup
if exist ".venv\Scripts\python.exe" goto install
%PROJECT_PY% -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m unittest discover -s scripts/tests -v
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m src.run
if errorlevel 1 goto failed
echo Completed. Open the reports and results folders.
pause
exit /b 0
:no_python
echo Install Python 3.11 or 3.12 with the Windows Python launcher first.
pause
exit /b 1
:failed
echo The command failed. Read the error above; later steps have stopped.
pause
exit /b 1
