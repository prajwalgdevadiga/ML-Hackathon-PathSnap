@echo off
setlocal
cd /d "%~dp0"
set "MODE=%~1"
if not defined MODE set "MODE=student-test"
set "HOST=%~2"
if not defined HOST set "HOST=0.0.0.0"
set "PORT=%~3"

if /i "%MODE%"=="student-test" (
    set "MODE=student-test"
    set "DATA=inputs\Student folder.zip"
    set "PARTITION=student_test"
    if not defined PORT set "PORT=8765"
) else if /i "%MODE%"=="evaluator" (
    set "MODE=evaluator"
    set "DATA=inputs\evaluator_test.zip"
    set "PARTITION=evaluator_test"
    if not defined PORT set "PORT=8766"
) else (
    echo Usage: start-server.cmd [student-test or evaluator] [bind address] [port]
    exit /b 2
)

if not exist ".venv\Scripts\python.exe" (
    echo First run: py -3.13 -m venv .venv
    echo Then run: .venv\Scripts\python.exe -m pip install -r stage-6\requirements.txt
    if "%~1"=="" pause
    exit /b 1
)
if not exist "%DATA%" (
    echo Dataset missing. Put the archive at "%DATA%".
    if "%~1"=="" pause
    exit /b 1
)
set "PYTHONDONTWRITEBYTECODE=1"
set "LOKY_MAX_CPU_COUNT=3"
".venv\Scripts\python.exe" -u "stage-6\server\service.py" --data "%DATA%" --partition "%PARTITION%" --mode "%MODE%" --host "%HOST%" --port "%PORT%" --log "receiving.log"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" if "%~1"=="" pause
exit /b %EXIT_CODE%
