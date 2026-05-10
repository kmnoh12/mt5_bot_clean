@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Heartbeat + lock/process health check for MT5 bot.
set "BOT_DIR=%~dp0"
set "PYTHON_EXE=%PYTHON_EXE%"

if not defined PYTHON_EXE (
    for /f "delims=" %%I in ('where python 2^>nul') do (
        set "PYTHON_EXE=%%I"
        goto :python_found
    )
)
:python_found
if not defined PYTHON_EXE (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
)
if not exist "%PYTHON_EXE%" (
    echo [watchdog_healthcheck] Python executable not found: %PYTHON_EXE%
    exit /b 1
)

if not exist "%BOT_DIR%runtime" mkdir "%BOT_DIR%runtime"

set "LOG_FILE=%BOT_DIR%runtime\\watchdog_healthcheck.log"
if "%*"=="" (
    set "PY_ARGS=--notify"
) else (
    set "PY_ARGS=--notify %*"
)

pushd "%BOT_DIR%"
%PYTHON_EXE% watchdog_healthcheck.py --verbose %PY_ARGS% >> "%LOG_FILE%" 2>&1
set "HC_EXIT=%ERRORLEVEL%"
popd

if %HC_EXIT% GEQ 2 (
    echo [watchdog_healthcheck] BLOCK detected. return=%HC_EXIT% >> "%LOG_FILE%"
) else if %HC_EXIT% EQU 1 (
    echo [watchdog_healthcheck] WARN detected. return=%HC_EXIT% >> "%LOG_FILE%"
) else (
    echo [watchdog_healthcheck] OK. return=0 >> "%LOG_FILE%"
)

exit /b %HC_EXIT%
