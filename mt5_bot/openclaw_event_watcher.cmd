@echo off
setlocal EnableExtensions EnableDelayedExpansion

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
    echo [openclaw_event_watcher] Python executable not found: %PYTHON_EXE%
    exit /b 1
)

if not exist "%BOT_DIR%runtime" mkdir "%BOT_DIR%runtime"
set "LOG_FILE=%BOT_DIR%runtime\\openclaw_event_watcher.log"

pushd "%BOT_DIR%"
%PYTHON_EXE% openclaw_event_watcher.py %* >> "%LOG_FILE%" 2>&1
set "BRIDGE_EXIT=%ERRORLEVEL%"
popd

exit /b %BRIDGE_EXIT%
