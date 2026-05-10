@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "TASK_NAME=MT5_WatchdogHealthCheck"
set "SCRIPT_DIR=%~dp0"
set "TASK_CMD=%SCRIPT_DIR%watchdog_healthcheck.cmd"

if "%~1"=="/delete" (
    schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1
    if errorlevel 1 (
        echo [watchdog_healthcheck] Task delete failed or task does not exist.
        exit /b 1
    )
    echo [watchdog_healthcheck] Task deleted: %TASK_NAME%
    exit /b 0
)

if not exist "%TASK_CMD%" (
    echo [watchdog_healthcheck] Missing task command: %TASK_CMD%
    exit /b 1
)

schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if not errorlevel 1 (
    schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1
)

schtasks /create /tn "%TASK_NAME%" /tr "\"%TASK_CMD%\"" /sc minute /mo 1 /f /ru "NT AUTHORITY\SYSTEM" /rl HIGHEST >nul 2>&1
if errorlevel 1 (
    schtasks /create /tn "%TASK_NAME%" /tr "\"%TASK_CMD%\"" /sc minute /mo 1 /f /ru "%USERNAME%" >nul 2>&1
    if errorlevel 1 (
        echo [watchdog_healthcheck] Register failed as SYSTEM and as current user. run from elevated shell or check rights.
        exit /b 1
    )
)

echo [watchdog_healthcheck] Scheduled: %TASK_NAME% (every 1 minute)
echo To remove: %~f0 /delete
exit /b 0
