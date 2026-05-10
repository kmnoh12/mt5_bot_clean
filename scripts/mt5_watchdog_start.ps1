$ErrorActionPreference = 'Stop'
$pythonExe = (Get-Command python -ErrorAction Stop).Source

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceDir = Split-Path -Parent $scriptDir
$botDir = Join-Path $workspaceDir 'mt5_bot'
$wdPath = Join-Path $botDir 'watchdog.py'
$desiredStatePath = Join-Path $botDir 'runtime\desired_state.json'
$runtimeControlPath = Join-Path $botDir 'runtime_control.json'

function Test-StopLatched {
  if (Test-Path $desiredStatePath) {
    try {
      $ds = Get-Content $desiredStatePath -Raw | ConvertFrom-Json
      if (($ds.state + '') -eq 'STOP') { return $true }
    } catch {}
  }
  if (Test-Path $runtimeControlPath) {
    try {
      $rc = Get-Content $runtimeControlPath -Raw | ConvertFrom-Json
      if ($rc.manual_halt -or $rc.intentional_stop_requested) { return $true }
    } catch {}
  }
  return $false
}

if (Test-StopLatched) {
  "WATCHDOG_SUPPRESSED_STOP_LATCH=1"
  exit 0
}

# Kill duplicates if any
$procs = Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -in @('python.exe', 'pythonw.exe')) -and $_.CommandLine -and ($_.CommandLine -like "*$wdPath*")
}
if (($procs | Measure-Object).Count -gt 1) {
  $procs | Sort-Object ProcessId | Select-Object -Skip 1 | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
}

# Start if not running
$procs = Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -in @('python.exe', 'pythonw.exe')) -and $_.CommandLine -and ($_.CommandLine -like "*$wdPath*")
}
if (($procs | Measure-Object).Count -eq 0) {
  Start-Process -FilePath $pythonExe -ArgumentList @($wdPath) -WindowStyle Hidden -WorkingDirectory $botDir
  Start-Sleep -Milliseconds 1200
}

$procs = Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -in @('python.exe', 'pythonw.exe')) -and $_.CommandLine -and ($_.CommandLine -like "*$wdPath*")
}
"WATCHDOG_COUNT_AFTER=" + (($procs | Measure-Object).Count)
$procs | Select-Object ProcessId, Name, CommandLine | Format-List | Out-String -Width 400
