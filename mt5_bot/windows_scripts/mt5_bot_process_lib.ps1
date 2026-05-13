$ErrorActionPreference = 'SilentlyContinue'

function Get-Mt5BotRoot {
  if ($env:LOCALAPPDATA) {
    return (Join-Path $env:LOCALAPPDATA 'Hermes\internal\mt5_bot_runtime')
  }
  return 'C:\Users\kmnoh\AppData\Local\Hermes\internal\mt5_bot_runtime'
}

function Get-Mt5BotRuntimeDir {
  return (Join-Path (Get-Mt5BotRoot) 'runtime')
}

function Read-Mt5TextFileShared {
  param([string]$Path)
  if (!(Test-Path $Path)) { return $null }
  try {
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
      $reader = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8, $true)
      try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
    } finally { $fs.Dispose() }
  } catch {
    return $null
  }
}

function Get-Mt5RuntimeLockOwner {
  $lockPath = Join-Path (Get-Mt5BotRoot) 'runtime.lock'
  $raw = Read-Mt5TextFileShared -Path $lockPath
  if (!$raw) { return $null }
  $raw = $raw.Trim()
  if (!$raw) { return $null }
  $pidValue = 0
  if ([int]::TryParse($raw, [ref]$pidValue)) {
    if ($pidValue -gt 0) {
      return [pscustomobject]@{ Pid = $pidValue; StartTime = $null; Source = 'runtime.lock' }
    }
    return $null
  }
  try {
    $payload = $raw | ConvertFrom-Json
    $pidValue = [int]($payload.pid)
    if ($pidValue -gt 0) {
      return [pscustomobject]@{ Pid = $pidValue; StartTime = $payload.start_time; Source = 'runtime.lock' }
    }
  } catch {}
  return $null
}

function Get-Mt5HeartbeatAgeSeconds {
  $heartbeat = Join-Path (Get-Mt5BotRuntimeDir) 'heartbeat.json'
  try {
    $raw = Read-Mt5TextFileShared -Path $heartbeat
    if (!$raw) { return $null }
    $payload = $raw | ConvertFrom-Json
    $ts = [double]($payload.ts)
    if ($ts -le 0) { return $null }
    return [math]::Round(([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $ts), 1)
  } catch {
    return $null
  }
}

function Test-Mt5HeartbeatFresh {
  param([double]$MaxAgeSeconds = 60.0)
  $age = Get-Mt5HeartbeatAgeSeconds
  return ($null -ne $age -and $age -ge -5 -and $age -le $MaxAgeSeconds)
}

function Get-Mt5ProcessById {
  param([int]$ProcessId)
  if ($ProcessId -le 0) { return $null }
  return Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId"
}

function Test-Mt5RunnerCommandLine {
  param(
    [string]$Name,
    [string]$CommandLine
  )
  if (!$CommandLine) { return $false }
  $cmd = $CommandLine.ToLowerInvariant()
  $root = (Get-Mt5BotRoot).ToLowerInvariant()
  $runtimeName = 'mt5_bot_runtime'
  $hasRunner = $cmd.Contains('runner.py')
  if (!$hasRunner) { return $false }
  $hasRuntime = ($cmd.Contains($runtimeName) -or $cmd.Contains($root))
  $hasConfigYaml = ($cmd.Contains('--config') -and $cmd.Contains('config.yaml'))
  $hasConfigLive = ($hasConfigYaml -and $cmd.Contains('--mode') -and $cmd.Contains('live'))
  $hostName = ('' + $Name).ToLowerInvariant()
  $knownHost = ($hostName -eq 'cmd.exe' -or $hostName -eq 'python.exe' -or $hostName -eq 'pythonw.exe' -or $hostName -eq 'py.exe')
  return ($hasRuntime -or ($knownHost -and ($hasConfigLive -or $hasConfigYaml)))
}

function Get-Mt5BotProcesses {
  $matches = @()
  $all = @(Get-CimInstance Win32_Process)
  foreach ($p in $all) {
    if (Test-Mt5RunnerCommandLine -Name $p.Name -CommandLine $p.CommandLine) {
      $p | Add-Member -NotePropertyName Mt5BotMatch -NotePropertyValue 'cmdline' -Force
      $matches += $p
    }
  }

  $owner = Get-Mt5RuntimeLockOwner
  if ($owner -and (Test-Mt5HeartbeatFresh -MaxAgeSeconds 60.0)) {
    $locked = Get-Mt5ProcessById -ProcessId ([int]$owner.Pid)
    if ($locked) {
      $already = @($matches | Where-Object { [int]$_.ProcessId -eq [int]$locked.ProcessId }).Count -gt 0
      $hostName = ('' + $locked.Name).ToLowerInvariant()
      $hostLooksPlausible = ($hostName -eq 'cmd.exe' -or $hostName -eq 'python.exe' -or $hostName -eq 'pythonw.exe' -or $hostName -eq 'py.exe')
      $cmdLooksPlausible = (Test-Mt5RunnerCommandLine -Name $locked.Name -CommandLine $locked.CommandLine)
      if (!$already -and ($hostLooksPlausible -or $cmdLooksPlausible)) {
        $locked | Add-Member -NotePropertyName Mt5BotMatch -NotePropertyValue 'lock_heartbeat' -Force
        $matches += $locked
      }
    }
  }

  return @($matches | Sort-Object ProcessId -Unique)
}

function Get-Mt5TerminalProcesses {
  return @(Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'terminal64.exe' -or $_.Name -ieq 'terminal.exe' })
}

function Get-Mt5SafeConfigFlags {
  $cfg = Join-Path (Get-Mt5BotRoot) 'config.yaml'
  if (!(Test-Path $cfg)) { return '' }
  $lines = [System.IO.File]::ReadAllLines($cfg)
  return (($lines | Where-Object { $_ -match '^\s*(dry_run|live_trading_enabled|close_positions_on_exit|variant|symbol):' }) -join '; ')
}

function Get-Mt5DesiredStateText {
  $desired = Join-Path (Get-Mt5BotRuntimeDir) 'desired_state.json'
  if (Test-Path $desired) {
    return ([System.IO.File]::ReadAllText($desired) -replace "\r|\n",'')
  }
  return ''
}

function Format-Mt5ShortCommandLine {
  param([string]$CommandLine, [int]$MaxLength = 280)
  if (!$CommandLine) { return '' }
  $oneLine = ($CommandLine -replace "\r|\n", ' ')
  if ($oneLine.Length -le $MaxLength) { return $oneLine }
  return ($oneLine.Substring(0, $MaxLength) + '...')
}
