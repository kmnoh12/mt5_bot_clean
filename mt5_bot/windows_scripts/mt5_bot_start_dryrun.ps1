$ErrorActionPreference = 'SilentlyContinue'
. (Join-Path $PSScriptRoot 'mt5_bot_process_lib.ps1')

$Root = Get-Mt5BotRoot
$Runtime = Get-Mt5BotRuntimeDir
$Py = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
$Mt5 = 'C:\Program Files\XM Global MT5\terminal64.exe'
if (!(Test-Path $Mt5)) { $Mt5 = 'C:\Program Files\MetaTrader 5\terminal64.exe' }
$Log = Join-Path $Runtime 'runner_stdout.log'
$Err = Join-Path $Runtime 'runner_stderr.log'
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
function BotProcs { @(Get-Mt5BotProcesses) }
function Mt5Procs { @(Get-Mt5TerminalProcesses) }
function Get-ConfigBool {
  param([string]$Section, [string]$Key, [bool]$Default = $false)
  $cfg = Join-Path $Root 'config.yaml'
  if (!(Test-Path $cfg)) { return $Default }
  $current = ''
  foreach ($line in [System.IO.File]::ReadAllLines($cfg)) {
    if ($line -match '^([A-Za-z_][\w-]*):\s*$') { $current = $Matches[1]; continue }
    if ($current -eq $Section -and $line -match ('^\s+' + [regex]::Escape($Key) + ':\s*(true|false)\s*$')) {
      return ($Matches[1].ToLowerInvariant() -eq 'true')
    }
  }
  return $Default
}

$existing = @(BotProcs)
if ($existing.Count -gt 0) { "ALREADY_RUNNING bot_pids=$($existing.ProcessId -join ',') mt5_running=$((Mt5Procs).Count -gt 0)"; exit 0 }
$generalDryRun = Get-ConfigBool -Section 'general' -Key 'dry_run' -Default $true
$executionDryRun = Get-ConfigBool -Section 'execution' -Key 'dry_run' -Default $true
$liveTradingEnabled = Get-ConfigBool -Section 'execution' -Key 'live_trading_enabled' -Default $false
$ordersAllowed = ((-not $generalDryRun) -and (-not $executionDryRun) -and $liveTradingEnabled)
$reason = if ($ordersAllowed) { 'quick_start_live' } else { 'quick_start_paper_forward' }
$now = [DateTimeOffset]::UtcNow.ToString('o')
@{paused=$false; manual_halt=$false; flatten_requested=$false; resume_requested=$true; manual_entry=$null; intentional_stop_requested=$false; updated_at_utc=$now; desired_state='RUN'; source='hermes'; reason=$reason; orders_allowed=$ordersAllowed} | ConvertTo-Json -Compress | Set-Content -Encoding UTF8 (Join-Path $Root 'runtime_control.json')
@{state='RUN'; updated_at_utc=$now; source='hermes'; reason=$reason; metadata=@{orders_allowed=$ordersAllowed; dry_run=($generalDryRun -or $executionDryRun); live_trading_enabled=$liveTradingEnabled; start_terminal=$true}} | ConvertTo-Json -Compress | Set-Content -Encoding UTF8 (Join-Path $Runtime 'desired_state.json')
if ((Mt5Procs).Count -eq 0 -and (Test-Path $Mt5)) { Start-Process -FilePath $Mt5 | Out-Null }
Remove-Item -Force $Log,$Err -ErrorAction SilentlyContinue
$liveEnv = if ($ordersAllowed) { 'set MT5_ALLOW_LIVE_TRADING=YES_I_ACCEPT_RISK&& ' } else { '' }
$cmd = "cd /d `"$Root`" && $liveEnv`"$Py`" -u runner.py --config config.yaml --mode live > runtime\runner_stdout.log 2> runtime\runner_stderr.log"
Start-Process -FilePath 'cmd.exe' -ArgumentList @('/d','/c',$cmd) -WindowStyle Hidden | Out-Null
Start-Sleep -Milliseconds 250
$label = if ($ordersAllowed) { 'START_SENT_LIVE' } else { 'START_SENT_PAPER' }
"$label bot_running=$((BotProcs).Count -gt 0) mt5_running=$((Mt5Procs).Count -gt 0) dry_run=$($generalDryRun -or $executionDryRun) live_orders=$ordersAllowed close_positions_on_exit=false"
