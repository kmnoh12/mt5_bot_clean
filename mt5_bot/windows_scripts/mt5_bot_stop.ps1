$ErrorActionPreference = 'SilentlyContinue'
. (Join-Path $PSScriptRoot 'mt5_bot_process_lib.ps1')

$Root = Get-Mt5BotRoot
$Runtime = Get-Mt5BotRuntimeDir
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
$now = [DateTimeOffset]::UtcNow.ToString('o')
@{
  paused=$true; manual_halt=$true; flatten_requested=$false; resume_requested=$false;
  manual_entry=$null; intentional_stop_requested=$true; intentional_stop_reason='quick_stop_with_terminal';
  intentional_stop_source='hermes'; intentional_stop_requested_at_utc=$now;
  updated_at_utc=$now; desired_state='STOP'; source='hermes'; reason='quick_stop_with_terminal'; orders_allowed=$false
} | ConvertTo-Json -Compress | Set-Content -Encoding UTF8 (Join-Path $Root 'runtime_control.json')
@{state='STOP'; updated_at_utc=$now; source='hermes'; reason='quick_stop_with_terminal'; metadata=@{orders_allowed=$false; kill_terminal=$true}} | ConvertTo-Json -Compress | Set-Content -Encoding UTF8 (Join-Path $Runtime 'desired_state.json')

$bot = @(Get-Mt5BotProcesses)
foreach($p in $bot){ Stop-Process -Id $p.ProcessId -Force }
$mt5 = @(Get-Mt5TerminalProcesses)
foreach($p in $mt5){ Stop-Process -Id $p.ProcessId -Force }
Start-Sleep -Milliseconds 300
$leftBot = @(Get-Mt5BotProcesses)
$leftMt5 = @(Get-Mt5TerminalProcesses)
"STOPPED bot_killed=$($bot.Count) mt5_killed=$($mt5.Count) bot_left=$($leftBot.Count) mt5_left=$($leftMt5.Count)"
