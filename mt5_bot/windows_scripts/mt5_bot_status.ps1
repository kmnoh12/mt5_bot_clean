$ErrorActionPreference = 'SilentlyContinue'
. (Join-Path $PSScriptRoot 'mt5_bot_process_lib.ps1')

$bot = @(Get-Mt5BotProcesses)
$mt5 = @(Get-Mt5TerminalProcesses)
$owner = Get-Mt5RuntimeLockOwner
$heartbeatAge = Get-Mt5HeartbeatAgeSeconds
$lockPid = if ($owner) { $owner.Pid } else { '' }
$match = (($bot | ForEach-Object { $_.Mt5BotMatch }) -join ',')
$flags = Get-Mt5SafeConfigFlags
$state = Get-Mt5DesiredStateText

"STATUS bot_running=$($bot.Count -gt 0) bot_pids=$($bot.ProcessId -join ',') bot_match=$match lock_pid=$lockPid heartbeat_age_sec=$heartbeatAge mt5_running=$($mt5.Count -gt 0) mt5_pids=$($mt5.ProcessId -join ',') flags=$flags desired=$state"
