$ErrorActionPreference = 'SilentlyContinue'
. (Join-Path $PSScriptRoot 'mt5_bot_process_lib.ps1')

$bot = @(Get-Mt5BotProcesses)
$mt5 = @(Get-Mt5TerminalProcesses)
$owner = Get-Mt5RuntimeLockOwner
$heartbeatAge = Get-Mt5HeartbeatAgeSeconds
$lockPid = if ($owner) { $owner.Pid } else { '' }
"PROBE bot_count=$($bot.Count) bot_pids=$($bot.ProcessId -join ',') lock_pid=$lockPid heartbeat_age_sec=$heartbeatAge mt5_count=$($mt5.Count) mt5_pids=$($mt5.ProcessId -join ',')"
foreach ($p in $bot) {
  $cmd = Format-Mt5ShortCommandLine -CommandLine $p.CommandLine
  "BOT pid=$($p.ProcessId) parent=$($p.ParentProcessId) name=$($p.Name) match=$($p.Mt5BotMatch) cmd=$cmd"
}
