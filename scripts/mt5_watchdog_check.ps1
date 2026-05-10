$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceDir = Split-Path -Parent $scriptDir
$wdPath = Join-Path $workspaceDir 'mt5_bot\watchdog.py'

$procs = Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -in @('python.exe', 'pythonw.exe')) -and $_.CommandLine -and ($_.CommandLine -like "*$wdPath*")
}

$procsCount = ($procs | Measure-Object).Count
"WATCHDOG_COUNT=$procsCount"
if ($procsCount -gt 0) {
  $procs | Select-Object ProcessId, Name, CommandLine | Format-List | Out-String -Width 400
}
