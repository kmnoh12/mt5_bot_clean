$ErrorActionPreference = 'Stop'
$procs = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and ($_.CommandLine -match 'watchdog\.py')
}
$cnt = ($procs | Measure-Object).Count
"WATCHDOG_MATCH_COUNT=$cnt"
if($cnt -gt 0){
  $procs | Select-Object ProcessId, Name, CommandLine | Format-List | Out-String -Width 400
}
