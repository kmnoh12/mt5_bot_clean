import subprocess
import os
import sys
import json
from pathlib import Path


def is_stop_latched() -> bool:
    base = Path(__file__).resolve().parent / "mt5_bot"
    desired_path = base / "runtime" / "desired_state.json"
    control_path = base / "runtime_control.json"

    try:
        if desired_path.exists():
            desired = json.loads(desired_path.read_text(encoding="utf-8"))
            if str(desired.get("state", "")).strip().upper() == "STOP":
                return True
    except Exception:
        pass

    try:
        if control_path.exists():
            ctrl = json.loads(control_path.read_text(encoding="utf-8"))
            if bool(ctrl.get("manual_halt")) or bool(ctrl.get("intentional_stop_requested")):
                return True
    except Exception:
        pass

    return False

def get_watchdog_processes():
    try:
        # Use tasklist/wmic or psutil if available, but let's stick to standard wmic for zero-dep
        cmd = 'wmic process where "commandline like \'%%watchdog.py%%\'" get commandline,processid'
        output = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
        lines = [line.strip() for line in output.split('\n') if line.strip() and 'processid' not in line.lower()]
        
        processes = []
        for line in lines:
            # Line looks like: python mt5_bot\watchdog.py   1234
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                cmdline, pid = parts
                # Exclude the wmic command itself and this script if it somehow matches
                if 'wmic' not in cmdline.lower() and str(os.getpid()) != pid:
                    processes.append({'cmdline': cmdline, 'pid': pid})
        return processes
    except Exception as e:
        print(f"Error finding processes: {e}")
        return []

procs = get_watchdog_processes()
print(f"Found {len(procs)} watchdog processes.")

if is_stop_latched():
    print("STOP_LATCHED: watchdog auto-start suppressed.")
elif len(procs) == 0:
    print("Starting watchdog...")
    # Start as a detached background process
    subprocess.Popen([sys.executable, 'mt5_bot\\watchdog.py'], 
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                     cwd=os.getcwd())
    print("Watchdog started.")
elif len(procs) > 1:
    print("Multiple watchdogs found. Cleaning up...")
    # Keep the first one, kill others
    for p in procs[1:]:
        print(f"Killing duplicate PID {p['pid']}")
        subprocess.run(['taskkill', '/F', '/PID', p['pid']], capture_output=True)
    print("Cleanup complete.")
else:
    print("Watchdog is running correctly.")
