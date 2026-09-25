"""Sleep until a target local time, then print the TWAP bot status report."""
import subprocess, sys, time
from datetime import datetime

target = datetime.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2026-09-21T17:15:00")
wait_s = (target - datetime.now()).total_seconds()
if wait_s > 0:
    print(f"sleeping {wait_s:.0f}s until {target}", flush=True)
    time.sleep(wait_s)
print("== FIRING TWAP BOT STATUS CHECK ==", flush=True)
subprocess.run([sys.executable, r"C:\Users\user\Desktop\bot\check_status.py"])
