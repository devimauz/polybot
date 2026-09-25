"""Wait until Polymarket CLOB+Gamma are reachable again, then launch the
TWAP bot (paper mode) as a detached background process and exit.

Usage: python launch_watchdog.py [--max-wait 900] [--interval 10]
"""
import sys, time, subprocess, argparse, os
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
# Load POLYSIM_API_KEY from the bot's .env (same file the bot reads).
try:
    for line in (HERE / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith("POLYSIM_API_KEY="):
            os.environ["POLYSIM_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
except OSError:
    pass
api_key = os.environ.get("POLYSIM_API_KEY", "")
# Paper mode only needs Polysimulator (market data + fills) and Binance
# (spot/TWAP oracle). Polymarket CLOB is optional (settlement fallback only,
# already guarded), so we gate on the two that MUST be reachable.
BINANCE = "https://api.binance.com/api/v3/ping"
POLYSIM_MKT = "https://api.polysimulator.com/v1/markets/updown"
BOT_CMD = [sys.executable, str(HERE / "twap_bot.py"), "--mode", "paper", "--stake", "10"]

ap = argparse.ArgumentParser()
ap.add_argument("--max-wait", type=float, default=900.0)
ap.add_argument("--interval", type=float, default=10.0)
args = ap.parse_args()

def reachable(url):
    try:
        r = requests.get(url, timeout=8)
        return r.status_code < 500
    except Exception:
        return False

def both_up():
    # Polysimulator /markets/updown is the discovery endpoint the bot actually
    # calls; it needs the API key header to return 200.
    try:
        r = requests.get(POLYSIM_MKT, headers={"X-API-Key": api_key}, timeout=8)
        return r.status_code == 200 and reachable(BINANCE)
    except Exception:
        return False

deadline = time.time() + args.max_wait
consec = 0
print(f"watchdog: waiting for polysim+binance (max {args.max_wait:.0f}s)", flush=True)
while time.time() < deadline:
    if both_up():
        consec += 1
        print(f"watchdog: both up (consec={consec})", flush=True)
        if consec >= 2:
            break
    else:
        consec = 0
        print(f"watchdog: not up yet, {args.interval}s", flush=True)
    time.sleep(args.interval)

if consec < 2:
    print("watchdog: TIMEOUT - polysim/binance unreachable, NOT launching", flush=True)
    sys.exit(3)

# Detach the bot so it survives this watchdog/terminal session exiting.
log = open(HERE / "bot_out.log", "ab")
flags = 0x00000208  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
p = subprocess.Popen(BOT_CMD, stdout=log, stderr=subprocess.STDOUT,
                     cwd=str(HERE), creationflags=flags, close_fds=True)
print(f"watchdog: LAUNCHED bot pid={p.pid} cmd={' '.join(BOT_CMD)}", flush=True)
time.sleep(3)
print(f"watchdog: bot still alive after 3s: {p.poll() is None}", flush=True)
log.close()
sys.exit(0)
