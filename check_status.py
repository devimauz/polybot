"""One-shot status check for the TWAP paper bot. Prints a concise report."""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
log = ROOT / "twap_bot.log"
trades = ROOT / "price_logs/twap/trades.jsonl"

# 1. Bot alive?
out = subprocess.run(["powershell","-NoProfile","-Command",
    "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object "
    "{ $_.CommandLine -match 'twap_bot.py' } | ForEach-Object { $_.ProcessId })"],
    capture_output=True, text=True).stdout.strip()
pids = [p for p in out.split() if p.strip().isdigit()]
alive = bool(pids)

# 2. Account balance
def balance():
    try:
        import httpx
        from dotenv import load_dotenv
        load_dotenv(ROOT/".env")
        k = os.environ["POLYSIM_API_KEY"].strip()
        d = httpx.get("https://api.polysimulator.com/v1/account/balance",
                      headers={"X-API-Key":k}, timeout=10).json()
        return d
    except Exception as e:
        return {"error": str(e)}

acc = balance()

# 3. Trades: fills + settles
fills, settles = [], []
if trades.exists():
    for line in trades.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except ValueError: continue
        if d.get("event") == "settle":
            settles.append(d)
        elif "slug" in d and "cost" in d:
            fills.append(d)

settled_slugs = {s["slug"] for s in settles}
won = [s for s in settles if s.get("won")]
lost = [s for s in settles if not s.get("won")]
pnl = sum(s.get("pnl", 0.0) for s in settles)
n_settled = len(settles)
win_rate = (len(won)/n_settled*100) if n_settled else 0.0
session_pnl = settles[-1].get("session_pnl") if settles else 0.0
open_pos = [f for f in fills if f["slug"] not in settled_slugs]

# 4. Recent PAPER BUY lines from log (last 15)
recent = []
if log.exists():
    for line in log.read_text(encoding="utf-8").splitlines():
        if "PAPER BUY" in line or "SETTLE" in line or "RECOVER" in line:
            recent.append(line)
recent = recent[-15:]

print("=" * 54)
print("TWAP PAPER BOT STATUS")
print("=" * 54)
print(f"BOT ALIVE : {'YES (pids ' + str(pids) + ')' if alive else 'NO  <-- check twap_bot.log tail for crash'}")
print(f"          :")
# account
if "error" in acc and "balance" not in acc:
    print(f"BALANCE   : ERR {acc.get('error')}")
else:
    print(f"BALANCE   : ${acc.get('balance')}  (total_value ${acc.get('total_value')})")
    print(f"          : start $10000  realized_pnl ${acc.get('unrealized_pnl')}")
print("-" * 54)
print(f"TRADES    : {n_settled} settled  |  {len(open_pos)} open")
print(f"          : {len(won)}W / {len(lost)}L  win_rate {win_rate:.0f}%")
print(f"PAPER PNL : ${pnl:+.2f}   session_pnl ${session_pnl:+.2f}")
# per-trade
if settles:
    print("  last 10 settles:")
    for s in settles[-10:]:
        print(f"    {s['slug'][:34]:34} {s['side']:5} {'WIN' if s['won'] else 'LOSE':4} pnl {s['pnl']:+7.2f}  sess {s.get('session_pnl',0):+6.2f}")
if open_pos:
    print(f"  open positions ({len(open_pos)}):")
    for f in open_pos[-8:]:
        print(f"    {f['slug'][:34]:34} {f['side']:5} cost {f['cost']:.2f} model {f.get('model_p')}")
print("-" * 54)
print("RECENT LOG LINES:")
for r in recent:
    print("  " + r[:110])
print("=" * 54)
# verdict
if not alive:
    print("VERDICT   : BOT IS DEAD - inspect twap_bot.log and restart.")
elif n_settled < 3:
    print(f"VERDICT   : Only {n_settled} trades settled - sample too small yet. Keep running.")
elif pnl > 0:
    print(f"VERDICT   : PROFITABLE so far (${pnl:+.2f} on {n_settled} trades, {win_rate:.0f}% win).")
else:
    print(f"VERDICT   : NOT profitable (${pnl:+.2f} on {n_settled} trades, {win_rate:.0f}% win). Consider pausing.")
