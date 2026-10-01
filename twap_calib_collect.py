"""Calibration data collector for the TWAP model.

Runs forward, sampling every second per asset. At decision times t* in the
window it computes the model's p_up (Binance path + sigma) and records the
market's bid/ask/mid for both outcomes. When a window resolves (live_price
converges to 0/1), the outcome is recorded. Output: price_logs/twap/calib.jsonl
One row per (asset, window, t*) plus one row per (asset, window) outcome.

The analysis script (twap_calib_analyze.py) then compares:
  - model Brier vs market-mid Brier (is the model better than the market?)
  - EV of a taker strategy (buy side with best model-vs-ask edge, net of fee)
"""
import json, math, os, statistics, time
from datetime import datetime, timezone
from pathlib import Path
import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
API = "https://api.polysimulator.com/v1"
BINANCE = "https://api.binance.com"
ASSETS = ["btc", "sol", "doge", "eth", "bnb"]
SYMBOL = {a: a.upper() + "USDT" for a in ASSETS}
WINDOW = 300
T_STAR = (90.0, 150.0, 210.0)   # decision times to evaluate the model
KEY = os.environ["POLYSIM_API_KEY"].strip()
sim = httpx.Client(timeout=10, headers={"X-API-Key": KEY})
bnc = httpx.Client(timeout=8)

OUT = ROOT / "price_logs" / "twap" / "calib.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
jf = open(OUT, "a", encoding="utf-8")

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

class Win:
    __slots__ = ("asset","start","ref","spot","path","sigma","done","winner")
    def __init__(self, asset, start, ref):
        self.asset = asset; self.start = start; self.ref = ref
        self.spot = None; self.path = []
        self.sigma = None; self.done = False; self.winner = None

wins = {}
sigcache = {}
fired = set()   # (slug, t*) already sampled

def dec_list(v):
    return json.loads(v) if isinstance(v, str) else v

def get_market(asset):
    r = sim.get(API + "/markets/updown", params={"asset": asset.upper(), "interval": "5M"}, timeout=8)
    j = r.json()
    rows = j if isinstance(j, list) else j.get("markets", [])
    start = int(time.time() // WINDOW * WINDOW)
    slug = f"{asset}-updown-5m-{start}"
    for m in rows:
        if m.get("slug") != slug:
            continue
        nm = (m.get("markets") or [{}])[0]
        oc = dec_list(nm.get("outcomes")); tokens = dec_list(nm.get("token_ids"))
        lv = m.get("live_price") or {}
        return {
            "slug": slug, "start": start,
            "cid": m.get("condition_id"),
            "thr": m.get("group_item_threshold"),
            "tokens": dict(zip(oc, tokens)) if oc and tokens else {},
            "live_buy": lv.get("buy"), "live_sell": lv.get("sell"),
            "active": m.get("active"),
        }
    return None

def book(token):
    r = sim.get(API + "/book", params={"token_id": token}, timeout=8)
    if r.status_code != 200:
        return None, None
    b = r.json()
    bids = [float(x["price"]) for x in b.get("bids", []) if float(x["size"]) > 0]
    asks = [float(x["price"]) for x in b.get("asks", []) if float(x["size"]) > 0]
    return (max(bids) if bids else None), (min(asks) if asks else None)

def sigma1m(asset):
    try:
        r = bnc.get(BINANCE + "/api/v3/klines", params={"symbol": SYMBOL[asset], "interval": "1m", "limit": 30}, timeout=6)
        closes = [float(k[4]) for k in r.json()]
        rets = [(b - a) / a for a, b in zip(closes, closes[1:])]
        return statistics.pstdev(rets) if len(rets) > 1 else None
    except httpx.HTTPError:
        return None

def model_p(w, now):
    t = now - w.start
    if w.ref is None or w.spot is None or not w.sigma or w.sigma <= 0:
        return None, None
    r = w.start + WINDOW - now
    T = float(WINDOW)
    if w.path:
        integ = 0.0; prev = 0.0
        for off, p in w.path:
            if off > t: break
            integ += (off - prev) * p; prev = off
        integ += (t - prev) * w.spot
        a_int = integ / T
    else:
        a_int = w.spot * t / T
    e_final = a_int + r / T * w.spot
    sig_s = w.sigma / 60.0
    var = w.spot ** 2 * sig_s ** 2 * r ** 3 / (3 * T * T)
    sd = math.sqrt(max(var, 1e-18))
    sd_raw = sd   # pre-floor, for offline recalibration
    # Floor: residual Binance-vs-Chainlink basis + vol estimation + discretisation.
    # At t=90s (r=210) this caps z so p_up can't saturate before we trust it.
    sd = max(sd, 0.0012 * w.spot)
    z = (e_final - w.ref) / sd if sd > 0 else 0.0
    return norm_cdf(z), {"t": round(t,1), "e_final": round(e_final,4),
                         "sd": round(sd,4), "sd_raw": round(sd_raw,4),
                         "a_int": round(a_int,4), "z": round(z,3),
                         "sigma_min": round(w.sigma,8)}

def log(rec):
    jf.write(json.dumps(rec) + "\n")
    jf.flush()

def main():
    t0 = time.time()
    last_sigma = 0.0
    last_out = {}   # (asset,start) -> ts we last logged outcome
    while True:
        for asset in ASSETS:
            try:
                mk = get_market(asset)
                if not mk or not mk.get("active"):
                    continue
                start = mk["start"]; slug = mk["slug"]
                # resolve detection
                w = wins.get(slug)
                if mk.get("live_buy") is not None and (mk["live_buy"] >= 0.995 or mk["live_buy"] <= 0.005):
                    winner = "Up" if mk["live_buy"] >= 0.995 else "Down"
                    if not w or w.winner is None:
                        if w is None:
                            w = wins[slug] = Win(asset, start, (float(mk["thr"]) if mk["thr"] else None))
                        if w.winner is None:
                            w.winner = winner
                            log({"ts": datetime.now(timezone.utc).isoformat(), "kind": "outcome",
                                 "asset": asset, "window": slug, "start": start,
                                 "ref": w.ref, "winner": winner,
                                 "live_buy": mk["live_buy"], "live_sell": mk.get("live_sell")})
                            # drop old wins
                        last_out[slug] = time.time()
                        if w.winner is not None:
                            w.done = True
                    continue
                if w is None:
                    if not mk.get("thr"):
                        continue
                    w = wins[slug] = Win(asset, start, float(mk["thr"]))
                # sigma refresh
                if time.time() - last_sigma > 60:
                    pass
                if w.sigma is None or time.time() - sigcache.get(asset, (0,))[0] > 90:
                    s = sigma1m(asset)
                    if s:
                        w.sigma = s; sigcache[asset] = (time.time(), s)
                # sample spot every cycle (~1s)
                try:
                    sp = float(bnc.get(BINANCE + "/api/v3/ticker/price", params={"symbol": SYMBOL[asset]}, timeout=5).json()["price"])
                    w.spot = sp
                    w.path.append((time.time() - w.start, sp))
                    if len(w.path) > 700: w.path = w.path[-700:]
                except httpx.HTTPError:
                    pass
                now = time.time()
                t = now - w.start
                # evaluate at decision times (crossing trigger, robust to loop latency)
                for ts in T_STAR:
                    if (slug, ts) in fired or t < ts:
                        continue
                    fired.add((slug, ts))
                    p_up, info = model_p(w, now)
                    rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": "sample",
                           "asset": asset, "window": slug, "start": start, "t": round(t,1),
                           "ref": w.ref, "spot": w.spot, "p_up": p_up, "info": info,
                           "sigma": w.sigma, "live_buy": mk.get("live_buy"), "live_sell": mk.get("live_sell")}
                    if p_up is not None:
                        for outcome, token in mk["tokens"].items():
                            bb, ba = book(token)
                            rec[f"bid_{outcome}"] = bb; rec[f"ask_{outcome}"] = ba
                            rec[f"model_{outcome}"] = p_up if outcome == "Up" else round(1 - p_up, 6)
                            rec[f"mid_{outcome}"] = round((bb + ba) / 2, 4) if (bb is not None and ba is not None) else None
                    log(rec)
            except httpx.HTTPError:
                continue
            except Exception as e:
                print("err", asset, repr(e))
        # prune closed wins older than 10 min to bound memory
        now = time.time()
        for slug in [s for s, w in wins.items() if w.done and now - w.start > 600]:
            del wins[slug]
            for ts in T_STAR:
                fired.discard((slug, ts))
        time.sleep(1.0)

if __name__ == "__main__":
    main()
