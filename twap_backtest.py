"""Backtest the TWAP fair-value model against historical Binance data.

For each 5-min window we reconstruct the 1-second price path from aggTrades,
compute the model's P(Up) at a decision time t* (e.g. 45s, 90s in), and
compare against the actual resolution: final_TWAP >= open price.

Also compares against the naive "spot momentum" baseline P(spot_now > open)
that the existing bots implicitly use, to show the TWAP model is different
and more accurate.
"""
import json
import math
import statistics
import sys
import time
import httpx

BINANCE = "https://api.binance.com"
SYMBOL = "BTCUSDT"
WINDOW = 300


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fetch_trades(bnc, symbol, start_ms, end_ms):
    out = []
    cur = start_ms
    while cur < end_ms:
        r = bnc.get(BINANCE + "/api/v3/aggTrades",
                    params={"symbol": symbol, "startTime": cur, "endTime": end_ms,
                            "limit": 1000}, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        cur = batch[-1]["T"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    return out


def build_windows(trades, window=WINDOW):
    """Group trades into aligned 5-min windows; return list of dicts."""
    if not trades:
        return []
    first_ts = float(trades[0]["T"]) / 1000
    start0 = int(first_ts // window * window)
    buckets = {}
    for t in trades:
        ts = float(t["T"]) / 1000
        p = float(t["p"])
        w = int(ts // window * window)
        buckets.setdefault(w, []).append((ts - w, p))  # offset within window
    wins = []
    for w in sorted(buckets):
        path = sorted(buckets[w])
        if len(path) < 50 or path[-1][0] < window - 5:  # needs ~full window
            continue
        wins.append({"start": w, "path": path})
    return wins


def model_p_up(path, ref, t_star, sigma_s):
    """P(Up) at time t_star given path up to t_star and per-second sigma."""
    T = float(WINDOW)
    t = t_star
    r = T - t
    if t < 5 or r < 5:
        return None
    # observed integral 0..t (trapezoid over the 1s path)
    segs = []
    prev = 0.0
    for off, p in path:
        if off > t_star:
            break
        segs.append((off - prev) * p)
        prev = off
    a_int = sum(segs) / T
    spot = path[-1][1] if path else ref
    # find spot at t_star (last price <= t_star)
    spot_t = ref
    for off, p in path:
        if off <= t_star:
            spot_t = p
        else:
            break
    e_final = a_int + r / T * spot_t
    var = spot_t ** 2 * sigma_s ** 2 * r ** 3 / (3 * T * T)
    sd = math.sqrt(max(var, 1e-18))
    z = (e_final - ref) / sd
    return norm_cdf(z), z, e_final, sd


def sigma_from_path(path, T=WINDOW):
    # per-second log return std from the full window path
    rets = []
    for i in range(1, len(path)):
        t0, p0 = path[i - 1]
        t1, p1 = path[i]
        dt = t1 - t0
        if dt <= 0 or p0 <= 0:
            continue
        rets.append(math.log(p1 / p0) / math.sqrt(dt))
    return statistics.pstdev(rets) if len(rets) > 20 else 0.0


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 3
    t_star = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    bnc = httpx.Client(timeout=15)
    now = int(time.time())
    end = now * 1000
    start = now - int(hours * 3600)
    print(f"Fetching {SYMBOL} aggTrades last {hours}h ...", flush=True)
    trades = fetch_trades(bnc, SYMBOL, start * 1000, end)
    print(f"  {len(trades)} trades", flush=True)
    wins = build_windows(trades)
    print(f"  {len(wins)} complete windows", flush=True)

    results = []
    for w in wins:
        path = w["path"]
        ref = path[0][1]
        sigma_s = sigma_from_path(path)
        if sigma_s <= 0:
            continue
        # actual final TWAP
        # actual final TWAP: time-weighted integral over real trade times
        integ = 0.0
        prev_off, prev_p = path[0][0], path[0][1]
        denom_end = path[-1][0]
        for off, p in path[1:]:
            integ += (off - prev_off) * prev_p
            prev_off, prev_p = off, p
        integ += (denom_end - prev_off) * prev_p
        final_twap = integ / max(denom_end, 1.0)
        up = final_twap >= ref
        m = model_p_up(path, ref, t_star, sigma_s)
        if m is None:
            continue
        p_up, z, e_final, sd = m
        # naive spot-momentum baseline: P(spot_t > ref) approx via z of spot
        spot_t = ref
        for off, p in path:
            if off <= t_star:
                spot_t = p
            else:
                break
        # naive: treat remaining as driftless, P(spot_T > ref)
        naive_z = math.log(spot_t / ref) / (sigma_s * math.sqrt(WINDOW - t_star))
        naive_p = norm_cdf(naive_z)
        results.append({"window": w["start"], "up": up, "p_model": p_up,
                        "z": z, "e_final": e_final, "sd": sd,
                        "spot_t": spot_t, "naive_p": naive_p,
                        "ref": ref, "final_twap": final_twap})

    if not results:
        print("no results"); return
    n = len(results)
    up_frac = sum(r["up"] for r in results) / n
    # calibration: bin model p_up
    bins = {}
    for r in results:
        b = round(r["p_model"] * 10) / 10
        bins.setdefault(b, []).append(r["up"])
    print(f"\n{n} windows, overall Up freq = {up_frac:.3f}, decision t*={t_star}s")
    print("\nModel calibration (bin of P(Up) -> actual Up freq):")
    print(f"  {'bin':>5} {'n':>5} {'actual_up':>9} {'model':>7}")
    for b in sorted(bins):
        arr = bins[b]
        print(f"  {b:5.1f} {len(arr):5d} {sum(arr)/len(arr):9.3f} {b:7.2f}")

    # Brier score: model vs naive
    def brier(pairs):
        return sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    model_brier = brier([(r["p_model"], 1 if r["up"] else 0) for r in results])
    naive_brier = brier([(r["naive_p"], 1 if r["up"] else 0) for r in results])
    print(f"\nBrier (lower=better):  model={model_brier:.4f}   naive_spot={naive_brier:.4f}")

    # accuracy at a 0.5 threshold
    def acc(pairs):
        return sum((p >= 0.5) == (y == 1) for p, y in pairs) / len(pairs)
    print(f"Accuracy @0.5:         model={acc([(r['p_model'], 1 if r['up'] else 0) for r in results]):.3f}   naive={acc([(r['naive_p'], 1 if r['up'] else 0) for r in results]):.3f}")

    # save raw for later
    with open("twap_backtest.json", "w") as f:
        json.dump({"t_star": t_star, "hours": hours, "results": results}, f)
    print(f"\nsaved twap_backtest.json ({len(results)} windows)")


if __name__ == "__main__":
    main()
