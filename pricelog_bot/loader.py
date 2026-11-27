"""Shared loader for the full-book polymarket 5m logs.

Loads every window file once into a compact cache (pickle) and exposes frames
with a CAUSAL clock: t = max(Up.book_ts, Down.book_ts, coinbase.exchange_ts) - window_start.
Never trust the logger's own seconds_into_window (skew measured per source).
"""
import json, os, glob, pickle, sys

DATA_DIR = r"C:\Users\user\Desktop\bot\price_logs\polymarket"
CACHE = r"C:\Users\user\Desktop\bot\pricelog_bot\snapshots.pkl"
TAKER_FEE = 0.05  # 0.05 * min(p,1-p) * shares
MIN_ORDER_SHARES = 5  # polymarket per-market min_order_size in SHARES


def parse_side(d):
    if not d or d.get("ask") is None or d.get("bid") is None:
        return None
    ask, bid = float(d["ask"]), float(d["bid"])
    if not (0 < bid <= ask < 1):
        return None
    if ask - bid > 0.05:  # wide/stale book a live quote_book filter would reject
        return None
    return {
        "bid": bid, "ask": ask,
        "bid_sz": float(d.get("bid_size") or 0),
        "ask_sz": float(d.get("ask_size") or 0),
        "asks": [(float(x["price"]), float(x["size"])) for x in d.get("asks", [])[:8]],
        "bids": [(float(x["price"]), float(x["size"])) for x in d.get("bids", [])[:8]],
        "bts": float(d.get("book_timestamp") or 0) / 1000.0,
    }


def load(force=False):
    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    windows = {}
    files = sorted(glob.glob(os.path.join(DATA_DIR, "btc-updown-5m-*.jsonl")))
    for fp in files:
        slug = os.path.basename(fp)[:-6]
        wstart = int(slug.split("-")[-1])
        frames, winner = [], None
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("event") == "resolution":
                    winner = row.get("winner")
                    continue
                if row.get("event") != "snapshot":
                    continue
                up = parse_side(row.get("Up"))
                dn = parse_side(row.get("Down"))
                cb = row.get("coinbase") or {}
                cb_ts = cb.get("received_ts") or 0
                cb_price = cb.get("price")
                if not (up and dn):
                    continue  # drop invalid-book frames (skill rule 6)
                t_causal = max(up["bts"], dn["bts"], cb_ts or 0) - wstart
                frames.append({
                    "t": t_causal, "ts": row.get("ts_epoch", 0),
                    "up": up, "dn": dn,
                    "cb": cb_price, "cb_ts": cb_ts,
                })
        if winner in ("Up", "Down") and frames:
            windows[slug] = {"start": wstart, "winner": winner, "frames": frames}
    with open(CACHE, "wb") as f:
        pickle.dump(windows, f)
    return windows


def depth_cost(levels, dollars_cap):
    """Max dollars of stake fillable walking the asks without exceeding displayed depth at the touch."""
    # FOK: order must fit inside displayed size at the ask for full stake (per skill).
    return levels


if __name__ == "__main__":
    force = "--force" in sys.argv
    ws = load(force=force)
    n = len(ws)
    fcount = sum(len(w["frames"]) for w in ws.values())
    upw = sum(1 for w in ws.values() if w["winner"] == "Up")
    span = (max(w["start"] for w in ws.values()) - min(w["start"] for w in ws.values())) / 3600
    print(f"windows={n} frames={fcount} up_winners={upw} span_h={span:.1f}")
    # skew: logger stamp vs causal
    import statistics
    sk = []
    for w in list(ws.values())[:50]:
        for fr in w["frames"][:5]:
            sk.append(fr["t"] - (fr["ts"] - w["start"]))
    print(f"skew causal-stamp median={statistics.median(sk):.3f}s")