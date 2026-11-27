"""Extract compact per-snapshot features from price_logs/polymarket/*.jsonl.

Reads BOTH plain .jsonl and gzipped .jsonl.gz windows (the daily rotation gzips
settled ones).  Writes one .npz per window into _pm_cache/ plus a manifest with
winners.  Features are intentionally kept flat + small so full re-analysis is fast.
"""
import gzip, json, glob, os, sys, pickle, time
import numpy as np

SRC = r"C:/Users/user/Desktop/bot/price_logs/polymarket"
OUT = r"C:/Users/user/Desktop/bot/_pm_cache"
os.makedirs(OUT, exist_ok=True)

MAXLV = 12  # book levels to keep per side


def slug_of(path):
    b = os.path.basename(path)
    for suf in (".jsonl.gz", ".jsonl", ".gz"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return b


def open_log(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def parse_file(path):
    slug = slug_of(path)
    recs = []
    winner = None
    cond = None
    with open_log(path) as fh:
        for line in fh:
            if '"snapshot"' in line:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                recs.append(r)
            elif '"resolution"' in line:
                try:
                    r = json.loads(line)
                    winner = r.get("winner")
                    cond = r.get("condition_id")
                except Exception:
                    pass
    if not recs:
        return None
    recs.sort(key=lambda r: r.get("seconds_into_window", 0.0))
    n = len(recs)
    F = {}
    F["t"] = np.array([r.get("seconds_into_window", 0.0) for r in recs], dtype=np.float64)
    F["ts_epoch"] = np.array([r.get("ts_epoch", 0.0) for r in recs], dtype=np.float64)
    F["ws"] = np.array([r.get("window_start", 0) for r in recs], dtype=np.int64)

    # SOURCE-ALIGNED time.  The logger stamps each row BEFORE its REST calls
    # run, so `seconds_into_window` is the time the cycle STARTED, not the
    # time the data in that row became observable.  A frame cannot be acted
    # on until its slowest component exists, so the earliest causal
    # timestamp for a frame is:
    #     max(Up.book_timestamp, Down.book_timestamp, coinbase.exchange_ts)
    # minus the window start.  This is the canonical loader used by the
    # repo's own _research_value_stop.py.
    ws = F["ws"][0] if n else 0

    def _epoch(v):
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if v > 1e12:          # milliseconds
            v = v / 1000.0
        elif v > 1e9:         # seconds
            pass
        else:                 # ISO string -> unparsable here
            return None
        return v

    src_t = np.full(n, np.nan)
    for i, r in enumerate(recs):
        cands = []
        for side in ("Up", "Down"):
            bt = _epoch((r.get(side) or {}).get("book_timestamp"))
            if bt is not None:
                cands.append(bt)
        cb = r.get("coinbase") or {}
        et = cb.get("exchange_ts")
        e = _epoch(et)
        if e is None and isinstance(et, str):
            # ISO-8601 -> epoch
            s = et.replace("Z", "+00:00")
            try:
                import datetime as _dt
                e = _dt.datetime.fromisoformat(s).timestamp()
            except Exception:
                e = None
        if e is not None:
            cands.append(e)
        if cands:
            src_t[i] = max(cands) - float(ws)
    # fall back to the logger stamp where the source stamps are missing
    fallback = np.where(np.isfinite(src_t), src_t, F["t"])
    F["t_src"] = fallback
    F["t_lag"] = fallback - F["t"]   # how optimistic the logger stamp was

    for side in ("Up", "Down"):
        d = [r.get(side) or {} for r in recs]
        F[side + "_bid"] = np.array([(x.get("bid") if x.get("bid") is not None else np.nan) for x in d], dtype=np.float64)
        F[side + "_ask"] = np.array([(x.get("ask") if x.get("ask") is not None else np.nan) for x in d], dtype=np.float64)
        F[side + "_mid"] = np.array([(x.get("mid") if x.get("mid") is not None else np.nan) for x in d], dtype=np.float64)
        F[side + "_spread"] = np.array([(x.get("spread") if x.get("spread") is not None else np.nan) for x in d], dtype=np.float64)
        F[side + "_bidsz"] = np.array([(x.get("bid_size") if x.get("bid_size") is not None else np.nan) for x in d], dtype=np.float64)
        F[side + "_asksz"] = np.array([(x.get("ask_size") if x.get("ask_size") is not None else np.nan) for x in d], dtype=np.float64)
        # top-of-book depth arrays
        for k in ("bids", "asks"):
            arr = np.full((n, MAXLV), np.nan, dtype=np.float64)
            sz = np.full((n, MAXLV), np.nan, dtype=np.float64)
            for i, x in enumerate(d):
                lv = x.get(k) or []
                for j, e in enumerate(lv[:MAXLV]):
                    try:
                        arr[i, j] = float(e["price"])
                        sz[i, j] = float(e["size"])
                    except Exception:
                        break
            F[f"{side}_{k}_px"] = arr
            F[f"{side}_{k}_sz"] = sz

    cb = [(r.get("coinbase") or {}) for r in recs]
    F["cb_px"] = np.array([(c.get("price") if c.get("price") is not None else np.nan) for c in cb], dtype=np.float64)
    F["cb_age"] = np.array([(c.get("age_ms") if c.get("age_ms") is not None else np.nan) for c in cb], dtype=np.float64)
    F["cb_recv"] = np.array([(c.get("received_ts") if c.get("received_ts") is not None else np.nan) for c in cb], dtype=np.float64)

    # Polymarket's own settlement TWAP (Chainlink 60s).  The window resolves on
    # the value of this series at the window end, against its value at the open.
    cl = [(r.get("chainlink_twap_60s") or {}) for r in recs]
    F["cl_px"] = np.array([(c.get("price") if c.get("price") is not None else np.nan) for c in cl], dtype=np.float64)
    F["cl_age"] = np.array([(c.get("age_ms") if c.get("age_ms") is not None else np.nan) for c in cl], dtype=np.float64)
    F["cl_src_ts"] = np.array([(c.get("source_ts_epoch") if c.get("source_ts_epoch") is not None else np.nan) for c in cl], dtype=np.float64)

    F["_winner"] = winner
    F["_cond"] = cond
    F["_slug"] = slug
    F["_n"] = n
    return F


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.jsonl"))
                   + glob.glob(os.path.join(SRC, "*.jsonl.gz")))
    manifest = []
    t0 = time.time()
    done = skipped = 0
    for i, p in enumerate(files):
        slug = slug_of(p)
        outp = os.path.join(OUT, slug + ".npz")
        if os.path.exists(outp):
            try:
                z = np.load(outp, allow_pickle=True)
                manifest.append({k: z[k].item() if z[k].shape == () else z[k] for k in ("_slug", "_winner", "_n", "_cond")})
                skipped += 1
                continue
            except Exception:
                pass
        F = parse_file(p)
        if F is None:
            continue
        np.savez_compressed(outp, **F)
        manifest.append({"_slug": F["_slug"], "_winner": F["_winner"], "_n": F["_n"], "_cond": F["_cond"]})
        done += 1
        if done % 25 == 0:
            print(f"{i}/{len(files)} parsed={done} cached={skipped} "
                  f"{slug} n={F['_n']} winner={F['_winner']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, "manifest.pkl"), "wb") as fh:
        pickle.dump(manifest, fh)
    print(f"DONE manifest={len(manifest)} parsed={done} already_cached={skipped} "
          f"files={len(files)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
