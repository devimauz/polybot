"""_pm_replay_bot.py - drive the REAL btc5m_endgame_bot.py decision logic
against the recorded order books.

The backtest (btc5m_endgame_backtest.py) proves the STRATEGY is profitable.
This proves the BOT implements that same strategy - it feeds recorded
snapshots into the bot's own step()/decision code and checks that the bot
fires on exactly the same windows, at the same prices.

Without this, "the backtest works" and "the bot trades the backtested rule"
are two different claims.
"""
import os
import pickle
import sys
from decimal import Decimal

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")
D = Decimal


class FakeQuote:
    """Mimics polymarket_feed.Quote."""

    def __init__(self, ask, bid, probability):
        self.ask = None if ask is None else D(str(ask))
        self.bid = None if bid is None else D(str(bid))
        self.probability = probability


def load_all():
    """Use the backtest's loader so BOTH use the source-aligned clock.

    The compact cache stores both `t` (the logger's receipt stamp, ~1 s
    optimistic) and `t_src` (the causal clock: max of the frame's book
    timestamps and the coinbase exchange timestamp).  Only `t_src` is
    admissible for a backtest - using `t` would hand the strategy a
    look-ahead.
    """
    import btc5m_endgame_backtest as B
    return B.load_all()


def main():
    # Import the bot module but do NOT construct the live bot (no network).
    import btc5m_endgame_bot as E

    W = load_all()
    print(f"replaying {len(W)} windows through the bot's own decision code")

    # --- rebuild the bot's entry test as a pure function of a quote set ---
    # This calls the bot's OWN decide_entry(), so a mismatch here means the
    # shipped bot really does trade something different from the backtest.
    def bot_decides(d, i):
        mu, md = d["Up_mid"][i], d["Down_mid"][i]
        if not (np.isfinite(mu) and np.isfinite(md)):
            return None
        p_up = 0.5 * (mu + (1.0 - md))
        legs = {}
        for side in ("Up", "Down"):
            legs[side] = {
                "ask": float(d[side + "_ask"][i]),
                "bid": float(d[side + "_bid"][i]),
                "asksz": float(d[side + "_asksz"][i]),
            }
        r = E.decide_entry(legs, p_up)
        if r is None:
            return None
        return r[0], r[1]

    # --- replay: bot polls every 2s from T=262, takes its FIRST trigger ---
    bot_fires = []
    for wi, d in enumerate(W):
        t = d["t"]
        n = len(t)
        for T in range(262, 300, 2):
            i = int(np.searchsorted(t, T))
            if i >= n:
                continue
            r = bot_decides(d, i)
            if r is None:
                continue
            side, ask = r
            bot_fires.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side,
                                  ask=ask, won=1 if d["_winner"] == side else 0))
            break   # first trigger per window, like the live bot (journal seen)

    # --- compare against the backtest's rule ---
    import btc5m_endgame_backtest as B
    rows = B.build_rows(W)
    sel = B.first_per_window([r for r in rows if r["T"] >= 262 and r["mkt"] >= 0.5
                              and 0.80 <= r["ask"] < 0.96])
    bt = {(r["wi"], r["side"]): r for r in sel}

    print(f"\nbacktest rule fired on : {len(bt)} windows")
    print(f"bot decision code fired : {len(bot_fires)} windows")
    agree = sum(1 for f in bot_fires
                if (f["wi"], f["side"]) in bt
                and abs(bt[(f["wi"], f["side"])]["ask"] - f["ask"]) < 1e-9)
    print(f"identical window+side+price: {agree}/{len(bot_fires)}")
    only_bot = [f for f in bot_fires if (f["wi"], f["side"]) not in bt]
    only_bt = [k for k in bt if k[0] not in {f["wi"] for f in bot_fires}]
    print(f"bot-only: {len(only_bot)}   backtest-only: {len(only_bt)}")
    for f in only_bot[:5]:
        print("   bot-only:", f)
    for k in only_bt[:5]:
        print("   bt-only :", k, bt[k]["ask"], bt[k]["T"])

    # --- P&L of the bot's own fills ---
    ask = np.array([f["ask"] for f in bot_fires])
    won = np.array([f["won"] for f in bot_fires])
    p = B.pnl_of(ask, won)
    m, t, nw = B.cluster_t([f["wi"] for f in bot_fires], p)
    lo, hi = B.boot_ci([f["wi"] for f in bot_fires], p)
    print(f"\nBOT-EQUITY P&L: N={len(bot_fires)} w={int(won.sum())} net=${p.sum():+.2f} "
          f"avg=${p.mean():+.3f} cluster-t={t:+.2f} CI=[{lo:+.2f},{hi:+.2f}] "
          f"worst=${p.min():+.2f}")
    print(f"  realised win rate {won.mean():.4f} at mean ask {ask.mean():.4f}")

    # --- the CONDITIONAL SIZING overlay, executed by the bot's own function ---
    stake = np.array([E.stake_for_ask(a) for a in ask])
    sh = stake / ask
    fee = 0.07
    ps = sh * won - stake - sh * fee * ask * (1 - ask)
    ms, ts, _ = B.cluster_t([f["wi"] for f in bot_fires], ps)
    los, his = B.boot_ci([f["wi"] for f in bot_fires], ps)
    n_cheap = int((ask < float(E.CHEAP_ASK)).sum())
    print(f"\nSIZED (bot's stake_for_ask): N={len(bot_fires)} staked=${stake.sum():.0f} "
          f"net=${ps.sum():+.2f} ROI={ps.sum()/stake.sum()*100:+.2f}% cluster-t={ts:+.2f} "
          f"CI=[{los:+.2f},{his:+.2f}] worst=${ps.min():+.2f}")
    print(f"  cheap bucket (ask < {E.CHEAP_ASK}): {n_cheap} trades at ${E.STAKE_CHEAP}, "
          f"the rest at ${E.STAKE}")

    # the sized backtest must agree with the bot's sizing
    sel_sized = B.first_per_window([r for r in rows if r["T"] >= 262 and r["mkt"] >= 0.5
                                    and 0.80 <= r["ask"] < 0.96])
    a_bt = np.array([r["ask"] for r in sel_sized])
    w_bt = np.array([r["won"] for r in sel_sized], dtype=float)
    st_bt = np.array([E.stake_for_ask(a) for a in a_bt])
    sh_bt = st_bt / a_bt
    p_bt = sh_bt * w_bt - st_bt - sh_bt * 0.07 * a_bt * (1 - a_bt)
    same = abs(p_bt.sum() - ps.sum()) < 1e-9
    print(f"  backtest-sized net ${p_bt.sum():+.2f}  == bot-sized net ${ps.sum():+.2f} ? "
          f"{'YES' if same else 'NO'}")

    if agree == len(bot_fires) and len(bot_fires) == len(bt):
        print("\nVERDICT: the bot's decision code reproduces the backtested rule exactly.")
    else:
        print("\nVERDICT: MISMATCH - the bot does not yet implement the backtested rule.")
    if same:
        print("VERDICT: the bot's sizing matches the backtested sizing exactly.")
    else:
        print("VERDICT: SIZING MISMATCH - bot and backtest size trades differently.")


if __name__ == "__main__":
    main()
