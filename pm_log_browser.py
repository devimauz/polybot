"""pm_log_browser.py - browse the Polymarket btc-updown-5m logs and graph them.

Dark-theme tkinter, dependency-free (numpy + stdlib tkinter only).

Tab "Window":
  left  = every logged window: start time, winner, frames, leader flips, overround
          (filter box: type a date/time fragment, "Up" or "Down"; buttons too)
  right = three charts + hover readout for the selected window:
          1) book: Up ask/bid (blue) and Down ask/bid (purple), price 0..1
          2) tape: Coinbase (green), Chainlink TWAP (cyan), strike (amber dashed)
          3) costs: overround cents (cyan); right-click toggles depth view

Tab "History": outcome strip (hoverable), rolling Up-rate, overround & flips
scatter, and P(leader at time T wins) curve computed from the logs themselves.

Run with the Hermes venv python (has numpy):
  "C:/Users/user/AppData/Local/hermes/installs/0d1790b230609f8c/environments/bb25ab9dd7a34e3599ece40a7efc18d9/venv/Scripts/python.exe" pm_log_browser.py
"""
import os
import sys
import threading
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import ttk

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import btc5m_endgame_backtest as B
import pm_rule_median as RULE

BG, PANEL, PANEL2, LINE = "#101217", "#181c24", "#1e2430", "#2a3140"
TEXT, MUTED = "#e9edf5", "#8896ac"
GREEN, RED = "#93e0be", "#f09caa"
AMBER, BLUE, PURPLE, CYAN = "#ffd080", "#80baff", "#c5a3ff", "#7fd4e8"
FONT = "Segoe UI"


class Chart(tk.Canvas):
    """Multi-series line chart; each series maps to the left ("l") or right ("r")
    y axis. Shared x axis, hover crosshair via callback."""

    def __init__(self, parent, height=220, ylabel_fmt="{:.2f}", ylabel_fmt_r=None):
        super().__init__(parent, height=height, bg=PANEL,
                         highlightthickness=1, highlightbackground=LINE)
        self.series = []
        self.markers = []
        self.mx = None
        self.on_hover = None
        self.ylabel_fmt = ylabel_fmt
        self.ylabel_fmt_r = ylabel_fmt_r
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Motion>", self._hover)
        self.bind("<Leave>", lambda e: (setattr(self, "mx", None), self.redraw()))

    def set(self, series, markers=None):
        out = []
        for s in series:
            x, y, c, w, d = s[:5]
            ax = s[5] if len(s) > 5 else "l"
            out.append((np.asarray(x, float), np.asarray(y, float), c, w, d, ax))
        self.series = out
        self.markers = markers or []
        self.redraw()

    def _finite(self, xs, ys):
        m = np.isfinite(xs) & np.isfinite(ys)
        return xs[m], ys[m]

    def _bounds(self):
        pts = [(a, b, ax) for xs, ys, _, _, _, ax in self.series
               for a, b in zip(*self._finite(xs, ys))]
        if len(pts) < 2:
            return None
        def bnd(axis):
            v = [(a, b) for a, b, ax in pts if ax == axis]
            if not v:
                return None
            xs = np.array([p[0] for p in v]); ys = np.array([p[1] for p in v])
            pad = (ys.max() - ys.min()) * 0.07 or 0.01
            return float(xs.min()), float(xs.max()), float(ys.min()) - pad, float(ys.max()) + pad
        bl = bnd("l")
        br = bnd("r") if self.ylabel_fmt_r else None
        if not bl:
            return None
        return bl, br

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60 or h < 50:
            return
        b = self._bounds()
        if not b:
            self.create_text(w / 2, h / 2, text="no data", fill=MUTED, font=(FONT, 10))
            return
        (x0, x1, ly0, ly1), br = b
        dual = br is not None
        L, R, T, Bo = 62, (52 if dual else 14), 10, 20
        pw, ph = w - L - R, h - T - Bo
        if pw < 10 or ph < 10:
            return
        def X(x): return L + (x - x0) / (x1 - x0 + 1e-12) * pw
        def Y(y): return T + (1 - (y - ly0) / (ly1 - ly0 + 1e-12)) * ph
        def Yr(y): return T + (1 - (y - br[2]) / (br[3] - br[2] + 1e-12)) * ph
        for f in range(6):
            gx = L + pw * f / 5
            self.create_line(gx, T, gx, T + ph, fill=LINE)
            self.create_text(gx, h - 9, text=f"{x0 + (x1 - x0) * f / 5:.0f}",
                             fill=MUTED, font=(FONT, 8))
        for f in range(5):
            gy = T + ph * f / 4
            self.create_line(L, gy, L + pw, gy, fill=LINE)
            self.create_text(L - 6, gy, text=self.ylabel_fmt.format(ly1 - (ly1 - ly0) * f / 4),
                             fill=MUTED, font=(FONT, 8), anchor="e")
            if dual:
                self.create_text(w - R + 6, gy, text=self.ylabel_fmt_r.format(br[3] - (br[3] - br[2]) * f / 4),
                                 fill=PURPLE if self.ylabel_fmt_r is not None else MUTED,
                                 font=(FONT, 8), anchor="w")
        for xs, ys, col, wd, dash, ax in self.series:
            Yf = Yr if (ax == "r" and dual) else Y
            pts = [(X(a), Yf(bv)) for a, bv in zip(*self._finite(xs, ys))]
            if len(pts) > 1:
                self.create_line(*[v for p in pts for v in p], fill=col, width=wd, dash=dash)
        for mx_, my_, ax, col, shape in self.markers:
            if not (np.isfinite(mx_) and np.isfinite(my_)):
                continue
            Yf = Yr if (ax == "r" and dual) else Y
            px_, py_ = X(mx_), Yf(my_)
            if not (L <= px_ <= L + pw and T <= py_ <= T + ph):
                continue
            if shape == "tri_up":
                self.create_polygon(px_ - 5, py_ + 8, px_ + 5, py_ + 8, px_, py_, fill=col, outline="")
            elif shape == "tri_dn":
                self.create_polygon(px_ - 5, py_ - 8, px_ + 5, py_ - 8, px_, py_, fill=col, outline="")
            else:
                self.create_rectangle(px_ - 4, py_ - 4, px_ + 4, py_ + 4, outline=col, width=2)
        if self.mx is not None:
            self.create_line(X(self.mx), T, X(self.mx), T + ph, fill=CYAN, dash=(2, 3))

    def _hover(self, e):
        b = self._bounds()
        if not b:
            return
        x0, x1, *_ = b[0]
        dual = b[1] is not None
        L, R = 62, (52 if dual else 14)
        pw = self.winfo_width() - L - R
        if pw > 0:
            self.mx = x0 + max(0.0, min(1.0, (e.x - L) / pw)) * (x1 - x0)
            self.redraw()
            if self.on_hover:
                self.on_hover(self.mx)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Polymarket 5m Log Browser")
        root.configure(bg=BG)
        root.geometry("1320x780")
        st = ttk.Style(root)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL, font=(FONT, 10))
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=24)
        st.configure("Treeview.Heading", background=PANEL2, foreground=MUTED, relief="flat")
        st.map("Treeview", background=[("selected", LINE)], foreground=[("selected", TEXT)])
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(18, 8))
        st.map("TNotebook.Tab", background=[("selected", PANEL2)], foreground=[("selected", TEXT)])

        self.W = None
        self.summ = []
        self.visible = []
        self.cur = None
        self.sumcur = None
        self.loaded = False
        self.pane3_mode = "or"
        self.rule_cache = {}
        self.rule_rows = None

        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(top, text="POLYMARKET · BTC-UPDOWN-5M · LOG BROWSER", bg=BG, fg=TEXT,
                 font=(FONT, 13, "bold")).pack(side="left")
        self.hdr = tk.Label(top, text="loading logs…", bg=BG, fg=AMBER, font=(FONT, 10))
        self.hdr.pack(side="right")
        self.rule_on = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="rule: cross strike after 60s, TP+20%", variable=self.rule_on,
                       bg=BG, fg=TEXT, selectcolor=PANEL2, activebackground=BG,
                       activeforeground=TEXT, font=(FONT, 10),
                       command=self.on_rule_toggle).pack(side="right", padx=14)
        self.rule_net = tk.Label(top, text="", bg=BG, fg=MUTED, font=(FONT, 10))
        self.rule_net.pack(side="right")

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=14, pady=8)
        tab_w = tk.Frame(nb, bg=BG)
        tab_h = tk.Frame(nb, bg=BG)
        nb.add(tab_w, text="  Window  ")
        nb.add(tab_h, text="  History  ")

        # ---------------- Window tab ----------------
        left = tk.Frame(tab_w, bg=PANEL)
        left.pack(side="left", fill="y", padx=(0, 10))
        fr = tk.Frame(left, bg=PANEL)
        fr.pack(fill="x", padx=8, pady=8)
        tk.Label(fr, text="filter:", bg=PANEL, fg=MUTED, font=(FONT, 9)).pack(side="left")
        self.filt = tk.Entry(fr, width=14, bg=PANEL2, fg=TEXT, insertbackground=TEXT,
                             relief="flat", font=(FONT, 10))
        self.filt.pack(side="left", padx=6, ipady=3)
        self.filt.bind("<KeyRelease>", lambda e: self.show_list())
        for lab in ("all", "Up", "Down", "clean"):
            ttk.Button(fr, text=lab, width=5,
                       command=lambda v="" if lab == "all" else lab: self.set_filter(v)
                       ).pack(side="left", padx=2)
        cols = ("time", "won", "frames", "flips", "ornd")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        for c, t, wd in zip(cols, ("start (local)", "won", "frames", "flips", "ovr¢"),
                            (112, 46, 58, 50, 50)):
            self.tree.heading(c, text=t, anchor="e")
            self.tree.column(c, width=wd, anchor="e")
        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=(0, 6))
        vsb.pack(side="left", fill="y", pady=(0, 6))
        self.tree.tag_configure("Up", foreground=GREEN)
        self.tree.tag_configure("Down", foreground=RED)
        self.tree.tag_configure("bad", foreground=AMBER)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        tk.Label(left, text="flips = leader changes vs strike · ovr¢ = overround cents/contract",
                 bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(anchor="w", padx=8, pady=(0, 8))

        right = tk.Frame(tab_w, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self.info = tk.Label(right, text="select a window on the left", anchor="w", bg=BG,
                             fg=MUTED, font=(FONT, 10), justify="left")
        self.info.pack(fill="x", pady=(0, 4))
        self.ch1 = Chart(right, height=300, ylabel_fmt="{:,.0f}", ylabel_fmt_r="{:.2f}")
        self.ch1.on_hover = self.hover_combined
        self.ch1.pack(fill="x", pady=(0, 6))
        row = tk.Frame(right, bg=BG)
        row.pack(fill="x")
        self.ch2 = Chart(row, height=180)
        self.ch2.on_hover = self.hover_book
        self.ch2.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.ch3 = Chart(row, height=180, ylabel_fmt="{:.1f}¢")
        self.ch3.pack(side="left", fill="both", expand=True)
        self.ch3.bind("<Button-3>", self.toggle_pane3)
        self.hoverlab = tk.Label(right, text="", bg=BG, fg=CYAN, font=("Consolas", 10), anchor="w")
        self.hoverlab.pack(fill="x", pady=(8, 0))
        lg = tk.Frame(right, bg=BG)
        lg.pack(fill="x", pady=2)
        for name, col in (("BTC price (left $)", GREEN), ("Up ask (right)", BLUE),
                          ("Down ask (right)", PURPLE), ("strike / TWAP", AMBER),
                          ("book bid-ask (pane 2)", MUTED), ("overround · R-click = depth", CYAN)):
            tk.Label(lg, text="▬ " + name, fg=col, bg=BG, font=(FONT, 9)).pack(side="left", padx=(0, 14))

        # ---------------- History tab ----------------
        hf = tk.Frame(tab_h, bg=BG)
        hf.pack(fill="both", expand=True, pady=6)
        self.stats = tk.Label(hf, text="", bg=BG, fg=TEXT, font=(FONT, 11), justify="left", anchor="w")
        self.stats.pack(fill="x", pady=(0, 8))
        self.strip = tk.Canvas(hf, height=60, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        self.strip.pack(fill="x")
        self.strip.bind("<Motion>", self.strip_hover)
        grid = tk.Frame(hf, bg=BG)
        grid.pack(fill="both", expand=True, pady=(10, 0))
        self.rate = Chart(grid, height=200, ylabel_fmt="{:.0f}%")
        self.rate.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.pwin = Chart(grid, height=200, ylabel_fmt="{:.0f}%")
        self.pwin.pack(side="left", fill="both", expand=True)
        self.striplab = tk.Label(hf, text="hover the strip for window details", bg=BG, fg=CYAN,
                                 font=("Consolas", 10), anchor="w")
        self.striplab.pack(fill="x", pady=(6, 0))

        threading.Thread(target=self.load_worker, daemon=True).start()
        root.after(300, self.tick)

    # ---------- rule overlay ----------
    def start_rule_replay(self):
        self.strikes = {s["ws"]: s["strike"] for s in self.summ}

        def work():
            rows = []
            for wi, d in enumerate(self.W):
                try:
                    r = RULE.window_trades(d, self.strikes.get(d["ws"]))
                except Exception:
                    r = None
                if r:
                    r["wi"] = wi
                    rows.append(r)
                    self.rule_cache[d["ws"]] = r
            self.rule_rows = rows
        self.rule_rows = None
        threading.Thread(target=work, daemon=True).start()

    def on_rule_toggle(self):
        if self.cur is not None:
            self.render_window()

    def _rule_overlay(self, d):
        """Returns (extra series, markers, summary text) for the selected window."""
        ser = []           # the amber strike line already drawn is the reference
        mk = []
        txt = "no trigger"
        tr = self.rule_cache.get(d["ws"])
        if tr is None and d["ws"] not in self.rule_cache:
            tr = RULE.window_trades(d, self.sumcur["strike"] if self.sumcur else None)
            self.rule_cache[d["ws"]] = tr or False
            if tr is False:
                tr = None
        if tr:
            col = BLUE if tr["side"] == "Up" else PURPLE
            mk.append((tr["te"], tr["pe"], "r", col, "tri_up" if tr["side"] == "Up" else "tri_dn"))
            if tr["kind"] == "tp":
                mk.append((tr["tx"], tr["px"], "r", GREEN, "box"))
                txt = (f"buy {tr['side']} @ {tr['pe']:.3f} (t={tr['te']:.0f}s) -> "
                       f"TP @ {tr['px']:.3f} (t={tr['tx']:.0f}s)  {self.money(tr['pnl'])}")
            else:
                pay = 1.0 if tr["kind"] == "settle-win" else 0.0
                mk.append((float(d["t"][-1]), pay, "r", GREEN if pay else RED, "box"))
                txt = (f"buy {tr['side']} @ {tr['pe']:.3f} (t={tr['te']:.0f}s) -> "
                       f"{tr['kind']}  {self.money(tr['pnl'])}")
        return ser, mk, txt

    def money(self, x):
        return f"{'+' if x >= 0 else ''}${x:.2f}"

    # ---------- data ----------
    def set_filter(self, v):
        self.filt.delete(0, "end")
        if v:
            self.filt.insert(0, v)
        self.show_list()

    def load_worker(self):
        W = [d for d in B.load_all() if d["_winner"] in ("Up", "Down")]
        by_ws = {d["ws"]: d for d in W}
        summ = []
        for d in W:
            t, cb = d["t"], d["cb_px"]
            cl = d.get("cl_px")
            f = np.isfinite(cb)
            strike = None
            if cl is not None and np.isfinite(cl).sum() > 5:
                strike = float(cl[np.isfinite(cl)][0])
            else:
                prev = by_ws.get(d["ws"] - 300)
                if prev is not None:
                    m = (prev["t"] >= 240) & np.isfinite(prev["cb_px"])
                    if m.sum() >= 20:
                        strike = float(np.mean(prev["cb_px"][m]))
            flips = 0
            if strike:
                sgn = np.where(f, np.sign(cb - strike), 0)
                sgn = sgn[sgn != 0]
                flips = int(np.count_nonzero(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0
            ou, ob, od, db = d["Up_ask"], d["Up_bid"], d["Down_ask"], d["Down_bid"]
            ok = np.isfinite(ou) & np.isfinite(od) & np.isfinite(ob) & np.isfinite(db)
            if ok.sum() >= 20:
                orl = (ou + od)[ok] - 1.0
                spu = np.maximum((ou - ob)[ok], (od - db)[ok])
                med_or = float(np.median(orl))
                med_sp = float(np.median(spu))
                frac_bad = float(np.mean((orl > 0.05) | (spu > 0.05)))
            else:
                med_or = med_sp = frac_bad = 1.0
            bad = med_or > 0.05 or med_sp > 0.05 or frac_bad > 0.20
            summ.append(dict(ws=d["ws"], win=d["_winner"], frames=int(len(t)), flips=flips,
                             ornd=med_or, med_sp=med_sp, frac_bad=frac_bad, bad=bad,
                             strike=strike,
                             cl_ok=bool(cl is not None and np.isfinite(cl).sum() > 5)))
        self.W, self.summ = W, summ
        self.loaded = True

    def filtered(self):
        q = self.filt.get().strip().lower()
        if not q:
            return list(self.summ)
        if q == "clean":
            return [s for s in self.summ if not s["bad"]]
        return [s for s in self.summ
                if (q in (s["win"] or "").lower()
                    or q in datetime.fromtimestamp(s["ws"]).strftime("%m-%d %H:%M").lower())]

    def show_list(self):
        self.tree.delete(*self.tree.get_children())
        if not self.summ:
            return
        self.visible = self.filtered()
        for s in self.visible:
            tags = [s["win"]] + (["bad"] if s["bad"] else [])
            self.tree.insert("", "end", tags=tags,
                             values=(datetime.fromtimestamp(s["ws"]).strftime("%m-%d %H:%M"),
                                     "!" if s["bad"] else s["win"], s["frames"], s["flips"],
                                     f"{s['ornd']*100:.1f}"))

    def on_select(self, _):
        sel = self.tree.selection()
        if not sel:
            return
        pos = list(self.tree.get_children()).index(sel[0])
        if pos >= len(self.visible):
            return
        s = self.visible[pos]
        self.cur = next(d for d in self.W if d["ws"] == s["ws"])
        self.sumcur = s
        self.render_window()

    def _frame_at(self, d, x):
        return int(np.clip(np.searchsorted(d["t"], x) - 1, 0, len(d["t"]) - 1))

    def _fmt(self, d, key, i):
        v = d[key][i]
        return f"{v:.3f}" if np.isfinite(v) else " -   "

    def hover_book(self, x):
        d = self.cur
        if d is None:
            return
        i = self._frame_at(d, x)
        self.hoverlab.configure(text=(
            f"t={d['t'][i]:6.1f}s   Up bid {self._fmt(d,'Up_bid',i)} / ask {self._fmt(d,'Up_ask',i)}   "
            f"Down bid {self._fmt(d,'Down_bid',i)} / ask {self._fmt(d,'Down_ask',i)}   "
            f"depth U {self._fmt(d,'Up_bidsz',i)} / {self._fmt(d,'Up_asksz',i)} sh"))

    def hover_combined(self, x):
        d = self.cur
        if d is None:
            return
        i = self._frame_at(d, x)
        cb = d["cb_px"][i]
        cl = d["cl_px"][i] if "cl_px" in d else np.nan
        s = self.sumcur
        txt = (f"t={d['t'][i]:6.1f}s   BTC ${cb:,.2f}" if np.isfinite(cb)
               else f"t={d['t'][i]:6.1f}s")
        if np.isfinite(cl):
            txt += f"   TWAP ${cl:,.2f}"
        txt += (f"   Up ask {self._fmt(d,'Up_ask',i)}   Down ask {self._fmt(d,'Down_ask',i)}")
        if s and s["strike"] and np.isfinite(cb):
            txt += f"   leading {'Up' if cb > s['strike'] else 'Down'}"
        self.hoverlab.configure(text=txt)

    def render_window(self):
        d, s = self.cur, self.sumcur
        t = d["t"]
        self.info.configure(
            text=(f"{datetime.fromtimestamp(d['ws']).strftime('%a %Y-%m-%d %H:%M:%S')}    "
                  f"winner  {d['_winner']}    frames {s['frames']}    flips {s['flips']}    "
                  f"overround {s['ornd']*100:.1f}¢    strike "
                  f"{'Chainlink' if s['cl_ok'] else 'rebuilt from prev window'}"
                  + ("    ⚠ BROKEN BOOK (quotes pulled) — excluded from 'clean' filter"
                     if s["bad"] else "")),
            fg=AMBER if s["bad"] else (GREEN if d["_winner"] == "Up" else RED))
        # combined chart: BTC + TWAP + strike on left $ axis, asks on right 0..1 axis
        p1 = [(t, d["cb_px"], GREEN, 2, (), "l")]
        if "cl_px" in d and np.isfinite(d["cl_px"]).sum() > 5:
            p1.append((t, d["cl_px"], AMBER, 1, (), "l"))
        if s["strike"]:
            p1.append((np.array([t[0], t[-1]]), np.array([s["strike"], s["strike"]]),
                       AMBER, 1, (6, 4), "l"))
        p1.append((t, d["Up_ask"], BLUE, 2, (), "r"))
        p1.append((t, d["Down_ask"], PURPLE, 2, (), "r"))
        markers = []
        if self.rule_on.get():
            extra, markers, rtxt = self._rule_overlay(d)
            p1 = extra + p1
            self.info.configure(text=self.info["text"] + "    |    RULE: " + rtxt)
        self.ch1.set(p1, markers=markers)
        self.ch2.set([(t, d["Up_ask"], BLUE, 2, ()), (t, d["Up_bid"], BLUE, 1, (3, 3)),
                      (t, d["Down_ask"], PURPLE, 2, ()), (t, d["Down_bid"], PURPLE, 1, (3, 3))])
        self.ch3.ylabel_fmt = "{:.1f}¢" if self.pane3_mode == "or" else "{:.0f}"
        self.ch3.set(self._pane3_series())

    def _pane3_series(self):
        d = self.cur
        t = d["t"]
        ou, od = d["Up_ask"], d["Down_ask"]
        if self.pane3_mode == "or":
            orl = np.where(np.isfinite(ou) & np.isfinite(od), (ou + od - 1) * 100, np.nan)
            return [(t, orl, CYAN, 2, ())]
        return [(t, d["Up_bidsz"], GREEN, 1, ()), (t, d["Up_asksz"], RED, 1, ()),
                (t, d["Down_bidsz"], BLUE, 1, ()), (t, d["Down_asksz"], PURPLE, 1, ())]

    def toggle_pane3(self, e):
        if self.cur is None:
            return
        self.pane3_mode = "depth" if self.pane3_mode == "or" else "or"
        self.render_window()

    # ---------- history ----------
    def render_history(self):
        n = len(self.summ)
        badn = sum(1 for s in self.summ if s["bad"])
        clean = [s for s in self.summ if not s["bad"]]
        ups = sum(1 for s in self.summ if s["win"] == "Up")
        flips = np.array([s["flips"] for s in self.summ], float)
        ornd = np.array([s["ornd"] * 100 for s in self.summ])
        span_h = (self.summ[-1]["ws"] - self.summ[0]["ws"]) / 3600
        self.stats.configure(text=(
            f"{n} windows · {span_h:.0f} h span · {badn} broken (excluded by 'clean' filter)   |   "
            f"Up base rate {ups / n * 100:.1f}%   |   "
            f"leader flips/window: mean {flips.mean():.2f}, median {np.median(flips):.0f}, "
            f"max {flips.max():.0f} ({(flips == 0).mean() * 100:.0f}% monotone)   |   "
            f"overround median (clean) {np.median([s['ornd'] for s in clean]) * 100:.1f}¢"))
        w = self.strip.winfo_width() or 1280
        self.strip.delete("all")
        bw = (w - 8) / max(n, 1)
        for i, s in enumerate(self.summ):
            x = 4 + i * bw
            col = "#4a5262" if s["bad"] else (GREEN if s["win"] == "Up" else RED)
            self.strip.create_rectangle(x, 8, x + max(bw - 0.5, 1), 52, fill=col, outline="")
        xs = np.arange(n)
        K = 24
        r = np.full(n, np.nan)
        for i in range(K, n):
            r[i] = sum(1 for x in self.summ[i - K:i] if x["win"] == "Up") / K * 100
        self.rate.set([(xs, r, BLUE, 2, ()),
                       (xs, ornd, CYAN, 1, ()),
                       (xs, flips, MUTED, 1, (3, 3))])
        Ts = np.arange(10, 300, 10)
        wins = np.zeros(len(Ts))
        tot = np.zeros(len(Ts))
        for d, s in zip(self.W, self.summ):
            if not s["strike"]:
                continue
            t, cb = d["t"], d["cb_px"]
            for k, T in enumerate(Ts):
                j = int(np.searchsorted(t, T)) - 1
                if j < 0 or not np.isfinite(cb[j]):
                    continue
                wins[k] += (d["_winner"] == "Up") == (cb[j] > s["strike"])
                tot[k] += 1
        p = np.where(tot > 0, wins / np.maximum(tot, 1) * 100, np.nan)
        self.pwin.set([(Ts, p, AMBER, 2, ()), (Ts, np.full_like(Ts, 50.0), MUTED, 1, (4, 4))])

    def strip_hover(self, e):
        if not self.summ:
            return
        w = self.strip.winfo_width()
        i = int((e.x - 4) / ((w - 8) / len(self.summ)))
        if 0 <= i < len(self.summ):
            s = self.summ[i]
            ts = datetime.fromtimestamp(s["ws"]).strftime("%a %m-%d %H:%M")
            self.striplab.configure(
                text=(f"{ts}   winner {s['win']}   flips {s['flips']}   "
                      f"overround {s['ornd']*100:.1f}¢   "
                      f"{'BROKEN BOOK' if s['bad'] else ('Chainlink strike' if s['cl_ok'] else 'rebuilt strike')}"))

    # ---------- loop ----------
    def tick(self):
        try:
            if self.loaded and not getattr(self, "_shown", False):
                self._shown = True
                self.hdr.configure(
                    text=f"{len(self.summ)} windows · "
                         f"{(self.summ[-1]['ws'] - self.summ[0]['ws']) / 3600:.0f} h", fg=GREEN)
                self.show_list()
                self.start_rule_replay()
                self.root.after(200, self.render_history)
            if self.rule_rows is not None and not getattr(self, "_rule_shown", False):
                self._rule_shown = True
                tot = sum(r["pnl"] for r in self.rule_rows)
                tp = sum(1 for r in self.rule_rows if r["kind"] == "tp")
                self.rule_net.configure(
                    text=f"rule all-sample: {len(self.rule_rows)} trades, {tp} TP-hits, "
                         f"net {'+' if tot >= 0 else ''}${tot:.2f}  ",
                    fg=GREEN if tot > 0 else RED)
                if self.cur is not None and self.rule_on.get():
                    self.render_window()
        except Exception as ex:
            self.hdr.configure(text=f"error: {ex}", fg=RED)
        self.root.after(1500, self.tick)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
