"""pm_latency_gui.py - live monitor for the Polymarket latency paper bot.

Real-time view of what the bot sees and does on the AWS box:
  - BTC vs strike with Up/Down asks (dual axis), signal/order/skip markers
  - live event stream (signals, paper orders, repriced-skips, settlements)
  - paper P&L since start, per-window settle strip, counters
  - pulsing LIVE dot + last-poll age (fast heartbeat, decoupled from polling)

Polls the box every ~5 s through _aws_ssh.run() in a worker thread; the Tk
thread only renders when a new snapshot arrives (proven pattern).

Run:  pm_latency_gui.bat   (venv python has paramiko + numpy)
"""
import json
import sys
import threading
import time
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, ".")
import os as _os
if _os.environ.get("PMGUI_HOST", "vps") == "aws":
    import _aws_ssh as _ssh
    _ssh_label = "AWS-Stockholm"
else:
    import _vps_ssh as _ssh
    _ssh_label = "AMS-live"
from pm_log_browser import Chart, BG, PANEL, PANEL2, LINE, TEXT, MUTED
from pm_log_browser import GREEN, RED, AMBER, BLUE, PURPLE, CYAN, FONT

POLL = 5.0
GOLD = "#F5B60A"
CMD_TMPL = (
    "date +%s; systemctl is-active pm-latency-bot; "
    "CUR=$(date -u +%s | awk '{print int($1/300)*300}'); "
    "echo '###COUNT'; "
    "for s in signal paper_order skip_repriced live_order live_err live_skip_minsize "
    'H"A"LT live_order_resp; do '
    "printf '%s ' $s; grep -h \"\\\"src\\\":\\\"$s\\\"\" ~/poly/latencylog/bot-*.jsonl 2>/dev/null | wc -l; done; "
    "printf 'settle '; grep -h '\"src\":\"settle\"' ~/poly/latencylog/bot-*.jsonl 2>/dev/null | "
    "grep -o '\"ws\":[0-9]*' | sort -u | wc -l; "
    "echo '###MODE'; grep -h '\"src\":\"boot\"' ~/poly/latencylog/bot-*.jsonl 2>/dev/null | tail -1 | "
    "grep -o '\"mode\":\"[a-z]*\"' | sed 's/.*:\"//;s/\"//'; "
    "echo '###SETTLE'; "
    "grep -h '\"src\":\"settle\"' ~/poly/latencylog/bot-*.jsonl 2>/dev/null | tail -40; "
    "echo '###BOT'; "
    "tail -n 60 ~/poly/latencylog/bot-$CUR.jsonl 2>/dev/null; "
    "echo '###CHART'; "
    "tail -n 9000 ~/poly/laglog/lag-$CUR.jsonl 2>/dev/null | "
    "grep -E '\"src\":\"pm\"|\"tp\":\"t\"' | awk 'NR%3'; "
    "echo '###STRIKE'; "
    "PREV=$((CUR-300)); "
    "tail -n 500 ~/poly/laglog/lag-$PREV.jsonl 2>/dev/null | "
    "grep '\"tp\":\"t\"' | tail -120 | "
    "grep -o '\"bb\":[0-9.]*' | cut -d: -f2 | "
    "awk '{s+=$1; n++} END {if(n>10) print s/n}'"
)


class App:
    def __init__(self, root):
        self.root = root
        root.title("PM Latency Bot - LIVE")
        root.configure(bg=BG)
        root.geometry("1280x780")
        self.snap = None
        self.err = None
        self.last_poll = 0.0
        self.seen = set()
        self.events = []
        self.pulse = False
        self.strike = None
        self.strike_ws = 0

        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=12, pady=(8, 2))
        self.dot = tk.Canvas(top, width=16, height=16, bg=BG, highlightthickness=0)
        self.dot.pack(side="left", padx=(2, 6))
        tk.Label(top, text=f"LATENCY BOT [{_ssh_label}]", bg=BG, fg=TEXT, font=(FONT, 13, "bold")).pack(side="left")
        self.mode_badge = tk.Label(top, text=" ? ", bg=PANEL, fg=MUTED, font=(FONT, 11, "bold"))
        self.mode_badge.pack(side="left", padx=8)
        self.age = tk.Label(top, text="", bg=BG, fg=MUTED, font=(FONT, 10))
        self.age.pack(side="left", padx=10)
        self.status = tk.Label(top, text="connecting…", bg=BG, fg=AMBER, font=(FONT, 10))
        self.status.pack(side="left", padx=10)
        self.countdown = tk.Label(top, text="", bg=BG, fg=MUTED, font=(FONT, 10))
        self.countdown.pack(side="left", padx=10)
        self.pnl = tk.Label(top, text="P&L $0.00", bg=BG, fg=TEXT, font=(FONT, 16, "bold"))
        self.pnl.pack(side="right")
        self.cnt = tk.Label(top, text="", bg=BG, fg=MUTED, font=(FONT, 10))
        self.cnt.pack(side="right", padx=16)

        self.chart = Chart(root, height=340, ylabel_fmt="{}", ylabel_fmt_r="{:.2f}")
        self.chart.ylabel_fmt = "{:,.0f}"
        self.chart.pack(fill="x", padx=12, pady=(4, 6))

        row = tk.Frame(root, bg=BG)
        row.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        left = tk.Frame(row, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(left, text="per-window settled paper P&L", bg=BG, fg=MUTED,
                 font=(FONT, 9)).pack(anchor="w")
        self.strip = tk.Canvas(left, height=120, bg=PANEL, highlightthickness=1,
                               highlightbackground=LINE)
        self.strip.pack(fill="x")
        self.legend = tk.Label(left, text="▬ BTC (left $)   ▬ strike   ▬ Up ask (right)   ▬ Down ask (right)   "
                                          "▲ signal   ◆ paper order   ✕ repriced-skip",
                               bg=BG, fg=MUTED, font=(FONT, 9), justify="left")
        self.legend.pack(anchor="w", pady=6)

        right = tk.Frame(row, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="event stream", bg=BG, fg=MUTED, font=(FONT, 9)).pack(anchor="w")
        self.log = tk.Text(right, bg=PANEL, fg=TEXT, insertbackground=TEXT, relief="flat",
                           font=("Consolas", 9), state="disabled", wrap="none")
        self.log.pack(fill="both", expand=True)
        for tag, col in (("sig", AMBER), ("ord", CYAN), ("skip", MUTED), ("set", GREEN),
                         ("setl", RED), ("sys", TEXT), ("halt", RED)):
            self.log.tag_configure(tag, foreground=col)

        threading.Thread(target=self.worker, daemon=True).start()
        root.after(120, self.render_tick)
        root.after(500, self.beat)

    # ---------- worker ----------
    def worker(self):
        while True:
            try:
                out, err, code = _ssh.run(CMD_TMPL, timeout=40)
                self.snap = out
                self.err = None
            except Exception as e:
                self.err = str(e)[:120]
            self.last_poll = time.time()
            time.sleep(POLL)

    # ---------- render ----------
    def beat(self):
        self.pulse = not self.pulse
        age = time.time() - self.last_poll if self.last_poll else 999
        col = GREEN if (self.pulse and age < POLL * 2.5) else ("#3a5a48" if age < POLL * 2.5 else RED)
        self.dot.delete("all")
        self.dot.create_oval(3, 3, 14, 14, fill=col, outline="")
        self.age.configure(text=f"last poll {age:.0f}s ago" if self.last_poll else "…")
        now = int(time.time())
        self.countdown.configure(text=f"window ends in {300 - now % 300}s")
        self.root.after(500, self.beat)

    def render_tick(self):
        snap = self.snap
        if snap is not None and snap is not getattr(self, "_last_snap", None):
            self._last_snap = snap
            try:
                self.render(snap)
            except Exception as e:
                self.status.configure(text=f"render error: {e}", fg=RED)
        elif self.err:
            self.status.configure(text=f"ssh: {self.err}", fg=RED)
        self.root.after(150, self.render_tick)

    def render(self, out):
        parts = {}
        key = "head"
        buf = []
        for line in out.splitlines():
            if line.startswith("###"):
                parts[key] = buf
                key = line[3:]
                buf = []
            else:
                buf.append(line)
        parts[key] = buf
        head = parts.get("head", [])
        active = head[1].strip() if len(head) > 1 else "?"
        self.status.configure(text=f"bot {active}", fg=GREEN if active == "active" else RED)

        counts = {}
        for ln in parts.get("COUNT", []):
            f = ln.split()
            if len(f) == 2:
                counts[f[0]] = int(f[1])
        mode = (parts.get("MODE") or ["paper"])[0].strip() or "paper"
        live = mode == "live"
        self.mode_badge.configure(text=" LIVE " if live else " paper ",
                                  fg="#111111" if live else MUTED,
                                  bg=GOLD if live else PANEL)
        lf = counts.get("live_order_resp", 0)
        self.cnt.configure(text=(f"signals {counts.get('signal',0)}   "
                                 f"orders {counts.get('paper_order',0)+counts.get('live_order',0)}"
                                 f" ({counts.get('live_order',0)} live: "
                                 f"{counts.get('live_order_resp',0)} filled, "
                                 f"{counts.get('live_err',0)} miss)   "
                                 f"skips {counts.get('skip_repriced',0)}   "
                                 f"settled {counts.get('settle',0)}"))

        byws = {}
        for ln in parts.get("SETTLE", []):
            try:
                r = json.loads(ln)
                byws[r["ws"]] = r          # last row per window wins -> restart-safe
            except Exception:
                pass
        settles = [byws[k] for k in sorted(byws)]
        tot = sum(s.get("pnl", 0) for s in settles)
        life = None                        # lifetime pnl straight from the bot heartbeat
        for ln in parts.get("BOT", []):
            try:
                h = json.loads(ln)
                if h.get("src") == "hb":
                    life = h.get("pnl")
            except Exception:
                pass
        if life is not None:
            self.pnl.configure(
                text=(f"P&L {'+' if life >= 0 else ''}${life:.2f}   "
                      f"(last 40: {'+' if tot >= 0 else ''}${tot:.2f})"),
                fg=GREEN if life >= 0 else RED)
        else:
            self.pnl.configure(text=f"recent P&L {'+' if tot >= 0 else ''}${tot:.2f}",
                           fg=GREEN if tot >= 0 else RED)
        self.draw_strip(settles)

        # events -> log + markers (dedupe by t_ns)
        new_ev = []
        for ln in parts.get("BOT", []):
            if '"t_ns"' not in ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r["t_ns"] in self.seen:
                continue
            self.seen.add(r["t_ns"])
            new_ev.append(r)
        self.events = (self.events + new_ev)[-400:]
        for r in new_ev:
            self.append_log(r)

        # strike estimate
        st = None
        for ln in parts.get("STRIKE", []):
            try:
                st = float(ln)
            except Exception:
                pass
        if st:
            self.strike = st

        # chart
        bt, bx, ut, ux, dt, dx = [], [], [], [], [], []
        for ln in parts.get("CHART", []):
            if '"t_ns"' not in ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            t = (r["t_ns"] / 1e9) % 300
            if r.get("src") == "pm":
                ut.append(t); ux.append(r["baU"])
                dt.append(t); dx.append(r["baD"])
            elif r.get("src") == "cb" and r.get("tp") == "t":
                if r.get("bb") and r.get("ba"):
                    bt.append(t); bx.append((r["bb"] + r["ba"]) / 2)
        mk = []
        for r in self.events:
            t = (r["t_ns"] / 1e9) % 300
            s = r.get("src")
            if s == "signal":
                mk.append((t, r.get("ask", np.nan), "r", AMBER, "box"))
            elif s in ("paper_order", "live_order"):
                col = (GOLD if s == "live_order"
                       else (BLUE if r.get("aligned") == "Up" else PURPLE))
                mk.append((t, r.get("ask", np.nan), "r", col,
                           "tri_up" if r.get("aligned") == "Up" else "tri_dn"))
            elif s == "skip_repriced":
                mk.append((t, r.get("a_now", np.nan), "r", "#667085", "box"))
        series = [
            (np.array(bt), np.array(bx), GREEN, 2, (), "l"),
            (np.array([0, 300]), np.array([self.strike or 0] * 2), AMBER, 1, (6, 4), "l")
            if self.strike else (np.array([]), np.array([]), AMBER, 1, (), "l"),
            (np.array(ut), np.array(ux), BLUE, 2, (), "r"),
            (np.array(dt), np.array(dx), PURPLE, 2, (), "r"),
        ]
        self.chart.set(series, markers=mk)

    def append_log(self, r):
        s = r.get("src")
        ts = datetime.fromtimestamp(r["t_ns"] / 1e9).strftime("%H:%M:%S.%f")[:-3]
        if s == "signal":
            line, tag = f"{ts}  SIGNAL  {r.get('d'):+.2f}$ -> {r.get('aligned')} (ask {r.get('ask')})", "sig"
        elif s == "paper_order":
            line, tag = (f"{ts}  ORDER   BUY {r.get('shares')} sh {r.get('aligned')} @ {r.get('ask')}"
                         f"  [$5 paper]"), "ord"
        elif s == "live_order":
            line, tag = (f"{ts}  LIVE ORDER  BUY {r.get('shares')} sh {r.get('aligned')} "
                         f"@ {r.get('ask')}  [FAK ~${(r.get('shares') or 0) * (r.get('ask') or 0):.2f}]"), "live"
        elif s == "live_order_resp":
            try:
                import ast as _ast
                d = _ast.literal_eval(r.get("resp", "{}"))
            except Exception:
                d = {}
            if d.get("success"):
                line, tag = (f"{ts}  *** FILL  {d.get('status','?')}  got {d.get('takingAmount')} sh "
                             f"for ${d.get('makingAmount')}  cap {r.get('cap')} "
                             f"rtt {r.get('rtt_ms')}ms ***"), "live"
            else:
                line, tag = f"{ts}  resp(no fill): {str(d.get('errorMsg') or d)[:80]}", "skip"
        elif s == "live_err":
            e = str(r.get("err", ""))
            if "no orders found" in e:
                line, tag = (f"{ts}  miss    race lost, ask gone by cap {r.get('cap')} "
                             f"(cost $0, rtt {r.get('rtt_ms')}ms)"), "skip"
            else:
                line, tag = f"{ts}  LIVE ERR: {e[:110]}", "err"
        elif s == "live_skip_minsize":
            line, tag = (f"{ts}  skip    depth {r.get('ask_depth')} < 5 sh min "
                         f"(ask {r.get('ask')})"), "skip"
        elif s == "skip_repriced":
            line, tag = f"{ts}  skip    {r.get('aligned')} repriced {r.get('a_prev')}->{r.get('a_now')}", "skip"
        elif s == "settle":
            p = r.get("pnl", 0)
            line, tag = (f"{ts}  SETTLE  {datetime.fromtimestamp(r['ws']).strftime('%H:%M')} "
                         f"{r.get('winner'):4s} n={r.get('n')}  {p:+.2f}$"), ("set" if p >= 0 else "setl")
        elif s == "HALT":
            line, tag = f"{ts}  *** HALT: {r.get('reason')} pnl {r.get('pnl')} ***", "halt"
        else:
            return
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n", tag)
        if int(self.log.index("end-1c").split(".")[0]) > 400:
            self.log.delete("1.0", "50.0")
        self.log.configure(state="disabled")
        self.log.see("end")

    def draw_strip(self, settles):
        c = self.strip
        c.delete("all")
        w = c.winfo_width() or 560
        n = len(settles)
        if not n:
            c.create_text(w / 2, 60, text="no settled windows yet", fill=MUTED, font=(FONT, 10))
            return
        s = settles[-30:]
        mx = max(abs(x.get("pnl", 0)) for x in s) or 1
        bw = (w - 16) / len(s)
        mid = 60
        c.create_line(8, mid, w - 8, mid, fill=LINE)
        for i, x in enumerate(s):
            v = x.get("pnl", 0)
            h = abs(v) / mx * 45
            col = GREEN if v >= 0 else RED
            x0 = 8 + i * bw
            if v >= 0:
                c.create_rectangle(x0, mid - h, x0 + max(bw - 2, 1), mid, fill=col, outline="")
            else:
                c.create_rectangle(x0, mid, x0 + max(bw - 2, 1), mid + h, fill=col, outline="")
        tot = sum(x.get("pnl", 0) for x in s)
        c.create_text(w - 10, 10, text=f"last {len(s)}: {'+' if tot>=0 else ''}${tot:.2f}",
                      fill=GREEN if tot >= 0 else RED, font=(FONT, 10, "bold"), anchor="e")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
