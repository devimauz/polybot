"""Native Windows performance monitor for bot.py. No trading actions."""

import csv
import ctypes
import logging
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from dashboard import Dashboard

BG = "#101217"
PANEL = "#181c24"
LINE = "#2c3340"
TEXT = "#e9edf5"
MUTED = "#8896ac"
GREEN = "#93e0be"
RED = "#f09caa"


def money(value, signed=False):
    if value is None:
        return "--"
    return ("+" if signed and value > 0 else "-" if value < 0 else "") + f"${abs(value):,.2f}"


class App:
    def __init__(self, root, source=None, start_worker=True):
        self.root = root
        self.source = source or Dashboard()
        self.data = None
        self.updated = None
        self.rows = []
        self.sort_column = None
        self.sort_reverse = False
        root.title("BTC Desk - Polymarket Bot Performance")
        root.geometry("1240x930")
        root.minsize(900, 700)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.configure_style()
        self.build()
        self.root.bind("<Configure>", self.resize)
        if start_worker:
            threading.Thread(target=self.source.run, name="account-monitor", daemon=True).start()
        self.tick()

    def configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TButton", background="#252d3a", foreground=TEXT, borderwidth=0,
                        padding=(13, 8), font=("Segoe UI", 9))
        style.map("TButton", background=[("active", "#35445a")])
        style.configure("TCombobox", fieldbackground=PANEL, background=LINE, foreground=TEXT,
                        arrowcolor=MUTED, padding=5, borderwidth=0)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])
        style.configure("Treeview", background=BG, fieldbackground=BG, foreground=TEXT,
                        rowheight=37, borderwidth=0, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=PANEL, foreground=MUTED,
                        font=("Segoe UI", 9), padding=(8, 10), relief="flat")
        style.map("Treeview", background=[("selected", "#30443f")], foreground=[("selected", "#ffffff")])
        style.map("Treeview.Heading", background=[("active", "#293241")])
        style.configure("Vertical.TScrollbar", background=LINE, troughcolor=BG, borderwidth=0,
                        arrowcolor=MUTED)
        style.configure("TEntry", fieldbackground=PANEL, foreground=TEXT, padding=7,
                        insertcolor=TEXT, borderwidth=1)
        self.root.option_add("*TCombobox*Listbox.background", PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)

    def label(self, parent, text="", size=10, color=TEXT, bold=False, **kwargs):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kwargs)

    def rule(self, parent):
        line = tk.Frame(parent, bg=LINE, height=1)
        line.pack(fill="x")
        return line

    def build(self):
        header = tk.Frame(self.root, bg=BG, padx=28, pady=17)
        header.pack(fill="x")
        mark = tk.Canvas(header, width=34, height=34, bg=BG, highlightthickness=0)
        mark.pack(side="left", padx=(0, 10))
        mark.create_rectangle(0, 0, 34, 34, fill=GREEN, outline="")
        mark.create_line(6, 24, 13, 16, 20, 20, 28, 8, fill="#153d30", width=3)
        self.label(header, "BTC DESK", 15, bold=True).pack(side="left")
        self.label(header, "POLYMARKET", 8, MUTED).pack(side="left", padx=20)
        self.connection = self.label(header, "Connecting...", 9, "#d7bd83")
        self.connection.pack(side="right", padx=(16, 0))
        ttk.Button(header, text="Refresh", command=self.refresh).pack(side="right")
        self.rule(self.root)
        content = tk.Frame(self.root, bg=BG, padx=28)
        content.pack(fill="both", expand=True)
        title = tk.Frame(content, bg=BG, pady=14)
        title.pack(fill="x")
        left = tk.Frame(title, bg=BG)
        left.pack(side="left")
        self.label(left, "Trading overview", 23, bold=True).pack(anchor="w")
        self.subtitle = self.label(left, "Loading account data", 9, MUTED)
        self.subtitle.pack(anchor="w", pady=(5, 0))
        right = tk.Frame(title, bg=BG)
        right.pack(side="right")
        self.label(right, "PERFORMANCE SCOPE", 8, MUTED).pack(anchor="w", pady=(0, 5))
        self.scope = tk.StringVar(value="Bot markets")
        scope = ttk.Combobox(right, textvariable=self.scope, values=["Bot markets", "All BTC positions"],
                             state="readonly", width=20)
        scope.pack()
        scope.bind("<<ComboboxSelected>>", lambda _: self.render())
        self.error = self.label(content, "", 9, "#edc58b", anchor="w", justify="left")
        metrics = tk.Frame(content, bg=PANEL)
        metrics.pack(fill="x")
        self.metric_labels = {}
        for i, (key, title, sub) in enumerate([
            ("realized", "REALIZED PROFIT / LOSS", "API-reported realized P&L"),
            ("cash", "CASH BALANCE", "Entire trading wallet"),
            ("value", "POSITION VALUE", "Current BTC holdings"),
            ("winrate", "WIN RATE", "Closed positions only")]):
            metrics.columnconfigure(i, weight=1, uniform="metric")
            box = tk.Frame(metrics, bg=PANEL, padx=18, pady=13)
            box.grid(row=0, column=i, sticky="nsew")
            self.label(box, title, 8, MUTED).pack(anchor="w")
            amount = self.label(box, "--", 25)
            amount.pack(anchor="w", pady=(9, 5))
            caption = self.label(box, sub, 8, MUTED)
            caption.pack(anchor="w")
            self.metric_labels[key] = (amount, caption)
        analysis = tk.Frame(content, bg=BG, pady=12)
        self.analysis = analysis
        analysis.pack(fill="x")
        plot = tk.Frame(analysis, bg=BG)
        plot.pack(side="left", fill="both", expand=True)
        chart_head = tk.Frame(plot, bg=BG)
        chart_head.pack(fill="x")
        self.label(chart_head, "Closed-position performance", 11, bold=True).pack(side="left")
        self.period = tk.StringVar(value="All time")
        periods = ttk.Combobox(chart_head, textvariable=self.period, values=["24 hours", "7 days", "All time"],
                              state="readonly", width=10)
        periods.pack(side="right")
        periods.bind("<<ComboboxSelected>>", lambda _: self.draw_chart())
        self.chart_total = self.label(plot, "--", 22, GREEN)
        self.chart_total.pack(anchor="w", pady=(10, 0))
        self.label(plot, "Cumulative realized P&L on closed positions", 8, MUTED).pack(anchor="w")
        self.canvas = tk.Canvas(plot, height=140, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="x", pady=(10, 0))
        self.canvas.bind("<Configure>", lambda _: self.draw_chart())
        self.canvas.bind("<Motion>", self.chart_hover)
        self.canvas.bind("<Leave>", lambda _: self.canvas.delete("hover"))
        tk.Frame(analysis, bg=LINE, width=1).pack(side="left", fill="y", padx=23)
        strategy = tk.Frame(analysis, bg=BG, width=235)
        strategy.pack(side="right", fill="y")
        self.label(strategy, "BTC strategy", 11, bold=True).pack(anchor="w", pady=(0, 12))
        for name, value in [("Order size", "$10 + fees"), ("UP / DOWN", ">80% / >=80%"),
                            ("Daily spending cap", "None")]:
            row = tk.Frame(strategy, bg=BG)
            row.pack(fill="x", pady=5)
            self.label(row, name, 9, MUTED).pack(side="left")
            self.label(row, value, 9).pack(side="right", padx=(18, 0))
        self.rule(strategy)
        self.label(strategy, "CLAIMABLE BTC VALUE", 8, MUTED).pack(anchor="w", pady=(13, 2))
        self.claimable = self.label(strategy, "--", 20)
        self.claimable.pack(anchor="w")
        self.order_count = self.label(strategy, "-- matched / -- to review", 8, MUTED)
        self.order_count.pack(anchor="w", pady=(8, 5))
        self.analysis_divider = self.rule(content)
        toolbar = tk.Frame(content, bg=BG, pady=12)
        toolbar.pack(fill="x")
        self.tab = tk.StringVar(value="Positions")
        tabs = ttk.Combobox(toolbar, textvariable=self.tab, state="readonly", values=["Positions", "Bot orders"], width=13)
        tabs.pack(side="left")
        tabs.bind("<<ComboboxSelected>>", self.tab_changed)
        self.filter = tk.StringVar(value="All")
        self.filter_box = ttk.Combobox(toolbar, textvariable=self.filter, state="readonly",
                                      values=["All", "Open", "Claimable", "Closed"], width=12)
        self.filter_box.pack(side="left", padx=8)
        self.filter_box.bind("<<ComboboxSelected>>", lambda _: self.render_table())
        ttk.Button(toolbar, text="Export CSV", command=self.export).pack(side="right")
        self.search = tk.StringVar()
        self.search.trace_add("write", lambda *_: self.render_table())
        ttk.Entry(toolbar, textvariable=self.search, width=22).pack(side="right", padx=(8, 14))
        self.label(toolbar, "Search", 9, MUTED).pack(side="right")
        table_frame = tk.Frame(content, bg=BG)
        table_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table_frame, show="headings", selectmode="browse", height=6)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.tag_configure("positive", foreground=GREEN)
        self.tree.tag_configure("negative", foreground=RED)
        self.tree.tag_configure("warning", foreground="#e7be88")
        self.tree.bind("<Double-1>", self.open_market)
        self.table_status = self.label(content, "Waiting for positions...", 8, MUTED)
        self.table_status.pack(side="bottom", anchor="w", pady=9, before=table_frame)
        footer = tk.Frame(content, bg=BG, pady=11)
        footer.pack(side="bottom", fill="x", before=table_frame)
        self.label(footer, "READ-ONLY MONITOR", 8, GREEN).pack(side="left")
        self.label(footer, "Same-position manual trades can affect P&L.  |  Account refresh: 15 sec", 8, MUTED).pack(side="right")

    def resize(self, event):
        if event.widget is not self.root:
            return
        if event.height < 800:
            self.analysis.pack_forget()
        elif not self.analysis.winfo_manager():
            self.analysis.pack(fill="x", before=self.analysis_divider)

    def view(self):
        if not self.data:
            return None
        return self.data.get("bot_view", self.data) if self.scope.get() == "Bot markets" else self.data

    def tick(self):
        with self.source.lock:
            snapshot = self.source.data.copy()
        errors = snapshot.get("errors", [])
        if errors:
            self.error.configure(text="  ".join(errors))
            if not self.error.winfo_manager():
                self.error.pack(fill="x", pady=5, before=self.error.master.winfo_children()[2])
        else:
            self.error.pack_forget()
        self.connection.configure(text="Partial / stale data" if errors else "Connecting..." if snapshot.get("loading") else "Account connected",
                                  fg="#e7be88" if errors else MUTED if snapshot.get("loading") else GREEN)
        if snapshot.get("updated_at") != self.updated and "metrics" in snapshot:
            self.data = snapshot
            self.updated = snapshot["updated_at"]
            self.render()
        self.root.after(1000, self.tick)

    def render(self):
        view = self.view()
        if not view:
            return
        m = view["metrics"]
        wallet = self.data.get("wallet", "")
        self.subtitle.configure(text=f"BTC / 5 min   |   {wallet[:6]}...{wallet[-4:]}   |   Updated {datetime.fromtimestamp(self.updated).strftime('%H:%M:%S')}")
        for key, value in (("realized", m["realized"]), ("cash", self.data.get("cash")), ("value", m["position_value"])):
            self.metric_labels[key][0].configure(text=money(value, key == "realized"),
                                                fg=(GREEN if value >= 0 else RED) if key == "realized" else TEXT)
        self.metric_labels["value"][1].configure(text="Unrealized " + money(m["unrealized"], True),
                                                fg=GREEN if m["unrealized"] >= 0 else RED)
        decided = m["wins"] + m["losses"]
        self.metric_labels["winrate"][0].configure(text=f"{100*m['wins']/decided:.1f}%" if decided else "--")
        self.metric_labels["winrate"][1].configure(text=f"{m['wins']} won / {m['losses']} lost - closed")
        self.claimable.configure(text=money(m["claimable"]))
        self.order_count.configure(text=f"{m['matched']} matched / {m['review']} to review")
        self.render_table()
        self.draw_chart()

    def tab_changed(self, _=None):
        self.filter_box.configure(values=["All", "Open", "Claimable", "Closed"] if self.tab.get() == "Positions"
                                  else ["All", "Matched", "Rejected", "Unknown", "Reserved", "Review"])
        self.filter.set("All")
        self.sort_column = None
        self.render_table()

    def sort(self, column):
        self.sort_reverse = not self.sort_reverse if self.sort_column == column else False
        self.sort_column = column
        self.render_table()

    def render_table(self):
        view = self.view()
        if not view:
            return
        positions = self.tab.get() == "Positions"
        fields = [("title", "Market", 320), ("outcome", "Side", 65), ("shares", "Shares", 85),
                  ("avg_price", "Avg. price", 90), ("value", "Value", 85), ("pnl", "Total P&L", 95),
                  ("state", "Status", 100)] if positions else [
                      ("timestamp", "Market window", 170), ("slug", "Market", 300), ("outcome", "Side", 65),
                      ("amount", "Amount + fees", 110), ("status", "Status", 100)]
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        self.tree.configure(columns=[f[0] for f in fields])
        for key, label, width in fields:
            self.tree.heading(key, text=label, command=lambda c=key: self.sort(c))
            self.tree.column(key, width=width, minwidth=55, stretch=key in ("title", "slug"),
                             anchor="w" if key in ("title", "slug", "timestamp") else "e")
        self.rows = [p for p in view["positions" if positions else "orders"]
                     if (self.filter.get() == "All" or (p.get("state") or p.get("status", "")).lower() == self.filter.get().lower())
                     and self.search.get().lower() in (p.get("title", "") + p["slug"] + p["outcome"]).lower()]
        if self.sort_column:
            self.rows.sort(key=lambda p: p.get(self.sort_column, ""), reverse=self.sort_reverse)
        for index, p in enumerate(self.rows):
            vals = []
            for key, _, _ in fields:
                v = p.get(key, "")
                if key in ("value", "pnl", "amount"):
                    v = money(v, key == "pnl")
                elif key == "avg_price":
                    v = f"{v*100:.1f} cents"
                elif key == "shares":
                    v = f"{v:,.3f}"
                elif key == "timestamp":
                    v = datetime.fromtimestamp(v).strftime("%b %d %H:%M")
                elif key == "title":
                    v = v.replace("Bitcoin Up or Down - ", "")
                vals.append(v)
            tag = ("positive" if p["pnl"] > 0 else "negative" if p["pnl"] < 0 else "") if positions else ("warning" if p["status"] in ("unknown", "reserved", "review") else "")
            self.tree.insert("", "end", iid=str(index), values=vals, tags=(tag,))
        if selected and self.tree.exists(selected[0]):
            self.tree.selection_set(selected[0])
        self.table_status.configure(text=f"{len(self.rows)} {'positions' if positions else 'orders'}" if self.rows else "No matching records")

    def draw_chart(self):
        self.canvas.delete("all")
        view = self.view()
        self.points = []
        if not view:
            return
        cutoff = time.time() - {"24 hours": 86400, "7 days": 604800}.get(self.period.get(), time.time())
        rows = sorted((p for p in view["positions"] if p["state"] == "closed" and p["timestamp"] >= cutoff), key=lambda p: p["timestamp"])
        running = 0
        values = [0]
        for p in rows:
            running += p["realized"]
            values.append(running)
        self.chart_total.configure(text=money(running, True), fg=GREEN if running >= 0 else RED)
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if not rows:
            self.canvas.create_text(width/2, height/2, text="No closed positions in this period", fill=MUTED, font=("Segoe UI", 10))
            return
        lo, hi = min(values), max(values)
        pad = max((hi-lo)*.15, 1)
        lo, hi = lo-pad, hi+pad
        left, right, top, bottom = 4, max(70, width-72), 10, height-25
        for i in range(4):
            y = top + i*(bottom-top)/3
            value = hi-i*(hi-lo)/3
            self.canvas.create_line(left, y, right, y, fill=LINE)
            self.canvas.create_text(right+8, y, text=money(value), anchor="w", fill=MUTED, font=("Segoe UI", 8))
        coords = []
        for i, v in enumerate(values):
            x, y = left+(right-left)*i/(len(values)-1), top+(hi-v)/(hi-lo)*(bottom-top)
            coords.extend((x, y))
            stamp = rows[max(0, i-1)]["timestamp"]
            self.points.append((x, y, v, stamp))
        self.canvas.create_line(*coords, fill=GREEN if running >= 0 else RED, width=2)
        for index, anchor in ((0, "w"), (len(rows)-1, "e")):
            self.canvas.create_text(left if index == 0 else right, height-5,
                                    text=datetime.fromtimestamp(rows[index]["timestamp"]).strftime("%b %d %H:%M"),
                                    anchor=anchor, fill=MUTED, font=("Segoe UI", 8))

    def chart_hover(self, event):
        self.canvas.delete("hover")
        if not getattr(self, "points", []):
            return
        x, y, v, stamp = min(self.points, key=lambda p: abs(p[0]-event.x))
        self.canvas.create_oval(x-3,y-3,x+3,y+3,fill=TEXT,outline="",tags="hover")
        label = datetime.fromtimestamp(stamp).strftime("%b %d %H:%M") + "   " + money(v, True)
        self.canvas.create_text(min(max(x,100),self.canvas.winfo_width()-100), 10, text=label,
                                fill=TEXT, font=("Segoe UI", 9, "bold"), tags="hover")

    def export(self):
        if not self.rows:
            messagebox.showinfo("Export", "No rows to export.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".csv",
                                          initialfile="btc-performance.csv", filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)

    def open_market(self, _=None):
        selected = self.tree.selection()
        if selected:
            webbrowser.open("https://polymarket.com/event/" + self.rows[int(selected[0])]["slug"])

    def refresh(self):
        self.connection.configure(text="Refreshing...", fg=MUTED)
        threading.Thread(target=self.source.refresh, daemon=True).start()

    def close(self):
        self.source.stop.set()
        self.root.destroy()


def main():
    logging.getLogger("py_clob_client_v2").setLevel(logging.CRITICAL)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
