"""Desktop viewer for per-window Polymarket BTC 5-minute JSONL logs."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk


ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECTORY = ROOT / "price_logs" / "polymarket"
BG, PANEL, PANEL_2 = "#0d1117", "#161b22", "#202733"
TEXT, MUTED, GRID = "#e6edf3", "#8b98a9", "#303947"
UP_BID, UP_ASK = "#2dd4bf", "#86efac"
DOWN_BID, DOWN_ASK, COINBASE = "#f97393", "#fbbf74", "#60a5fa"

COLUMNS = ("time", "elapsed", "up_bid", "up_ask", "up_bid_size", "up_ask_size",
           "down_bid", "down_ask", "down_bid_size", "down_ask_size", "coinbase", "cb_age")


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def snapshot_rows(rows):
    return [row for row in rows if isinstance(row, dict) and row.get("event") == "snapshot"]


def winner_of(rows):
    for row in reversed(rows):
        if row.get("event") == "resolution" and row.get("winner") in ("Up", "Down"):
            return row["winner"]
    return None


def row_values(row):
    stamp = str(row.get("ts", ""))
    try:
        stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00")).strftime("%H:%M:%S.%f")[:-3]
    except ValueError:
        pass
    up, down, cb = row.get("Up") or {}, row.get("Down") or {}, row.get("coinbase") or {}

    def fmt(value, places=3):
        value = number(value)
        return "—" if value is None else f"{value:.{places}f}"

    return (stamp, fmt(row.get("seconds_into_window"), 2), fmt(up.get("bid")),
            fmt(up.get("ask")), fmt(up.get("bid_size"), 2), fmt(up.get("ask_size"), 2),
            fmt(down.get("bid")), fmt(down.get("ask")), fmt(down.get("bid_size"), 2),
            fmt(down.get("ask_size"), 2), fmt(cb.get("price"), 2), fmt(cb.get("age_ms"), 0))


class IncrementalJsonl:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        self.offset = 0
        self.pending = b""
        self.identity = None
        self.modified = None
        self.skipped = 0

    def refresh(self):
        with self.path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            reset = (identity != self.identity or stat.st_size < self.offset or
                     (stat.st_size == self.offset and stat.st_mtime_ns != self.modified))
            if reset:
                self.rows.clear()
                self.offset, self.pending, self.skipped = 0, b"", 0
            self.identity, self.modified = identity, stat.st_mtime_ns
            stream.seek(self.offset)
            data = stream.read(stat.st_size - self.offset)
            self.offset += len(data)
        parts = (self.pending + data).split(b"\n")
        self.pending = parts.pop()
        for line in parts:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record is not an object")
                self.rows.append(row)
            except (ValueError, UnicodeDecodeError):
                self.skipped += 1
        return reset or bool(data)


class Viewer:
    def __init__(self, root, directory):
        self.root = root
        self.directory = Path(directory)
        self.files = {}
        self.reader = None
        self.snapshots = []
        self.page, self.page_size = 0, 500

        root.title("BTC 5m · Polymarket Log Viewer")
        root.geometry("1500x930")
        root.minsize(1050, 700)
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=TEXT, rowheight=25)
        style.configure("Treeview.Heading", background=PANEL_2, foreground=MUTED)
        style.map("Treeview", background=[("selected", "#264f78")])

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="BTC 5-MINUTE LOGS", font=("Segoe UI", 20, "bold")).pack(side="left")
        ttk.Button(header, text="Choose folder…", command=self.choose_folder).pack(side="right")
        ttk.Button(header, text="Refresh", command=self.refresh).pack(side="right", padx=8)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(12, 8))
        ttk.Label(controls, text="Window", style="Muted.TLabel").pack(side="left")
        self.file_var = tk.StringVar()
        self.file_box = ttk.Combobox(controls, textvariable=self.file_var, state="readonly", width=43)
        self.file_box.pack(side="left", padx=8)
        self.file_box.bind("<<ComboboxSelected>>", self.select_file)
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Follow newest window", variable=self.follow).pack(side="left", padx=10)
        self.path_label = ttk.Label(controls, text=str(self.directory), style="Muted.TLabel")
        self.path_label.pack(side="right")

        cards = ttk.Frame(outer)
        cards.pack(fill="x", pady=5)
        self.card_labels = {}
        for key, title, color in (("up", "UP BID / ASK", UP_BID),
                                  ("down", "DOWN BID / ASK", DOWN_BID),
                                  ("coinbase", "COINBASE BTC-USD", COINBASE),
                                  ("status", "WINDOW", TEXT)):
            card = ttk.Frame(cards, style="Card.TFrame", padding=12)
            card.pack(side="left", fill="x", expand=True, padx=(0, 8))
            ttk.Label(card, text=title, foreground=color, background=PANEL,
                      font=("Segoe UI", 9, "bold")).pack(anchor="w")
            value = ttk.Label(card, text="—", background=PANEL, foreground=TEXT,
                              font=("Consolas", 16, "bold"))
            value.pack(anchor="w", pady=(4, 0))
            self.card_labels[key] = value

        self.summary = ttk.Label(outer, style="Muted.TLabel")
        self.summary.pack(anchor="w", pady=(3, 2))
        self.canvas = tk.Canvas(outer, height=330, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="x", pady=(4, 9))
        self.canvas.bind("<Configure>", lambda _: self.draw_chart())

        table_frame = ttk.Frame(outer)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, columns=COLUMNS, show="headings", selectmode="browse")
        labels = {"time": "UTC time", "elapsed": "Window sec", "cb_age": "CB age ms"}
        for col in COLUMNS:
            self.table.heading(col, text=labels.get(col, col.replace("_", " ").title()))
            self.table.column(col, width=118 if col == "time" else 95, anchor="e", stretch=False)
        ybar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        xbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.table.bind("<Double-1>", self.show_json)
        self.table.bind("<Return>", self.show_json)

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(8, 0))
        ttk.Button(footer, text="← Newer", command=lambda: self.change_page(-1)).pack(side="left")
        ttk.Button(footer, text="Older →", command=lambda: self.change_page(1)).pack(side="left", padx=6)
        self.page_label = ttk.Label(footer, style="Muted.TLabel")
        self.page_label.pack(side="left", padx=8)
        ttk.Label(footer, text="Newest first · Double-click a row for full JSON",
                  style="Muted.TLabel").pack(side="right")

        self.refresh()
        root.after(1000, self.poll)

    def choose_folder(self):
        chosen = filedialog.askdirectory(initialdir=self.directory)
        if chosen:
            self.directory = Path(chosen)
            self.path_label.configure(text=chosen)
            self.reader = None
            self.file_var.set("")
            self.refresh()

    def select_file(self, _=None):
        path = self.files.get(self.file_var.get())
        self.reader = IncrementalJsonl(path) if path else None
        self.page = 0
        self.refresh_data()

    def refresh(self):
        try:
            paths = sorted(self.directory.glob("btc-updown-5m-*.jsonl"), reverse=True)
            self.files = {path.name: path for path in paths if path.is_file()}
            names = list(self.files)
            self.file_box.configure(values=names)
            desired = names[0] if names and (self.follow.get() or self.file_var.get() not in self.files) else self.file_var.get()
            if desired != self.file_var.get() or self.reader is None:
                self.file_var.set(desired or "")
                self.select_file()
            else:
                self.refresh_data()
        except OSError as exc:
            self.summary.configure(text=f"Cannot read logs: {exc}")

    def refresh_data(self):
        if not self.reader:
            self.snapshots = []
            self.summary.configure(text="No per-window JSONL files found")
            self.render()
            return
        changed = self.reader.refresh()
        if changed:
            self.snapshots = snapshot_rows(self.reader.rows)
            self.render()

    def poll(self):
        self.refresh()
        self.root.after(1000, self.poll)

    def render(self):
        self.update_cards()
        self.draw_chart()
        self.render_table()

    def update_cards(self):
        latest = self.snapshots[-1] if self.snapshots else {}
        up, down, cb = latest.get("Up") or {}, latest.get("Down") or {}, latest.get("coinbase") or {}
        fmt = lambda value, p=3: "—" if number(value) is None else f"{float(value):.{p}f}"
        self.card_labels["up"].configure(text=f"{fmt(up.get('bid'))} / {fmt(up.get('ask'))}")
        self.card_labels["down"].configure(text=f"{fmt(down.get('bid'))} / {fmt(down.get('ask'))}")
        self.card_labels["coinbase"].configure(text=("—" if number(cb.get("price")) is None
                                                      else f"${float(cb['price']):,.2f}"))
        elapsed = number(latest.get("seconds_into_window"))
        winner = winner_of(self.reader.rows) if self.reader else None
        status = f"{elapsed:.1f}s / 300s" if elapsed is not None else "—"
        if winner:
            status = f"WINNER: {winner}"
        self.card_labels["status"].configure(text=status)
        duration = 0 if not self.snapshots else max(0, number(self.snapshots[-1].get("seconds_into_window")) or 0)
        hz = (len(self.snapshots) - 1) / duration if duration > 0 else 0
        full = any("bids" in (row.get("Up") or {}) for row in self.snapshots[:5])
        self.summary.configure(text=(f"{len(self.snapshots):,} snapshots · {hz:.1f} samples/s · "
                                     f"{'full depth' if full else 'top of book'} · "
                                     f"{self.reader.skipped if self.reader else 0} malformed records"))

    def draw_chart(self):
        c = self.canvas
        c.delete("all")
        width, height = c.winfo_width(), c.winfo_height()
        if width < 100 or not self.snapshots:
            c.create_text(width / 2, height / 2, text="No snapshots", fill=MUTED)
            return
        left, right = 58, width - 70
        p_top, p_bottom = 30, 202
        cb_top, cb_bottom = 240, height - 27
        for value in (0, .25, .5, .75, 1):
            y = p_bottom - value * (p_bottom - p_top)
            c.create_line(left, y, right, y, fill=GRID)
            c.create_text(left - 10, y, text=f"{value:.2f}", anchor="e", fill=MUTED)
        for second in (0, 60, 120, 180, 240, 300):
            x = left + second / 300 * (right - left)
            c.create_line(x, p_top, x, p_bottom, fill=GRID)
            c.create_line(x, cb_top, x, cb_bottom, fill=GRID)
            c.create_text(x, height - 10, text=f"{second}s", fill=MUTED)

        series = (("Up", "bid", UP_BID), ("Up", "ask", UP_ASK),
                  ("Down", "bid", DOWN_BID), ("Down", "ask", DOWN_ASK))
        for group, key, color in series:
            points = []
            for row in self.snapshots:
                sec, value = number(row.get("seconds_into_window")), number((row.get(group) or {}).get(key))
                if sec is not None and value is not None and 0 <= value <= 1:
                    points.extend((left + min(300, max(0, sec)) / 300 * (right - left),
                                   p_bottom - value * (p_bottom - p_top)))
            if len(points) >= 4:
                c.create_line(*points, fill=color, width=2)

        cb_points = [(number(row.get("seconds_into_window")), number((row.get("coinbase") or {}).get("price")))
                     for row in self.snapshots]
        cb_points = [(sec, price) for sec, price in cb_points if sec is not None and price is not None]
        if cb_points:
            lo, hi = min(p for _, p in cb_points), max(p for _, p in cb_points)
            pad = max((hi - lo) * .08, 1.0)
            lo, hi = lo - pad, hi + pad
            points = []
            for sec, price in cb_points:
                points.extend((left + min(300, max(0, sec)) / 300 * (right - left),
                               cb_bottom - (price - lo) / (hi - lo) * (cb_bottom - cb_top)))
            if len(points) >= 4:
                c.create_line(*points, fill=COINBASE, width=2)
            c.create_text(right + 8, cb_top, text=f"${hi:,.0f}", anchor="w", fill=MUTED)
            c.create_text(right + 8, cb_bottom, text=f"${lo:,.0f}", anchor="w", fill=MUTED)
        c.create_text(left, 13, anchor="w", text="UP bid / ask", fill=UP_BID)
        c.create_text(left + 120, 13, anchor="w", text="DOWN bid / ask", fill=DOWN_BID)
        c.create_text(left, cb_top - 12, anchor="w", text="Coinbase BTC-USD", fill=COINBASE)

    def render_table(self):
        self.table.delete(*self.table.get_children())
        total_pages = max(1, (len(self.snapshots) + self.page_size - 1) // self.page_size)
        self.page = min(self.page, total_pages - 1)
        end = len(self.snapshots) - self.page * self.page_size
        start = max(0, end - self.page_size)
        for index in range(end - 1, start - 1, -1):
            self.table.insert("", "end", iid=str(index), values=row_values(self.snapshots[index]))
        self.page_label.configure(text=f"Page {self.page + 1} / {total_pages}")

    def change_page(self, delta):
        total_pages = max(1, (len(self.snapshots) + self.page_size - 1) // self.page_size)
        self.page = max(0, min(self.page + delta, total_pages - 1))
        self.render_table()

    def show_json(self, _=None):
        selected = self.table.selection()
        if not selected:
            return
        row = self.snapshots[int(selected[0])]
        window = tk.Toplevel(self.root)
        window.title("Snapshot · Full JSON")
        window.geometry("950x650")
        text = tk.Text(window, bg=PANEL, fg=TEXT, insertbackground=TEXT,
                       font=("Consolas", 10), padx=12, pady=12, wrap="none")
        text.pack(fill="both", expand=True)
        text.insert("1.0", json.dumps(row, indent=2, ensure_ascii=False))
        text.configure(state="disabled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    root = tk.Tk()
    Viewer(root, args.directory)
    root.mainloop()


if __name__ == "__main__":
    main()
