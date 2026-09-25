"""Read-only, dependency-free desktop viewer for price_logs/*.jsonl."""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
BG, PANEL, TEXT, MUTED = "#101217", "#181c24", "#e9edf5", "#8896ac"
SERIES = {"buy_price": "#93e0be", "bid_price": "#80baff", "sell_price": "#c5a3ff", "fill_price": "#ffd080"}
COLUMNS = ("timestamp", "asset", "market", "outcome", "mode", "event", "buy_price", "bid_price", "sell_price", "midpoint", "buy_limit", "fill_price")


class LogReader:
    """Read appended bytes only; leave incomplete final records for the next poll."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        self.offset = 0
        self.pending = b""
        self.skipped = 0
        self.identity = None
        self.modified = None

    def refresh(self):
        with self.path.open("rb") as stream:
            import os
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            reset = (identity != self.identity or stat.st_size < self.offset or
                     (stat.st_size == self.offset and stat.st_mtime_ns != self.modified))
            if reset:
                self.rows.clear()
                self.offset = self.skipped = 0
                self.pending = b""
            self.identity, self.modified = identity, stat.st_mtime_ns
            stream.seek(self.offset)
            data = stream.read(stat.st_size - self.offset)
            self.offset += len(data)
        parts = (self.pending + data).split(b"\n")
        self.pending = parts.pop()
        # Also accept a complete final JSON object without a trailing newline.
        if self.pending.strip():
            try:
                json.loads(self.pending)
            except (ValueError, UnicodeDecodeError):
                pass
            else:
                parts.append(self.pending)
                self.pending = b""
        for line in parts:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected an object")
                self.rows.append(row)
            except (ValueError, UnicodeDecodeError):
                self.skipped += 1
        return reset or bool(data)


def filter_rows(rows, market="All", outcome="All", event="All", mode="All", search=""):
    filters = {"market": market, "outcome": outcome, "event": event, "mode": mode}
    return [r for r in rows if all(v == "All" or str(r.get(k, "")) == v for k, v in filters.items())
            and (not search or search.lower() in json.dumps(r).lower())]


class Viewer:
    def __init__(self, root, directory):
        self.root, self.directory = root, Path(directory)
        self.reader = None
        self.filtered = []
        self.page = 0
        self.page_size = 500
        self.files = {}
        root.title("Price Logs · Polymarket")
        root.geometry("1400x880")
        root.minsize(1000, 650)
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL)
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=27)
        style.configure("Treeview.Heading", background=BG, foreground=MUTED)
        style.map("Treeview", background=[("selected", "#284963")])
        outer = ttk.Frame(root, padding=20)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="PRICE LOGS", font=("Segoe UI", 21, "bold")).pack(side="left")
        ttk.Button(header, text="Choose folder…", command=self.choose_folder).pack(side="right")
        ttk.Button(header, text="Export filtered JSONL…", command=self.export).pack(side="right", padx=10)
        self.folder_label = ttk.Label(outer, text=str(self.directory), foreground=MUTED)
        self.folder_label.pack(anchor="w", pady=(4, 16))
        bar = ttk.Frame(outer)
        bar.pack(fill="x")
        self.file = tk.StringVar()
        self.file_box = self.combo(bar, "File", self.file, 15, self.load_file)
        self.filters, self.boxes = {}, {}
        for name, width in (("market", 32), ("outcome", 9), ("event", 18), ("mode", 10)):
            var = self.filters[name] = tk.StringVar(value="All")
            self.boxes[name] = self.combo(bar, name.title(), var, width, self.apply_filters)
        searchbar = ttk.Frame(outer)
        searchbar.pack(fill="x", pady=12)
        ttk.Label(searchbar, text="Search").pack(side="left")
        self.search = tk.StringVar()
        entry = ttk.Entry(searchbar, textvariable=self.search, width=42)
        entry.pack(side="left", padx=10)
        entry.bind("<Return>", self.apply_filters)
        ttk.Button(searchbar, text="Apply", command=self.apply_filters).pack(side="left")
        self.live = tk.BooleanVar(value=True)
        ttk.Checkbutton(searchbar, text="Auto-refresh · 2s", variable=self.live).pack(side="right")
        ttk.Button(searchbar, text="Refresh", command=self.refresh).pack(side="right", padx=12)
        self.summary = ttk.Label(outer, foreground=MUTED)
        self.summary.pack(anchor="w", pady=(0, 8))
        ttk.Label(outer, text="Buy: green   Bid: blue   Sell: purple   Fill: gold dots  •  Prices in $ per share  •  Times in UTC", foreground=MUTED).pack(anchor="w")
        self.canvas = tk.Canvas(outer, height=220, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="x", pady=8)
        self.canvas.bind("<Configure>", lambda _: self.draw_chart())
        table_frame = ttk.Frame(outer)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, columns=COLUMNS, show="headings", selectmode="browse")
        for col in COLUMNS:
            self.table.heading(col, text=col.replace("_", " ").title())
            self.table.column(col, width=225 if col == "market" else 195 if col == "timestamp" else 110, stretch=False)
        vertical = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.table.bind("<Double-1>", self.details)
        self.table.bind("<Return>", self.details)
        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))
        ttk.Button(footer, text="← Newer", command=lambda: self.change_page(-1)).pack(side="left")
        ttk.Button(footer, text="Older →", command=lambda: self.change_page(1)).pack(side="left", padx=8)
        self.page_label = ttk.Label(footer)
        self.page_label.pack(side="left")
        ttk.Label(footer, text="Newest first · Double-click a row for full JSON", foreground=MUTED).pack(side="right")
        self.refresh()
        root.after(2000, self.poll)

    def combo(self, parent, label, variable, width, callback):
        frame = ttk.Frame(parent)
        frame.pack(side="left", padx=(0, 10))
        ttk.Label(frame, text=label, foreground=MUTED).pack(anchor="w", pady=(0, 5))
        box = ttk.Combobox(frame, textvariable=variable, width=width, state="readonly")
        box.pack()
        box.bind("<<ComboboxSelected>>", callback)
        return box

    def choose_folder(self):
        directory = filedialog.askdirectory(initialdir=self.directory)
        if directory:
            self.directory = Path(directory)
            self.folder_label.configure(text=directory)
            self.file.set("")
            self.reader = None
            self.refresh()

    def load_file(self, _=None):
        path = self.files.get(self.file.get())
        self.reader = LogReader(path) if path else None
        for var in self.filters.values():
            var.set("All")
        self.search.set("")
        self.page = 0
        self.refresh()

    def refresh(self):
        try:
            self.files = {p.name: p for p in sorted(self.directory.glob("*.jsonl")) if p.is_file()}
            self.file_box.configure(values=list(self.files))
            if self.files and self.file.get() not in self.files:
                self.file.set(next(iter(self.files), ""))
                self.load_file()
                if self.reader:
                    return
            if not self.files:
                self.file.set("")
                self.reader = None
            if self.reader:
                changed = self.reader.refresh()
                for name, box in self.boxes.items():
                    box.configure(values=["All"] + sorted({str(r[name]) for r in self.reader.rows if r.get(name) is not None}))
                if changed:
                    self.apply_filters(reset_page=False)
            else:
                self.apply_filters()
        except OSError as error:
            self.summary.configure(text=f"Cannot read logs: {error}")

    def poll(self):
        if self.live.get():
            self.refresh()
        self.root.after(2000, self.poll)

    def apply_filters(self, _=None, reset_page=True):
        if reset_page:
            self.page = 0
        rows = self.reader.rows if self.reader else []
        self.filtered = filter_rows(rows, **{k: v.get() for k, v in self.filters.items()}, search=self.search.get().strip())
        self.summary.configure(text=f"{len(self.filtered):,} matching / {len(rows):,} records  ·  {self.reader.skipped if self.reader else 0:,} malformed lines skipped" + ("  ·  No JSONL files in this folder" if not self.reader else ""))
        self.render_table()
        self.draw_chart()

    def change_page(self, delta):
        self.page = max(0, min(self.page + delta, max(0, (len(self.filtered) - 1) // self.page_size)))
        self.render_table()

    def render_table(self):
        self.page = min(self.page, max(0, (len(self.filtered) - 1) // self.page_size))
        self.table.delete(*self.table.get_children())
        end = len(self.filtered) - self.page * self.page_size
        for index in range(end - 1, max(-1, end - self.page_size - 1), -1):
            row = self.filtered[index]
            self.table.insert("", "end", iid=str(index), values=["—" if row.get(k) is None else str(row[k]) for k in COLUMNS])
        self.page_label.configure(text=f"Page {self.page + 1} / {max(1, (len(self.filtered) + self.page_size - 1) // self.page_size)}")

    def draw_chart(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        # Plot one market/outcome at a time so unrelated contracts never connect.
        if not self.filtered:
            c.create_text(w / 2, h / 2, text="No matching price records", fill=MUTED)
            return
        latest = self.filtered[-1]
        rows = [r for r in self.filtered if (r.get("market"), r.get("outcome")) == (latest.get("market"), latest.get("outcome"))]
        points = []
        for r in rows:
            try:
                stamp = datetime.fromisoformat(str(r.get("timestamp", "")).replace("Z", "+00:00")).timestamp()
                points.append((stamp, r))
            except (ValueError, OverflowError, OSError):
                continue
        points.sort(key=lambda item: item[0])
        c.create_text(55, 16, anchor="w", text=f"{latest.get('market', 'Unknown market')} · {latest.get('outcome', '')} · latest matching contract", fill=TEXT)
        if not points:
            return
        left, right, top, bottom = 55, max(65, w - 25), 40, h - 30
        start, end = points[0][0], points[-1][0]
        for n in range(5):
            value = n / 4
            y = bottom - value * (bottom - top)
            c.create_line(left, y, right, y, fill="#2c3340")
            c.create_text(left - 10, y, text=f"{value:.2f}", anchor="e", fill=MUTED)
        for stamp, anchor, x in ((start, "w", left), (end, "e", right)):
            from datetime import timezone
            c.create_text(x, h - 12, text=datetime.fromtimestamp(stamp, timezone.utc).strftime("%H:%M:%S UTC"), anchor=anchor, fill=MUTED)
        for key, color in SERIES.items():
            previous = None
            for stamp, row in points:
                try:
                    value = float(row.get(key))
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError()
                except (ValueError, TypeError):
                    previous = None
                    continue
                x = left + (stamp - start) / (end - start or 1) * (right - left)
                y = bottom - value * (bottom - top)
                if key == "fill_price" or len(points) == 1:
                    c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline=color)
                elif previous:
                    c.create_line(*previous, x, y, fill=color, width=2, dash=(4, 3) if key == "sell_price" else ())
                previous = (x, y)

    def details(self, _=None):
        selection = self.table.selection()
        if not selection:
            return
        window = tk.Toplevel(self.root)
        window.title("Price record · Full JSON")
        box = tk.Text(window, width=100, height=25, bg=PANEL, fg=TEXT, padx=15, pady=15)
        box.pack(fill="both", expand=True)
        box.insert("1.0", json.dumps(self.filtered[int(selection[0])], indent=2, ensure_ascii=False))
        box.configure(state="disabled")

    def export(self):
        filename = filedialog.asksaveasfilename(defaultextension=".jsonl", initialfile="filtered_prices.jsonl", filetypes=[("JSON Lines", "*.jsonl")])
        if filename:
            path = Path(filename).resolve()
            if path in [p.resolve() for p in self.files.values()]:
                messagebox.showerror("Export", "Choose a new file outside the source logs.")
                return
            try:
                with path.open("w", encoding="utf-8") as stream:
                    for row in self.filtered:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError as error:
                messagebox.showerror("Export failed", str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "price_logs")
    args = parser.parse_args()
    root = tk.Tk()
    Viewer(root, args.directory)
    root.mainloop()


if __name__ == "__main__":
    main()
