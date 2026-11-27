"""Focused window-level test of the best surface leader cells (correct leader semantics)."""
from backtest_lead import run_lead
from backtest import stats, items, N

train = set(range(0, N//2)); test = set(range(N//2, N))
cells = [
    (30, 90, 0.90, 0.93),
    (30, 90, 0.85, 0.90),
    (30, 90, 0.88, 0.95),
    (120, 150, 0.80, 0.85),
    (150, 180, 0.93, 0.95),
    (120, 180, 0.80, 0.93),
    (90, 180, 0.85, 0.93),
]
for c in cells:
    a = stats(run_lead(*c, subset=train)); b = stats(run_lead(*c, subset=test))
    f = stats(run_lead(*c))
    print(f"T{c[:2]} L{c[2:]} | full n={f['n']:>3} net={f['net']:+8.2f} t={f['t']:+5.2f} wr={f['wr']:.3f} roi={f['roi']:+.1f}% conc5={f['conc5']:.2f} | "
          f"train n={a['n']:>3} net={a['net']:+8.2f} | test n={b['n']:>3} net={b['net']:+8.2f} wr={b['wr']:.3f}")