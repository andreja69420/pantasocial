"""Plan rule at Flashscore Serbian-region prices, 2024-2026, at the opening price, along the open->close path, and the proposed fixes.
usage: run.py SIDES_DIR"""
import sys, numpy as np, pandas as pd
D = sys.argv[1]
def shin(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    lo, hi = np.zeros_like(a), np.full_like(a, 0.5); Bs = a + b
    f = lambda z, x: (np.sqrt(z ** 2 + 4 * (1 - z) * x ** 2 / Bs) - z) / (2 * (1 - z))
    for _ in range(50):
        z = (lo + hi) / 2; s = f(z, a) + f(z, b)
        hi, lo = np.where(s < 1, z, hi), np.where(s < 1, lo, z)
    return f((lo + hi) / 2, a)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
TOUR_LVL = {"G", "M", "A", "F", "C", "PM", "P", "I", "O"}   # no ITF, no Davis/BJK
def pick(S, t, ev_min, cap=5.0, guard=None, no_w125=False):
    o = np.exp((1 - t) * np.log(S.o) + t * np.log(S.c)); oo = np.exp((1 - t) * np.log(S.oo) + t * np.log(S.co))
    fair = shin(1 / o.values, 1 / oo.values)
    ev = 1 / (1 + np.exp(-(0.44 * lg(S.q.values) + 0.64 * lg(fair)))) * o.values - 1
    X = S.assign(px=o.values, ev=ev)
    m = (X.ev > ev_min) & (X.px <= cap)
    if guard is not None:
        m &= X.px <= X.o * (1 + guard)
    if no_w125:
        m &= ~((X.tour == "wta") & (X.lvl == "C"))
    X = X[m].sort_values("px", ascending=False).drop_duplicates(["tour", "index"])   # best price, one side per match
    X["pnl"] = np.where(X.stands, np.where(X.won, X.px - 1, -1.0), 0.0)
    return X
def stats(X):
    n = len(X)
    if n == 0: return dict(bets=0, roi=np.nan, lo90=np.nan, avg_odds=np.nan)
    r = X.pnl.mean(); se = X.pnl.std(ddof=1) / np.sqrt(n)
    return dict(bets=n, roi=round(100 * r, 1), lo90=round(100 * (r - 1.645 * se), 1), avg_odds=round(X.px.mean(), 2))
rows = []
for y in (2024, 2025, 2026):
    S = pd.read_parquet(f"{D}/sides_book_{y}.parquet")
    S = S[S.lvl.astype(str).isin(TOUR_LVL) & (S.c > 1) & (S.co > 1)]
    books = S.book.nunique()
    for name, kw in [("plan EV>5%", {}), ("EV>8%", dict(ev_min=0.08)), ("EV>5%, bez WTA 125", dict(no_w125=True)),
                     ("EV>8%, bez WTA 125", dict(ev_min=0.08, no_w125=True))]:
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            kw2 = dict(ev_min=0.05); kw2.update(kw)
            rows.append(dict(year=y, books=books, rule=name, t=t, **stats(pick(S, t, **kw2))))
        if True:   # guard: only bet while current price <= open*1.03 (meaningful for t>0)
            for t in (0.25, 0.5):
                kw2 = dict(ev_min=0.05, guard=0.03); kw2.update(kw)
                rows.append(dict(year=y, books=books, rule=name + ", kvota<=otv.+3%", t=t, **stats(pick(S, t, **kw2))))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.to_string(index=False))
R.to_csv("fs_years.csv", index=False)
