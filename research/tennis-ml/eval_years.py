"""Plan rule (blend EV>5%, odds<=5, best price among the run's books, flat 1 unit) per level group for several
eval_levels2026.py --bets=... runs (e.g. other seasons or book subsets): bets, ROI, lower 90% bound, units, bets a day
without ITF, and the same without ITF at EV>10% / EV>15%.

usage: eval_years.py NAME=BETS_PARQUET=DAYS [NAME=BETS_PARQUET=DAYS ...]
"""
import sys, re, numpy as np, pandas as pd
def grp(t, l):
    l = str(l)
    if re.match(r"^\d", l) or l == "S": return "ITF"
    if l == "C": return "Challenger" if t == "atp" else "WTA 125"
    if l == "D": return "Davis/BJK"
    return "turniri ATP" if t == "atp" else "turniri WTA"
def st(g):
    m = g.pnl.mean(); se = g.pnl.std() / np.sqrt(len(g)) if len(g) > 1 else np.nan
    return pd.Series({"bets": len(g), "roi%": round(100 * m, 1), "lo%": round(100 * (m - 1.645 * se), 1), "units": round(g.pnl.sum(), 1)})
rows = {}
for arg in sys.argv[1:]:
    name, path, days = arg.split("=")
    S = pd.read_parquet(path); b = S[(S.evb > 0.05) & (S.o <= 5)].copy()
    b["g"] = [grp(t, l) for t, l in zip(b.tour, b.lvl)]
    for g, x in b.groupby("g"): rows[(name, g)] = st(x)
    n = b[~b.g.isin(["ITF", "Davis/BJK"])]
    rows[(name, "BEZ ITF")] = st(n); rows[(name, "BEZ ITF")]["per_day"] = round(len(n) / int(days), 1)
    rows[(name, "SVE")] = st(b)
    for thr in (0.10, 0.15):
        x = n[n.evb > thr]; rows[(name, f"BEZ ITF EV>{int(thr*100)}%")] = st(x)
T = pd.DataFrame(rows).T
print(T.to_string())
