"""Pre-registered out-of-sample test (see README): ATP Rule A (Max odds, EV>5%, odds<=3) and Rule B (Pinnacle,
EV>2%, odds<=3), on 2020-2024 dev predictions and on the 2025-2026 holdout predictions.
Odds files: tennis-data.co.uk ATP 2020-2026 (mirrored in gmalbert/tennis-predictions)."""
import glob
import sys

import numpy as np
import pandas as pd

D, GM, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
s = pd.concat([pd.read_excel(f) for f in sorted(glob.glob(f"{GM}/*.xlsx"))], ignore_index=True)
s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
for c in ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL"]:
    s[c] = pd.to_numeric(s[c], errors="coerce")
s = s.dropna(subset=["B365W", "B365L"])
key = lambda w, l: np.round(w, 3).astype(str) + "|" + np.round(l, 3).astype(str)
s["k"] = key(s.B365W, s.B365L)
rng = np.random.default_rng(0)
f2 = pd.read_parquet(f"{D}/out/atp_f2.parquet", columns=["date", "odds_w", "odds_l"])
f2.index.name = "index"

for split, years in [("dev", (2020, 2024)), ("holdout", (2025, 2026))]:
    P = pd.read_parquet(f"{D}/out/preds3_atp_ens_{TAG}_{split}.parquet")
    P = P[(P.year >= years[0]) & (P.year <= years[1])]
    g = f2.loc[P.index].copy()
    g.index.name = "index"
    g["k"] = key(g.odds_w, g.odds_l)
    j = g.reset_index().merge(s[["k", "Date", "PSW", "PSL", "MaxW", "MaxL"]], on="k")
    j = j[(j.Date - j.date).dt.days.between(-2, 20)].drop_duplicates("index").set_index("index")
    P = P.join(j[["PSW", "PSL", "MaxW", "MaxL"]], how="inner")
    w1 = P.y == 1
    print(f"\n=== ATP {split} {years[0]}-{years[1]}: {len(P)} predictions matched to full odds")
    for name, ow, ol, thr in [("Rule A  Max odds, EV>5%, odds<=3", "MaxW", "MaxL", 0.05),
                              ("Rule B  Pinnacle,  EV>2%, odds<=3", "PSW", "PSL", 0.02)]:
        o1, o2 = np.where(w1, P[ow], P[ol]), np.where(w1, P[ol], P[ow])
        ok = ~np.isnan(o1) & ~np.isnan(o2)
        pnl, yr = [], []
        for s_, o in ((1, o1), (2, o2)):
            q = P.p.values if s_ == 1 else 1 - P.p.values
            win = (P.y == 1).values if s_ == 1 else (P.y == 0).values
            pick = ok & (q * o - 1 > thr) & (o <= 3.0)
            pnl.append(np.where(P.stands.values, np.where(win, o - 1, -1.0), 0.0)[pick]); yr.append(P.year.values[pick])
        pnl, yr = np.concatenate(pnl), np.concatenate(yr)
        boot = np.array([pnl[rng.integers(0, len(pnl), len(pnl))].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [5, 95])
        by = pd.Series(pnl).groupby(yr).agg(["size", "mean"])
        verdict = "PASS" if pnl.mean() > 0 and lo > 0 else "FAIL"
        print(f"  {name}: bets {len(pnl)}  ROI {pnl.mean():+.2%}  90% CI [{lo:+.2%}, {hi:+.2%}]  -> {verdict}")
        print("     by year: " + "  ".join(f"{y}: {r['mean']:+.1%} (n={int(r['size'])})" for y, r in by.iterrows()))
