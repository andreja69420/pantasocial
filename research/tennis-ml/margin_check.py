"""Settle the model's walk-forward predictions at Bet365, average, Pinnacle and best-available odds.
Full tennis-data.co.uk columns 2010-2019 come from the 0xsimulacra/MLT mirror (cross-checked against the CRAN
sportsR copy of 2019). The model never sees odds; only the price we bet at changes."""
import sys

import numpy as np
import pandas as pd

D, MLT, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = sys.argv[4] if len(sys.argv) > 4 else "ens"
RULES = [(0.02, 100), (0.05, 100), (0.02, 3), (0.05, 3), (0.10, 3)]
rng = np.random.default_rng(0)

for tour in ["atp", "wta"]:
    P = pd.read_parquet(f"{D}/out/preds3_{tour}_{MODEL}_{TAG}_dev.parquet")
    P = P[P.year <= 2019]
    f2 = pd.read_parquet(f"{D}/out/{tour}_f2.parquet", columns=["date", "odds_w", "odds_l"]).loc[P.index]
    f2.index.name = "index"
    s = pd.read_csv(f"{MLT}/df_{tour}.csv", low_memory=False)
    s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
    for c in ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["B365W", "B365L"])
    key = lambda w, l: np.round(w, 3).astype(str) + "|" + np.round(l, 3).astype(str)
    s["k"], f2["k"] = key(s.B365W, s.B365L), key(f2.odds_w, f2.odds_l)
    j = f2.reset_index().merge(s[["k", "Date", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]], on="k")
    j = j[(j.Date - j.date).dt.days.between(-2, 20)].drop_duplicates("index").set_index("index")
    P = P.join(j[["PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]], how="inner")
    w1 = P.y == 1
    for b in ["PS", "Max", "Avg"]:
        P[f"{b}1"], P[f"{b}2"] = np.where(w1, P[f"{b}W"], P[f"{b}L"]), np.where(w1, P[f"{b}L"], P[f"{b}W"])
    P["B3651"], P["B3652"] = P.o1, P.o2
    print(f"\n{tour.upper()} {P.year.min()}-{P.year.max()}: {len(P)} model predictions matched to full odds")
    for b in ["B365", "Avg", "PS", "Max"]:
        o1, o2 = P[f"{b}1"], P[f"{b}2"]
        ok = o1.notna() & o2.notna() & (o1 > 1) & (o2 > 1)
        print(f"  {b:4s} margin {(1 / o1 + 1 / o2 - 1)[ok].median():5.2%} (n={ok.sum()})")
        for thr, mo in RULES:
            pnl, yr = [], []
            for s_, o in ((1, o1), (2, o2)):
                q = P.p if s_ == 1 else 1 - P.p
                win = (P.y == 1) if s_ == 1 else (P.y == 0)
                pick = ok & (q * o - 1 > thr) & (o <= mo)
                pnl.append(np.where(P.stands, np.where(win, o - 1, -1.0), 0.0)[pick]); yr.append(P.year[pick].values)
            pnl, yr = np.concatenate(pnl), np.concatenate(yr)
            boot = np.array([pnl[rng.integers(0, len(pnl), len(pnl))].mean() for _ in range(1000)])
            lo, hi = np.percentile(boot, [5, 95])
            by = pd.Series(pnl).groupby(yr).mean()
            print(f"     EV>{thr:4.0%} odds<={mo:<3g} bets {len(pnl):5d}  ROI {pnl.mean():+6.2%}  90%CI [{lo:+.1%},{hi:+.1%}]  "
                  f"years+ {(by > 0).sum()}/{len(by)}  " + " ".join(f"{k}:{v:+.0%}" for k, v in by.items()))
