"""Evaluate saved Challenger/ITF/qualifying predictions at the Kaggle (mostly OddsPortal average) prices."""
import sys

import numpy as np
import pandas as pd

D, TAG = sys.argv[1], sys.argv[2]
P = pd.read_parquet(f"{D}/out/preds3_atp_ens_{TAG}_lower.parquet")
for c in ["o1", "o2", "p", "p_raw"]:
    P[c] = P[c].astype(float)
lvl = pd.read_parquet(f"{D}/out/atp_f2.parquet", columns=["tourney_level"]).tourney_level.astype(str).reindex(P.index)
P["seg"] = np.where(lvl == "C", "Challenger", np.where(lvl.isin(["S", "15", "25"]), "ITF/Futures", "Qualifying"))
P["p_mkt"] = (1 / P.o1) / (1 / P.o1 + 1 / P.o2)
rng = np.random.default_rng(0)


def ll(q, y):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))


for seg, G in [("ALL", P)] + list(P.groupby("seg")):
    print(f"\n=== {seg} 2014-2018: {len(G)} matches, margin {(1 / G.o1 + 1 / G.o2 - 1).median():.2%} | "
          f"logloss market {ll(G.p_mkt, G.y):.4f} model {ll(G.p, G.y):.4f} | accuracy market "
          f"{np.mean((G.p_mkt > .5) == G.y):.3f} model {np.mean((G.p > .5) == G.y):.3f}")
    for thr in [0.02, 0.05, 0.10]:
        for mo in [2.0, 3.0, 100.0]:
            pnl, yr = [], []
            for s_, o in ((1, G.o1.values), (2, G.o2.values)):
                q = G.p.values if s_ == 1 else 1 - G.p.values
                win = (G.y == 1).values if s_ == 1 else (G.y == 0).values
                pick = (q * o - 1 > thr) & (o <= mo)
                pnl.append(np.where(G.stands.values, np.where(win, o - 1, -1.0), 0.0)[pick]); yr.append(G.year.values[pick])
            pnl, yr = np.concatenate(pnl), np.concatenate(yr)
            if len(pnl) < 50:
                continue
            boot = np.array([pnl[rng.integers(0, len(pnl), len(pnl))].mean() for _ in range(1000)])
            lo, hi = np.percentile(boot, [5, 95])
            by = pd.Series(pnl).groupby(yr).mean()
            print(f"  EV>{thr:4.0%} odds<={mo:<3g} bets {len(pnl):5d}  ROI {pnl.mean():+6.2%}  90%CI [{lo:+.1%},{hi:+.1%}]  "
                  f"years+ {(by > 0).sum()}/{len(by)}")
