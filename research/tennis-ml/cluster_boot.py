"""Tournament-clustered bootstrap for the BetMGM 2025-26 opening-odds results: bets in the same tournament share
conditions, so resample whole tournaments instead of single bets. Also reports results with the single most
profitable tournament (and the top 5) removed.

usage: cluster_boot.py DATA_DIR   (env PREDTAG as in blend_transfer.py)
"""
import os
import sys

import numpy as np
import pandas as pd

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blend_transfer.py")).read()
src = src.replace('rows.append(pd.DataFrame({"seg": seg,', 'rows.append(pd.DataFrame({"mi": P.index, "seg": seg,')
exec(src.split("def rep(")[0].split('"""', 2)[2])
tid = pd.concat([pd.read_parquet(f"{D}/out/{t}_f2.parquet", columns=["tourney_name", "date"]).assign(tour=t) for t in ("wta", "atp")])
T["tour"] = np.where(T.seg == "Women ITF", "wta", "atp")
key = {}
for t in ("wta", "atp"):
    f = pd.read_parquet(f"{D}/out/{t}_f2.parquet", columns=["tourney_name", "date"])
    key[t] = (f.tourney_name.astype(str) + "|" + f.date.dt.strftime("%Y-%m-%d"))
T["tid"] = [key[t].get(i, "?") for t, i in zip(T.tour, T.mi)]
rng = np.random.default_rng(0)
for name, sel in (("raw EV>5% odds<=2", (T.ev > 0.05) & (T.o <= 2)), ("blend EV>5% odds<=2", (T.evb > 0.05) & (T.o <= 2)),
                  ("blend EV>5% odds<=5", (T.evb > 0.05) & (T.o <= 5)), ("blend EV>10% odds<=5", (T.evb > 0.10) & (T.o <= 5))):
    b = T[sel]
    g = b.groupby("tid").pnl.agg(["sum", "size"])
    s, n = g["sum"].values, g["size"].values
    boots = []
    for _ in range(5000):
        i = rng.integers(0, len(g), len(g))
        boots.append(s[i].sum() / n[i].sum())
    lo, hi = np.percentile(boots, [5, 95])
    top = g.sort_values("sum", ascending=False)
    drop1 = (s.sum() - top["sum"].iloc[0]) / (n.sum() - top["size"].iloc[0])
    drop5 = (s.sum() - top["sum"].iloc[:5].sum()) / (n.sum() - top["size"].iloc[:5].sum())
    print(f"{name:22s} {len(b):5d} bets in {len(g):4d} tournaments  ROI {s.sum() / n.sum():+6.2%}  "
          f"clustered 90% CI [{lo:+.1%}, {hi:+.1%}]  without best tournament {drop1:+.2%}, without best 5 {drop5:+.2%}")
