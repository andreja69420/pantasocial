"""ITF/Futures: is the profit the model's, or would any simple rule at the same prices win too?"""
import sys

import numpy as np
import pandas as pd

D, TAG = sys.argv[1], sys.argv[2]
P = pd.read_parquet(f"{D}/out/preds3_atp_ens_{TAG}_lower.parquet")
for c in ["o1", "o2", "p"]:
    P[c] = P[c].astype(float)
lvl = pd.read_parquet(f"{D}/out/atp_f2.parquet", columns=["tourney_level"]).tourney_level.astype(str).reindex(P.index)
P = P[lvl.isin(["S", "15", "25"]).values]
rng = np.random.default_rng(3)
pnl1 = np.where(P.stands, np.where(P.y == 1, P.o1 - 1, -1.0), 0.0)
pnl2 = np.where(P.stands, np.where(P.y == 0, P.o2 - 1, -1.0), 0.0)


def roi(m1, m2):
    v = np.r_[pnl1[m1], pnl2[m2]]
    return f"bets {len(v):5d}  ROI {v.mean():+6.2%} ± {1.65 * v.std() / np.sqrt(len(v)):.1%}"


f1, f2 = P.o1 < P.o2, P.o2 < P.o1
print(f"ITF/Futures 2014-2018, n={len(P)}")
m1, m2 = (P.p * P.o1 > 1.05) & (P.o1 <= 2), ((1 - P.p) * P.o2 > 1.05) & (P.o2 <= 2)
print("  model EV>5%, odds<=2        ", roi(m1, m2))
print("  every favourite (odds<=2)   ", roi(f1 & (P.o1 <= 2), f2 & (P.o2 <= 2)))
print("  every favourite             ", roi(f1, f2))
print("  every underdog              ", roi(~f1, ~f2))
print("  higher Elo player           ", end="")
el = pd.read_parquet(f"{D}/out/atp_f2.parquet", columns=["w_elo", "l_elo"]).reindex(P.index)
p1_is_w = P.y == 1
e1 = np.where(p1_is_w, el.w_elo, el.l_elo); e2 = np.where(p1_is_w, el.l_elo, el.w_elo)
print(roi(e1 > e2, e2 > e1))
pool1, pool2 = np.flatnonzero(P.o1 <= 2), np.flatnonzero(P.o2 <= 2)
pool = np.r_[pnl1[pool1], pnl2[pool2]]
n = int(m1.sum() + m2.sum())
sims = np.array([rng.choice(pool, n, replace=False).mean() for _ in range(2000)])
v = np.r_[pnl1[m1], pnl2[m2]].mean()
print(f"  random {n} bets at odds<=2: mean {sims.mean():+.2%}, 99th pct {np.percentile(sims, 99):+.2%}; "
      f"model {v:+.2%} beats {np.mean(sims < v):.1%} of random draws")
