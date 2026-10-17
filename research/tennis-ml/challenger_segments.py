"""Challenger segments: explore on 2012-2015, confirm on 2016-2018 (pre-registered split in README).
Also reports the fresh 2012-2013 ITF test."""
import sys

import numpy as np
import pandas as pd

D, TAG = sys.argv[1], sys.argv[2]
P = pd.concat([pd.read_parquet(f"{D}/out/preds3_atp_ens_{TAG}_lower_2012-2013.parquet"),
               pd.read_parquet(f"{D}/out/preds3_atp_ens_{TAG}_lower.parquet")])
for c in ["o1", "o2", "p"]:
    P[c] = P[c].astype(float)
F = pd.read_parquet(f"{D}/out/atp_f2.parquet", columns=[
    "tourney_level", "rord", "surface", "w_lvl_n", "l_lvl_n", "w_n_career", "l_n_career", "w_rank", "l_rank",
    "w_age", "l_age", "w_home", "l_home"]).reindex(P.index)
lvl = F.tourney_level.astype(str)
P["seg"] = np.where(lvl == "C", "Challenger", np.where(lvl.isin(["S", "15", "25"]), "ITF", "Qual"))
w1 = P.y == 1
side = lambda a, b: (np.where(w1, F[a], F[b]), np.where(w1, F[b], F[a]))


def bets(G, thr=0.05, mo=2.0):
    out = []
    for s_, o, q, win in ((1, G.o1, G.p, G.y == 1), (2, G.o2, 1 - G.p, G.y == 0)):
        pick = (q * o - 1 > thr) & (o <= mo)
        out.append(pd.DataFrame({"pnl": np.where(G.stands, np.where(win, o - 1, -1.0), 0.0), "year": G.year,
                                 "idx": G.index, "side": s_})[pick])
    return pd.concat(out)


def fmt(b):
    if len(b) < 30:
        return f"n={len(b)}"
    return f"n={len(b):5d} ROI {b.pnl.mean():+6.2%} ± {1.65 * b.pnl.std() / np.sqrt(len(b)):.1%}"


print("=== Fresh ITF test 2012-2013 (pre-registered rule: EV>5%, odds<=2)")
itf = P[(P.seg == "ITF") & P.year.isin([2012, 2013])]
b = bets(itf)
print(f"  ITF 2012-2013: {fmt(b)}  by year: " + " ".join(f"{y}:{v:+.1%}" for y, v in b.groupby('year').pnl.mean().items()))
print(f"  ITF 2012-2018 overall: {fmt(bets(P[P.seg == 'ITF']))}")

C = P[P.seg == "Challenger"].copy()
lvl_n1, lvl_n2 = side("w_lvl_n", "l_lvl_n")
rk1, rk2 = side("w_rank", "l_rank")
C["min_lvl_n"] = np.minimum(lvl_n1, lvl_n2)[(P.seg == "Challenger").values]
C["max_rank"] = np.maximum(np.nan_to_num(rk1, nan=3000), np.nan_to_num(rk2, nan=3000))[(P.seg == "Challenger").values]
C["rord"] = F.rord.reindex(C.index).values
C["surface"] = F.surface.reindex(C.index).values
segs = {
    "all": C.index == C.index,
    "a player with <20 Challenger-level matches": C.min_lvl_n < 20,
    "a player with <50 Challenger-level matches": C.min_lvl_n < 50,
    "both with 50+": C.min_lvl_n >= 50,
    "weaker player ranked >300": C.max_rank > 300,
    "both inside top-300": C.max_rank <= 300,
    "first round (R32/R64)": C.rord <= 7,
    "later rounds": C.rord > 7,
    "clay": C.surface == "Clay",
    "hard": C.surface == "Hard",
}
for thr, mo in [(0.05, 2.0), (0.05, 100.0)]:
    print(f"\n=== Challenger, rule EV>{thr:.0%} odds<={mo:g}: explore 2012-2015 | confirm 2016-2018")
    for name, mask in segs.items():
        G = C[mask]
        e, c = bets(G[G.year <= 2015], thr, mo), bets(G[G.year >= 2016], thr, mo)
        print(f"  {name:45s} explore {fmt(e)}   | confirm {fmt(c)}")
