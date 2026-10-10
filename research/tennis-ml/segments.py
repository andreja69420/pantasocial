"""Where (if anywhere) does the model beat the price, and what margin could the edge survive?"""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
out = []
for tour in ["atp", "wta"]:
    P = pd.read_parquet(f"{D}/out/{tour}_preds.parquet")
    F = pd.read_parquet(f"{D}/out/{tour}_features.parquet", columns=["w_n", "l_n", "winner_rank", "loser_rank", "rord", "tourney_level"])
    P = P.join(F)
    P["tour"] = tour
    out.append(P)
P = pd.concat(out)
P["min_n"] = np.minimum(P.w_n, P.l_n)
P["max_rank"] = np.maximum(P.winner_rank.fillna(2000), P.loser_rank.fillna(2000))


def bets(P, pcol, thr, odds_fn=None):
    rows = []
    for s in (1, 2):
        o = P.o1 if s == 1 else P.o2
        if odds_fn is not None:
            o = odds_fn(P, s)
        p = P[pcol] if s == 1 else 1 - P[pcol]
        win = P.y == 1 if s == 1 else P.y == 0
        pick = (p * o - 1 > thr) & P[pcol].notna()
        pnl = np.where(P.stands, np.where(win, o - 1, -1.0), 0.0)
        rows.append(pd.DataFrame({"pnl": pnl, "odds": o, "year": P.year})[pick])
    return pd.concat(rows)


def roi(b):
    if len(b) < 30:
        return f"n={len(b)}"
    se = b.pnl.std() / np.sqrt(len(b))
    return f"n={len(b):5d} ROI {b.pnl.mean():+6.1%} ± {1.96 * se:.1%}"


print("XGBoost value bets (EV>5%) by segment, both tours:")
segs = {
    "all": P.index == P.index,
    "lesser player has <50 career matches (all levels)": P.min_n < 50,
    "lesser player has 50-200": (P.min_n >= 50) & (P.min_n < 200),
    "both have 200+": P.min_n >= 200,
    "weaker player ranked outside top-100": P.max_rank > 100,
    "both top-100": P.max_rank <= 100,
    "Grand Slams": P.tourney_level == "G",
    "qualifying-heavy early rounds (R128/R64)": P.rord <= 6,
}
for name, mask in segs.items():
    print(f"  {name:52s} model: {roi(bets(P[mask], 'p_model', .05))}   stacked: {roi(bets(P[mask], 'p_stack', .03))}")

print("\nIf the same bets could be placed at a lower margin (odds rebuilt from Bet365's fair price):")
for mg in [0.065, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]:
    f = lambda P, s, mg=mg: 1 / ((P.p_mkt if s == 1 else 1 - P.p_mkt) * (1 + mg))
    print(f"  margin {mg:.1%}:  stacked EV>2%: {roi(bets(P, 'p_stack', .02, f))}   model EV>5%: {roi(bets(P, 'p_model', .05, f))}")
