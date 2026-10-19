"""ROI per odds band on the whole-2026 test (eval_levels2026.py --bets=...): every side at its best UK opening price.
For each band: all sides (what the market alone gives), and the sides the blend rule picks at EV>5/10/15%, per model.

usage: eval_oddsbands.py BETS_PARQUET [TAG ...]
"""
import sys

import numpy as np
import pandas as pd

S = pd.read_parquet(sys.argv[1])
TAGS = sys.argv[2:] or sorted(S.tag.unique())
EDGES = [1.0, 1.3, 1.5, 1.8, 2.2, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 1000.0]
S["band"] = pd.cut(S.o, EDGES, right=True, labels=[f"{a:g}-{b:g}" if b < 1000 else f">{a:g}" for a, b in zip(EDGES, EDGES[1:])])


def stat(g):
    if len(g) == 0:
        return pd.Series({"n": 0, "roi": np.nan, "lo": np.nan})
    m, se = g.pnl.mean(), g.pnl.std() / np.sqrt(len(g)) if len(g) > 1 else np.nan
    return pd.Series({"n": len(g), "roi": m, "lo": m - 1.645 * se})


pd.set_option("display.width", 250)
for scope, sub in [("men + women", S), ("men", S[S.tour == "atp"]), ("women", S[S.tour == "wta"])]:
    cols = {}
    base = sub[sub.tag == TAGS[0]]
    cols["all sides"] = base.groupby("band", observed=False).apply(stat, include_groups=False)
    for tag in TAGS:
        t = sub[sub.tag == tag]
        for thr in (0.05, 0.10, 0.15):
            cols[f"{tag} EV>{thr:.0%}"] = t[t.evb > thr].groupby("band", observed=False).apply(stat, include_groups=False)
    T = pd.concat(cols, axis=1)
    for c in T.columns:
        if c[1] in ("roi", "lo"):
            T[c] = (100 * T[c]).round(1)
        else:
            T[c] = T[c].astype("Int64")
    print(f"\n== {scope}: ROI % at the best opening price per odds band (n bets, roi, lower 90% bound)")
    print(T.to_string())
