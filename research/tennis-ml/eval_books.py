"""Summary of whole-2026 test runs at different bookmakers' prices (eval_levels2026.py --bets=... --book=... [--close]):
the plan rule (blend EV>5%, odds<=5, best price among the run's books), flat 1 unit: bets, ROI, lower 90% bound,
average odds, per tour and per level group (ITF / Challenger + WTA 125 / tour events).

usage: eval_books.py NAME=BETS_PARQUET [NAME=BETS_PARQUET ...]
"""
import sys

import numpy as np
import pandas as pd

GROUP = {"15": "ITF", "25": "ITF", "35": "ITF", "50": "ITF", "75": "ITF", "100": "ITF", "C": "Challenger / WTA 125"}


def stat(g):
    if len(g) < 2:
        return pd.Series({"bets": len(g), "roi%": np.nan, "lo%": np.nan, "odds": np.nan})
    m, se = g.pnl.mean(), g.pnl.std() / np.sqrt(len(g))
    return pd.Series({"bets": len(g), "roi%": 100 * m, "lo%": 100 * (m - 1.645 * se), "odds": g.o.mean()})


rows = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    S = pd.read_parquet(path)
    b = S[(S.evb > 0.05) & (S.o <= 5)]
    b = b.assign(grp=b.lvl.astype(str).map(GROUP).fillna("tour events"))
    rows[(name, "all")] = stat(b)
    for (t, g), x in b.groupby(["tour", "grp"]):
        rows[(name, f"{t} {g}")] = stat(x)
pd.set_option("display.width", 200)
T = pd.DataFrame(rows).T
T["bets"] = T.bets.astype(int)
print(T.round(2).to_string())
