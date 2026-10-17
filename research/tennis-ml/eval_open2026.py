"""Pre-registered 2026 opening-odds test (see README): EV>5% & odds<=2 at the BetMGM.us opening price.
Reports ROI at the opening and the closing price, 90% lower bound, and closing-line value of the picks vs random sides.

usage: eval_open2026.py DATA_DIR TOUR PREDS[,PREDS..] LEVEL_REGEX [LABEL]
env YEARS (default 2026, e.g. 2025,2026) and BXO (odds file names, default bxo_odds, comma-separated to stack).
"""
import os
import sys

import numpy as np
import pandas as pd

D, TOUR, PREDS, LEVEL = sys.argv[1:5]
LABEL = sys.argv[5] if len(sys.argv) > 5 else TOUR
YEARS = [int(y) for y in os.environ.get("YEARS", "2026").split(",")]
P = pd.concat([pd.read_parquet(f"{D}/out/{f}") for f in PREDS.split(",")])
P = P[~P.index.duplicated()]
O = pd.concat([pd.read_parquet(f"{D}/out/{TOUR}_{n}.parquet") for n in os.environ.get("BXO", "bxo_odds").split(",")])
O = O[~O.index.duplicated()]
lvl = pd.read_parquet(f"{D}/out/{TOUR}_f2.parquet", columns=["tourney_level"]).tourney_level.astype(str)
P = P[P.year.isin(YEARS)].join(O, how="inner")
P = P[lvl.reindex(P.index).str.match(LEVEL).fillna(False).values]
P = P[(P.open_w > 1) & (P.open_l > 1) & (P.close_w > 1) & (P.close_l > 1)]
rows = []
for won in (True, False):
    q = np.where((P.y == 1) == won, P.p, 1 - P.p)
    oo, oc = (P.open_w, P.close_w) if won else (P.open_l, P.close_l)
    rows.append(pd.DataFrame({"mi": P.index, "q": q, "oo": oo.values, "oc": oc.values, "won": won,
                              "stands": P.stands.values & ~P.retired.values, "moves": P.moves.values,
                              "fair_c": (1 / oc.values) / (1 / P.close_w.values + 1 / P.close_l.values)}))
S = pd.concat(rows, ignore_index=True)
S["ev"] = S.q * S.oo - 1
S["clv"] = S.oo / S.oc - 1
for px in ("oo", "oc"):
    S[f"pnl_{px}"] = np.where(S.stands, np.where(S.won, S[px] - 1, -1.0), 0.0)
y = S.won[S.index < len(P)]  # winner side rows: outcome 1
print(f"{LABEL} {YEARS}: {len(P)} matches with BetMGM open+close; median margin open "
      f"{(1 / P.open_w + 1 / P.open_l - 1).median():.2%} close {(1 / P.close_w + 1 / P.close_l - 1).median():.2%}")
W = S[S.won]  # log-loss on the winner side = -mean log p(winner)
print(f"log-loss  model {-np.log(W.q.clip(1e-4)).mean():.4f}  fair close {-np.log(W.fair_c.clip(1e-4)).mean():.4f}")


def report(name, b):
    out = [f"{name:34s} bets {len(b):5d}"]
    for px in ("oo", "oc"):
        x = b[f"pnl_{px}"]
        lo = x.mean() - 1.645 * x.std() / np.sqrt(max(len(x), 1))
        out.append(f"ROI@{'open' if px == 'oo' else 'close'} {x.mean():+6.2%} (90%lo {lo:+5.1%})")
    out.append(f"CLV {b.clv.mean():+5.2%}")
    print("  ".join(out))


base = S[S.oo <= 2.0]
report("random sides odds<=2 (baseline)", base)
report("PRIMARY: EV>5% & open<=2", S[(S.ev > 0.05) & (S.oo <= 2.0)])
for e in (0.0, 0.02, 0.10):
    report(f"EV>{e:.0%} & open<=2", S[(S.ev > e) & (S.oo <= 2.0)])
report("EV>5% & open<=2.5", S[(S.ev > 0.05) & (S.oo <= 2.5)])
b = S[(S.ev > 0.05) & (S.oo <= 2.0)]
m = pd.read_parquet(f"{D}/out/{TOUR}_f2.parquet", columns=["date"]).date.reindex(b.mi)
print("primary by quarter:", b.groupby(m.dt.to_period("Q").astype(str).values).pnl_oo.agg(["size", "mean"]).round(3).to_dict("index"))
