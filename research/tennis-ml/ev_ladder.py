"""EV threshold ladder on the pooled 2025+2026 opening-odds sample (women's ITF, Challenger, men's ITF; BetMGM.us
open, odds<=CAP): bets per day, ROI, 90% lower bound, by year and segment, up to the threshold giving ~1 bet a day.

usage: ev_ladder.py DATA_DIR [CAP=2.0]
"""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
CAP = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
SEGS = (("Women ITF", "wta", ["bxo25_witf", "bxo_odds"], r"^\d"), ("Challenger", "atp", ["bxo25_chal", "bxo_odds_chal"], r"^C$"),
        ("Men ITF", "atp", ["bxo25_mitf", "bxo_odds_mitf"], r"^(\d|S)"))
rows = []
for seg, tour, bxo, rx in SEGS:
    P = pd.concat([pd.read_parquet(f"{D}/out/preds3_{tour}_ens_r6_hist1991_ens_lower_{y}.parquet") for y in (2025, 2026)])
    O = pd.concat([pd.read_parquet(f"{D}/out/{tour}_{n}.parquet") for n in bxo])
    P, O = P[~P.index.duplicated()], O[~O.index.duplicated()]
    F = pd.read_parquet(f"{D}/out/{tour}_f2.parquet", columns=["tourney_level"])
    P = P[P.year.isin([2025, 2026])].join(O, how="inner")
    P = P[F.tourney_level.astype(str).reindex(P.index).str.match(rx).fillna(False).values & (P.open_w > 1) & (P.open_l > 1)]
    for won in (True, False):
        q = np.where((P.y == 1) == won, P.p, 1 - P.p)
        o = (P.open_w if won else P.open_l).values
        st = P.stands.values & ~P.retired.values
        rows.append(pd.DataFrame({"seg": seg, "year": P.year.values, "day": P.date_bx.values, "ev": q * o - 1, "o": o,
                                  "pnl": np.where(st, np.where(won, o - 1, -1.0), 0.0)}))
S = pd.concat(rows, ignore_index=True)
S = S[S.o <= CAP]
days = (S.day.max() - S.day.min()).days + 1
print(f"sample {S.day.min():%Y-%m-%d}..{S.day.max():%Y-%m-%d} = {days} days, odds<={CAP}")
print(f"{'EV >':>6} {'bets':>5} {'/day':>5} {'ROI':>7} {'90%lo':>7} {'units':>6} {'maxDD':>6} | {'2025':>14} {'2026':>14} | by segment")
for thr in (0.05, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30):
    b = S[S.ev > thr].sort_values("day")
    if len(b) < 20:
        break
    m, se = b.pnl.mean(), b.pnl.std() / np.sqrt(len(b))
    c = b.pnl.cumsum().values
    yr = " ".join(f"{b[b.year == y].pnl.mean():+6.1%}({(b.year == y).sum():4d})" for y in (2025, 2026))
    sg = " ".join(f"{s[:5]} {g.pnl.mean():+.1%}({len(g)})" for s, g in b.groupby("seg"))
    print(f"{thr:6.0%} {len(b):5d} {len(b) / days:5.2f} {m:+7.2%} {m - 1.645 * se:+7.2%} {c[-1]:6.1f} {(np.maximum.accumulate(c) - c).max():6.1f} | {yr} | {sg}")
