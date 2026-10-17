"""Betting systems by odds range on walk-forward predictions: odds floors (only bet at odds >= X), odds bands,
and 'model flips the favourite' (market underdog the model rates above 50%).
Post-hoc exploration: every row is reported for an explore half and a confirm half of the years.

usage: odds_systems.py DATA_DIR PRED_FILE TOUR LEVEL_REGEX [ODDS_FILE PREFIX]
  PRED_FILE  preds3_*.parquet in DATA_DIR/out (o1/o2 used unless ODDS_FILE given)
  LEVEL_REGEX  regex on tourney_level, e.g. '^\\d' for women's ITF, '^(\\d|S)' for men's ITF, 'C' Challenger
  ODDS_FILE PREFIX  e.g. wta_te_odds.parquet best  -> uses best_w/best_l
"""
import sys

import numpy as np
import pandas as pd

D, PRED, TOUR, LEVEL = sys.argv[1:5]
P = pd.concat([pd.read_parquet(f"{D}/out/{f}") for f in PRED.split(",")])
P = P[~P.index.duplicated()]
for c in ["p", "o1", "o2"]:
    P[c] = P[c].astype(float)
if len(sys.argv) > 6:
    O = pd.read_parquet(f"{D}/out/{sys.argv[5]}").reindex(P.index)
    w, l = O[f"{sys.argv[6]}_w"].astype(float), O[f"{sys.argv[6]}_l"].astype(float)
    P["o1"], P["o2"] = np.where(P.y == 1, w, l), np.where(P.y == 1, l, w)
lvl = pd.read_parquet(f"{D}/out/{TOUR}_f2.parquet", columns=["tourney_level"]).tourney_level.astype(str)
P = P[lvl.reindex(P.index).str.match(LEVEL).fillna(False).values]
P = P[(P.o1 > 1) & (P.o2 > 1)]
years = sorted(P.year.unique())
cut = years[len(years) // 2]

# one row per side: the bet on p1 and the bet on p2
S = pd.DataFrame({
    "year": np.concatenate([P.year, P.year]),
    "q": np.concatenate([P.p, 1 - P.p]),
    "o": np.concatenate([P.o1, P.o2]),
    "o_other": np.concatenate([P.o2, P.o1]),
    "pnl": np.concatenate([np.where(P.stands, np.where(P.y == 1, P.o1 - 1, -1.0), 0.0),
                           np.where(P.stands, np.where(P.y == 0, P.o2 - 1, -1.0), 0.0)]),
})
S["ev"] = S.q * S.o - 1
rng = np.random.default_rng(0)
print(f"{TOUR} level~{LEVEL} odds={sys.argv[6] if len(sys.argv) > 6 else 'own'}: {len(P)} matches, years {years[0]}-{years[-1]}, "
      f"median margin {np.median(1 / P.o1 + 1 / P.o2 - 1):.2%}; explore < {cut} <= confirm")


def line(name, pick, pool):
    b = S[pick]
    if len(b) < 50:
        return
    boot = np.array([b.pnl.values[rng.integers(0, len(b), len(b))].mean() for _ in range(500)])
    lo, hi = np.percentile(boot, [5, 95])
    pv = S.pnl[pool].values
    rnd = np.mean([rng.choice(pv, len(b), replace=False).mean() for _ in range(200)]) if len(pv) > len(b) else np.nan
    by = b.groupby("year").pnl.mean()
    e, c = b[b.year < cut].pnl, b[b.year >= cut].pnl
    print(f"  {name:34s} bets {len(b):6d} ROI {b.pnl.mean():+6.2%} [{lo:+.1%},{hi:+.1%}] yrs+ {(by > 0).sum():2d}/{len(by):<2d} "
          f"random {rnd:+6.1%} | explore {e.mean():+6.1%} (n={len(e)}) confirm {c.mean():+6.1%} (n={len(c)})")


print("\n-- odds floor: only bet when odds >= X")
for thr in [0.0, 0.05, 0.10, 0.20]:
    for lo_ in [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        line(f"EV>{thr:.0%} odds>={lo_:g}", (S.ev > thr) & (S.o >= lo_), S.o >= lo_)
print("\n-- odds bands (EV>5%)")
for a, b in [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 1000)]:
    line(f"EV>5% odds {a:g}-{b:g}", (S.ev > 0.05) & (S.o >= a) & (S.o < b), (S.o >= a) & (S.o < b))
print("\n-- model flips the favourite (market underdog, model p > X)")
dog = (S.o > S.o_other)
for x in [0.50, 0.55, 0.60]:
    line(f"underdog & model p>{x:.0%}", dog & (S.q > x), dog)
