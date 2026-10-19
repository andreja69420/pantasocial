"""Opening vs closing odds for the basketball model, per bookmaker: ROI of EV rules, closing-line value, random sides,
a void stress test (refund bets whose fair win probability rose >12 points before the close) and a model+market
Shin blend fitted leave-one-season-out.

usage: eval_bball.py LEAGUE_DIR [LABEL]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DIR = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else DIR
P = pd.read_parquet(f"{DIR}/preds.parquet").set_index("mid")
O = pd.read_parquet(f"{DIR}/odds.parquet")
P = P.join(O, how="inner")


def shin(a, b):
    lo, hi = np.zeros_like(a), np.full_like(a, 0.5)
    B = a + b
    f = lambda z, x: (np.sqrt(z ** 2 + 4 * (1 - z) * x ** 2 / B) - z) / (2 * (1 - z))
    for _ in range(50):
        z = (lo + hi) / 2
        s = f(z, a) + f(z, b)
        hi, lo = np.where(s < 1, z, hi), np.where(s < 1, lo, z)
    return f((lo + hi) / 2, a)


lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
for bk, name in (("b365", "bet365.us"), ("mgm", "BetMGM.us"), ("stake", "Stake.com")):
    if f"{bk}_oh" not in P:
        continue
    Q = P[(P[f"{bk}_oh"] > 1) & (P[f"{bk}_oa"] > 1) & (P[f"{bk}_ch"] > 1) & (P[f"{bk}_ca"] > 1)]
    rows = []
    for home in (True, False):
        o, oo = (Q[f"{bk}_oh"], Q[f"{bk}_oa"]) if home else (Q[f"{bk}_oa"], Q[f"{bk}_oh"])
        c, cc = (Q[f"{bk}_ch"], Q[f"{bk}_ca"]) if home else (Q[f"{bk}_ca"], Q[f"{bk}_ch"])
        won = (Q.y == 1) if home else (Q.y == 0)
        rows.append(pd.DataFrame({"season": Q.season.values, "q": (Q.p_home if home else 1 - Q.p_home).values,
                                  "o": o.values, "oo": oo.values, "c": c.values, "cc": cc.values, "won": won.values}))
    S = pd.concat(rows, ignore_index=True)
    S["fair_o"] = (1 / S.o) / (1 / S.o + 1 / S.oo)
    S["fair_c"] = (1 / S.c) / (1 / S.c + 1 / S.cc)
    S["shin_o"] = shin(1 / S.o.values, 1 / S.oo.values)
    S["pnl_o"], S["pnl_c"] = np.where(S.won, S.o - 1, -1.0), np.where(S.won, S.c - 1, -1.0)
    S["pv"] = np.where(S.fair_c - S.fair_o > 0.12, 0.0, S.pnl_o)
    S["clv"] = S.o / S.c - 1
    S["qb"] = np.nan
    for s in S.season.unique():
        tr, te = S.season != s, S.season == s
        X = lambda d: np.c_[lg(d.q.values), lg(d.shin_o.values)]
        S.loc[te, "qb"] = LogisticRegression(C=1e4).fit(X(S[tr]), S.won[tr]).predict_proba(X(S[te]))[:, 1]
    S["ev"], S["evb"] = S.q * S.o - 1, S.qb * S.o - 1
    W = S[S.won]
    print(f"\n== {LABEL} @ {name}: {len(Q)} games, seasons {sorted(Q.season.unique())}, margin open "
          f"{np.median(1 / Q[f'{bk}_oh'] + 1 / Q[f'{bk}_oa'] - 1):.1%}")
    print(f"   log-loss: model {-np.log(W.q.clip(1e-4)).mean():.4f}  open fair {-np.log(W.fair_o).mean():.4f}  "
          f"close fair {-np.log(W.fair_c).mean():.4f}  blend {-np.log(W.qb.clip(1e-4)).mean():.4f}")

    def rep(lab, b):
        if len(b) < 30:
            print(f"   {lab:30s} {len(b):5d} bets")
            return
        m, se = b.pnl_o.mean(), b.pnl_o.std() / np.sqrt(len(b))
        yrs = " ".join(f"{b[b.season == y].pnl_o.mean():+.0%}" for y in sorted(b.season.unique()))
        print(f"   {lab:30s} {len(b):5d}  open {m:+6.2%} (lo {m - 1.645 * se:+5.1%})  close {b.pnl_c.mean():+6.2%}  "
              f"void-stress {b.pv.mean():+6.2%}  CLV {b.clv.mean():+5.2%} | {yrs}")
    rep("random sides odds<=2", S[S.o <= 2])
    rep("PRIMARY EV>5% odds<=2", S[(S.ev > 0.05) & (S.o <= 2)])
    for thr in (0.0, 0.03, 0.10):
        rep(f"EV>{thr:.0%} odds<=2", S[(S.ev > thr) & (S.o <= 2)])
    rep("EV>5% odds<=3", S[(S.ev > 0.05) & (S.o <= 3)])
    rep("EV>5% any odds", S[S.ev > 0.05])
    for thr in (0.03, 0.05, 0.10):
        rep(f"blend EV>{thr:.0%} odds<=3", S[(S.evb > thr) & (S.o <= 3)])
