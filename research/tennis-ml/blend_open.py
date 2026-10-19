"""Betting-stage recalibration for the opening-odds rule: P(win) = logistic(a*logit(model) + b*logit(fair opening
price) + c), fitted on one year and applied to the other (2025 <-> 2026). The model itself stays odds-free; the
market price only enters the bet decision, as it already does through EV. Input: open_sides_2526.parquet (one row
per match side: seg, year, q, o, fair, won, stands, pnl, ev).

usage: blend_open.py DATA_DIR
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

S = pd.read_parquet(f"{sys.argv[1]}/out/open_sides_2526.parquet")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
X = lambda d: np.c_[lg(d.q.values), lg(d.fair.values)]
out = []
for fit_y, test_y in ((2025, 2026), (2026, 2025)):
    tr, te = S[(S.year == fit_y) & S.stands], S[S.year == test_y].copy()
    m = LogisticRegression(C=1e4).fit(X(tr), tr.won.values)
    print(f"fit {fit_y}: weight model {m.coef_[0][0]:.2f}, market {m.coef_[0][1]:.2f}")
    te["qb"] = m.predict_proba(X(te))[:, 1]
    out.append(te)
T = pd.concat(out)
T["evb"] = T.qb * T.o - 1


def rep(name, b):
    m, se = b.pnl.mean(), b.pnl.std() / np.sqrt(len(b))
    print(f"{name:28s} {len(b):5d} {m:+7.2%} {m - 1.645 * se:+7.2%} | "
          + " / ".join(f"{b[b.year == y].pnl.mean():+.1%}({(b.year == y).sum()})" for y in (2025, 2026)))


rep("raw EV>5% odds<=2", T[(T.ev > 0.05) & (T.o <= 2)])
for cap in (2, 2.5, 3, 4, 100):
    for thr in (0.03, 0.05, 0.10):
        rep(f"blend EV>{thr:.0%} odds<={cap}", T[(T.evb > thr) & (T.o <= cap)])
