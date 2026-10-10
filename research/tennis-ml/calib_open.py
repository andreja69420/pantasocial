"""Calibration methods for the opening-odds bet decision, walk-forward on Bet365 TE data: every test year Y is
scored with a calibration map fitted on the years before it (from 2014). The model stays odds-free; the market
price only enters these betting-stage maps.

  raw      - the model's own (Platt, validation-year) probability
  beta     - beta calibration of the model (Kull et al. 2017): logistic on ln q and -ln(1-q)
  blend    - logistic on logit(model) + logit(fair opening price), margin removed proportionally
  shin_b   - same with Shin's margin removal (more margin on the longshot)
  shin_fx  - Shin blend + separate slopes above odds 2 (lets the map bend for outsiders)

Scored on 2020-2025: log-loss on all sides and EV>thr bets by odds band, also with two void stress tests
(refund bets whose price was cut by >30%, or whose fair win probability rose by >12 points, before the close).

usage: calib_open.py DATA_DIR
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

D = sys.argv[1]
TE = "2014-2015-2016-2017-2018-2019-2020-2021-2022-2023-2024-2025"
SEGS = (("Women ITF", "wta", r"^\d"), ("WTA125", "wta", r"^C$"), ("WTA Tour", "wta", r"^[GPIF]"), ("Challenger", "atp", r"^C$"))


def shin(a, b):
    """Shin fair probabilities for a two-way market with inverse odds a, b (vectorised bisection on z)."""
    lo, hi = np.zeros_like(a), np.full_like(a, 0.5)
    B = a + b
    f = lambda z, x: (np.sqrt(z ** 2 + 4 * (1 - z) * x ** 2 / B) - z) / (2 * (1 - z))
    for _ in range(50):
        z = (lo + hi) / 2
        s = f(z, a) + f(z, b)
        hi, lo = np.where(s < 1, z, hi), np.where(s < 1, lo, z)
    z = (lo + hi) / 2
    return f(z, a)


rows = []
for seg, tour, rx in SEGS:
    P = pd.read_parquet(f"{D}/out/preds3_{tour}_ens_r6_hist1991_ens_te_{TE}.parquet")
    E = pd.read_parquet(f"{D}/out/{tour}_te_extra.parquet").reindex(P.index)
    C = pd.read_parquet(f"{D}/out/{tour}_te_odds.parquet").reindex(P.index)
    lvl = pd.read_parquet(f"{D}/out/{tour}_f2.parquet", columns=["tourney_level"]).tourney_level.astype(str).reindex(P.index)
    keep = lvl.str.match(rx).fillna(False).values
    P, E, C = P[keep], E[keep], C[keep]
    ow, ol = E.b365o_w.astype(float).values, E.b365o_l.astype(float).values
    cw, cl = C.b365_w.astype(float).values, C.b365_l.astype(float).values
    yw = (P.y == 1).values
    for side in (1, 2):
        won = yw if side == 1 else ~yw
        o, oo = np.where(won, ow, ol), np.where(won, ol, ow)
        c, cc = np.where(won, cw, cl), np.where(won, cl, cw)
        rows.append(pd.DataFrame({"seg": seg, "year": P.year.values, "q": P.p.values if side == 1 else 1 - P.p.values,
                                  "o": o, "oo": oo, "c": c, "cc": cc, "won": won, "stands": P.stands.values,
                                  "pnl": np.where(P.stands.values, np.where(won, o - 1, -1.0), 0.0)}))
S = pd.concat(rows, ignore_index=True)
S = S[(S.o > 1) & (S.oo > 1) & (S.c > 1) & (S.cc > 1)].reset_index(drop=True)
S["fair"] = (1 / S.o) / (1 / S.o + 1 / S.oo)
S["shin"] = shin(1 / S.o.values, 1 / S.oo.values)
S = S[S[["q", "fair", "shin"]].notna().all(axis=1)].reset_index(drop=True)
S["pv"] = np.where(S.o / S.c > 1.3, 0.0, S.pnl)
# second stress on the probability scale (fair to all odds levels): void if the side's fair win probability rose
# by more than 12 points between open and close
S["pv2"] = np.where((1 / S.c) / (1 / S.c + 1 / S.cc) - S.fair > 0.12, 0.0, S.pnl)
cl = lambda p: np.clip(p, 1e-4, 1 - 1e-4)
lg = lambda p: np.log(cl(p) / (1 - cl(p)))
hi = lambda d: (d.o.values > 2).astype(float)
FEAT = {
    "beta": lambda d: np.c_[np.log(cl(d.q.values)), -np.log(1 - cl(d.q.values))],
    "blend": lambda d: np.c_[lg(d.q.values), lg(d.fair.values)],
    "shin_b": lambda d: np.c_[lg(d.q.values), lg(d.shin.values)],
    "shin_fx": lambda d: np.c_[lg(d.q.values), lg(d.shin.values), hi(d), hi(d) * lg(d.q.values), hi(d) * lg(d.shin.values)],
}
for name, fx in FEAT.items():
    S[name] = np.nan
    for Y in range(2016, 2026):
        tr, te = (S.year < Y) & S.stands, S.year == Y
        m = LogisticRegression(C=1e4, max_iter=1000).fit(fx(S[tr]), S.won[tr])
        S.loc[te, name] = m.predict_proba(fx(S[te]))[:, 1]
S["raw"] = S.q
T = S[S.year >= 2020]
y = T.won.values
print(f"Bet365 opening, women's ITF + WTA125 + WTA Tour + Challenger, 2020-2025: {len(T) // 2} matches")
print("log-loss on all sides:  " + "  ".join(
    f"{k} {-np.mean(np.where(y, np.log(cl(T[k].values)), np.log(1 - cl(T[k].values)))):.4f}"
    for k in ("fair", "shin", "raw", "beta", "blend", "shin_b", "shin_fx")) + "  (fair/shin = market only)")


def rep(label, b):
    if len(b) < 30:
        return
    m, se = b.pnl.mean(), b.pnl.std() / np.sqrt(len(b))
    v, sv = b.pv.mean(), b.pv.std() / np.sqrt(len(b))
    w, sw = b.pv2.mean(), b.pv2.std() / np.sqrt(len(b))
    print(f"  {label:18s} {len(b):6d}  ROI {m:+6.2%} (lo {m - 1.645 * se:+5.1%})  void>30% price {v:+6.2%} (lo {v - 1.645 * sv:+5.1%})"
          f"  void>12pt prob {w:+6.2%} (lo {w - 1.645 * sw:+5.1%})")


BANDS = ((1, 2), (2, 3), (3, 5), (5, 100))
for k in ("raw", "beta", "blend", "shin_b", "shin_fx"):
    ev = T[k] * T.o - 1
    for thr in (0.03, 0.05, 0.10):
        print(f"{k} EV>{thr:.0%}")
        for lo, up in BANDS:
            rep(f"odds {lo}-{up}", T[(ev > thr) & (T.o > lo) & (T.o <= up)])
        rep("odds <=3 total", T[(ev > thr) & (T.o <= 3)])
