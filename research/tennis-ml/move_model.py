"""Line-movement model: predict how much a side's fair win probability will rise between the Bet365 opening and
closing price, from information available at the opening only (model probability, Shin fair opening probability,
their gap, odds level, segment). Walk-forward by year on TE data (train on years before Y, 2014+), then:
  - TE Bet365 2020-2025: bet when the predicted move > threshold (and the side's blend EV > 0)
  - transfer: model fitted on TE 2014-2024 applied to the fresh BetMGM 2025-26 opening prices.

usage: move_model.py DATA_DIR
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

D = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "calib_open.py")).read().split("cl = lambda p:")[0].split('"""', 2)[2])  # S, shin()
from sklearn.linear_model import LogisticRegression  # noqa: E402

cl = lambda p: np.clip(p, 1e-4, 1 - 1e-4)
lg = lambda p: np.log(cl(p) / (1 - cl(p)))
SEGN = {"Women ITF": 0, "WTA125": 1, "WTA Tour": 2, "Challenger": 3, "Men ITF": 4}


def feats(d):
    return pd.DataFrame({"lq": lg(d.q.values), "ls": lg(d.shin.values), "gap": d.q.values - d.shin.values,
                         "lgap": lg(d.q.values) - lg(d.shin.values), "o": d.o.values, "seg": d.seg.map(SEGN).values})


S["move"] = (1 / S.c) / (1 / S.c + 1 / S.cc) - S.fair
S["pred"] = np.nan
params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
              verbose=-1, n_jobs=2)
for Y in range(2016, 2026):
    tr, te = S.year < Y, S.year == Y
    m = lgb.LGBMRegressor(**params).fit(feats(S[tr]), S.move[tr])
    S.loc[te, "pred"] = m.predict(feats(S[te]))
bl = LogisticRegression(C=1e4).fit(np.c_[lg(S.q[S.year <= 2024]), lg(S.shin[S.year <= 2024])], S.won[S.year <= 2024])
S["evb"] = bl.predict_proba(np.c_[lg(S.q), lg(S.shin)])[:, 1] * S.o - 1
T = S[S.year >= 2020]
print(f"TE Bet365 2020-2025: corr(predicted, actual move) {np.corrcoef(T.pred, T.move)[0, 1]:.3f}; "
      f"corr(model-market gap, move) {np.corrcoef(T.q - T.shin, T.move)[0, 1]:.3f}")


def rep(lab, b):
    m_, se = b.pnl.mean(), b.pnl.std() / np.sqrt(max(len(b), 1))
    print(f"  {lab:34s} {len(b):6d}  ROI {m_:+6.2%} (lo {m_ - 1.645 * se:+5.1%})  void>12pt {b.pv2.mean():+6.2%}  "
          f"actual move {b.move.mean():+.3f}")


for cap in (2, 5):
    print(f"odds <= {cap}:")
    rep("blend EV>5% (reference)", T[(T.evb > 0.05) & (T.o <= cap)])
    for thr in (0.01, 0.02, 0.03, 0.05):
        rep(f"predicted move > {thr:.2f}", T[(T.pred > thr) & (T.o <= cap)])
        rep(f"predicted move > {thr:.2f} & blend EV>0", T[(T.pred > thr) & (T.evb > 0) & (T.o <= cap)])

# transfer to the fresh BetMGM 2025-26 data
m = lgb.LGBMRegressor(**params).fit(feats(S[S.year <= 2024]), S.move[S.year <= 2024])
SEGS = (("Women ITF", "wta", ["bxo25_witf", "bxo_odds"], r"^\d"), ("Challenger", "atp", ["bxo25_chal", "bxo_odds_chal"], r"^C$"),
        ("Men ITF", "atp", ["bxo25_mitf", "bxo_odds_mitf"], r"^(\d|S)"))
rows = []
for seg, tour, bxo, rx in SEGS:
    P = pd.concat([pd.read_parquet(f"{D}/out/preds3_{tour}_ens_r6_hist1991_ens_lower_{y}.parquet") for y in (2025, 2026)])
    O = pd.concat([pd.read_parquet(f"{D}/out/{tour}_{n}.parquet") for n in bxo])
    P, O = P[~P.index.duplicated()], O[~O.index.duplicated()]
    F = pd.read_parquet(f"{D}/out/{tour}_f2.parquet", columns=["tourney_level"])
    P = P[P.year.isin([2025, 2026])].join(O, how="inner")
    P = P[F.tourney_level.astype(str).reindex(P.index).str.match(rx).fillna(False).values & (P.open_w > 1) & (P.open_l > 1)
          & (P.close_w > 1) & (P.close_l > 1)]
    for won in (True, False):
        o, oo = (P.open_w, P.open_l) if won else (P.open_l, P.open_w)
        c, cc = (P.close_w, P.close_l) if won else (P.close_l, P.close_w)
        st = P.stands.values & ~P.retired.values
        rows.append(pd.DataFrame({"seg": seg, "year": P.year.values, "q": np.where((P.y == 1) == won, P.p, 1 - P.p),
                                  "o": o.values, "oo": oo.values, "c": c.values, "cc": cc.values,
                                  "pnl": np.where(st, np.where(won, o.values - 1, -1.0), 0.0)}))
M = pd.concat(rows, ignore_index=True)
M["shin"] = shin(1 / M.o.values, 1 / M.oo.values)
M["fair"] = (1 / M.o) / (1 / M.o + 1 / M.oo)
M["move"] = (1 / M.c) / (1 / M.c + 1 / M.cc) - M.fair
M["pv2"] = np.where(M.move > 0.12, 0.0, M.pnl)
M["pred"] = m.predict(feats(M))
M["evb"] = bl.predict_proba(np.c_[lg(M.q), lg(M.shin)])[:, 1] * M.o - 1
print(f"\nBetMGM 2025-26 (move model fitted on TE Bet365 2014-2024): corr(predicted, actual move) {np.corrcoef(M.pred, M.move)[0, 1]:.3f}")
for cap in (2, 5):
    print(f"odds <= {cap}:")
    rep("blend EV>5% (reference)", M[(M.evb > 0.05) & (M.o <= cap)])
    for thr in (0.01, 0.02, 0.03, 0.05):
        rep(f"predicted move > {thr:.2f}", M[(M.pred > thr) & (M.o <= cap)])
        rep(f"predicted move > {thr:.2f} & blend EV>0", M[(M.pred > thr) & (M.evb > 0) & (M.o <= cap)])
