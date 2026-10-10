"""Out-of-sample check of the betting-stage Shin blend: P = logistic(a*logit(model) + b*logit(Shin fair opening) + c)
fitted ONLY on TennisExplorer Bet365 opening prices 2014-2024 (women's ITF, WTA125, WTA Tour, Challenger), then
applied unchanged to the fresh BetMGM.us 2025-26 opening prices (women's ITF, Challenger, men's ITF).

usage: blend_transfer.py DATA_DIR
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.argv = [sys.argv[0], sys.argv[1]]
D = sys.argv[1]
src = open(__file__.replace("blend_transfer.py", "calib_open.py")).read()
exec(src.split("cl = lambda p:")[0].split('"""', 2)[2])  # builds S (TE Bet365 sides) and shin()
cl = lambda p: np.clip(p, 1e-4, 1 - 1e-4)
lg = lambda p: np.log(cl(p) / (1 - cl(p)))
tr = S[(S.year <= 2024) & S.stands]
m = LogisticRegression(C=1e4).fit(np.c_[lg(tr.q.values), lg(tr.shin.values)], tr.won.values)
print(f"blend fitted on Bet365 2014-2024 ({len(tr)} sides): model {m.coef_[0][0]:.2f}, market {m.coef_[0][1]:.2f}, "
      f"const {m.intercept_[0]:+.2f}")
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
T = pd.concat(rows, ignore_index=True)
T["shin"] = shin(1 / T.o.values, 1 / T.oo.values)
T["qb"] = m.predict_proba(np.c_[lg(T.q.values), lg(T.shin.values)])[:, 1]
T["ev"], T["evb"] = T.q * T.o - 1, T.qb * T.o - 1
T["pv2"] = np.where((1 / T.c) / (1 / T.c + 1 / T.cc) - (1 / T.o) / (1 / T.o + 1 / T.oo) > 0.12, 0.0, T.pnl)


def rep(name, b):
    m_, se = b.pnl.mean(), b.pnl.std() / np.sqrt(len(b))
    print(f"{name:30s} {len(b):5d}  ROI {m_:+6.2%} (lo {m_ - 1.645 * se:+5.1%})  void>12pt {b.pv2.mean():+6.2%} | "
          + " ".join(f"{y}: {b[b.year == y].pnl.mean():+.1%}({(b.year == y).sum()})" for y in (2025, 2026)) + " | "
          + " ".join(f"{s[:5]} {g.pnl.mean():+.1%}" for s, g in b.groupby("seg")))


print("BetMGM.us opening 2025-26, women's ITF + Challenger + men's ITF:")
rep("raw EV>5% odds<=2", T[(T.ev > 0.05) & (T.o <= 2)])
for cap in (2, 3, 5):
    for thr in (0.03, 0.05, 0.10):
        rep(f"blend EV>{thr:.0%} odds<={cap}", T[(T.evb > thr) & (T.o <= cap)])
