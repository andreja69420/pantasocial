"""Bet365 opening-odds strategy on the TennisExplorer data (2020-2025, walk-forward predictions): EV ladder, odds
bands and a void stress test (bets whose Bet365 price was cut by more than 30% before the close count as refunded).

usage: b365_open.py DATA_DIR [FROM_YEAR=2020]
"""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
FROM = int(sys.argv[2]) if len(sys.argv) > 2 else 2020
TE = "2014-2015-2016-2017-2018-2019-2020-2021-2022-2023-2024-2025"
SEGS = (("Women ITF", "wta", r"^\d"), ("WTA125", "wta", r"^C$"), ("WTA Tour", "wta", r"^[GPIF]"),
        ("Challenger", "atp", r"^C$"), ("ATP Tour", "atp", r"^[GMAF]"))
rows = []
for seg, tour, rx in SEGS:
    P = pd.read_parquet(f"{D}/out/preds3_{tour}_ens_r6_hist1991_ens_te_{TE}.parquet")
    E = pd.read_parquet(f"{D}/out/{tour}_te_extra.parquet").reindex(P.index)
    C = pd.read_parquet(f"{D}/out/{tour}_te_odds.parquet").reindex(P.index)
    F = pd.read_parquet(f"{D}/out/{tour}_f2.parquet", columns=["tourney_level", "date"]).reindex(P.index)
    keep = F.tourney_level.astype(str).str.match(rx).fillna(False).values & (P.year >= FROM).values
    P, E, C, F = P[keep], E[keep], C[keep], F[keep]
    ow, ol, cw, cl = (E.b365o_w.astype(float).values, E.b365o_l.astype(float).values,
                      C.b365_w.astype(float).values, C.b365_l.astype(float).values)
    yw = (P.y == 1).values  # p is P(player1 wins); winner/loser odds -> player1/player2 sides
    for side in (1, 2):
        won = yw if side == 1 else ~yw
        o = np.where(won, ow, ol)
        c = np.where(won, cw, cl)
        q = P.p.values if side == 1 else 1 - P.p.values
        rows.append(pd.DataFrame({"seg": seg, "year": P.year.values, "day": F.date.values, "q": q, "o": o, "c": c,
                                  "pnl": np.where(P.stands.values, np.where(won, o - 1, -1.0), 0.0)}))
S = pd.concat(rows, ignore_index=True)
S = S[(S.o > 1) & (S.c > 1)]
S["ev"] = S.q * S.o - 1
S["pv"] = np.where(S.o / S.c > 1.3, 0.0, S.pnl)
days = (S.day.max() - S.day.min()).days + 1


def rep(name, b):
    m, se = b.pnl.mean(), b.pnl.std() / np.sqrt(len(b))
    v, sv = b.pv.mean(), b.pv.std() / np.sqrt(len(b))
    yrs = " ".join(f"{b[b.year == y].pnl.mean():+.0%}" for y in sorted(b.year.unique()))
    print(f"{name:30s} {len(b):6d} {len(b) / days:5.1f}/d  ROI {m:+6.2%} (lo {m - 1.645 * se:+5.1%})  "
          f"void-stress {v:+6.2%} (lo {v - 1.645 * sv:+5.1%})  big movers {(b.o / b.c > 1.3).mean():4.1%} | {yrs}")


print(f"Bet365 opening prices, {FROM}-2025 ({days} days); by year {sorted(S.year.unique())}")
rep("random sides odds<=2", S[S.o <= 2])
for seg, g in S.groupby("seg"):
    rep(f"{seg} EV>5% odds<=2", g[(g.ev > 0.05) & (g.o <= 2)])
core = ["Women ITF", "Challenger", "WTA125", "WTA Tour"]
G = S[S.seg.isin(core)]
print("\npooled women's ITF + Challenger + WTA125 + WTA Tour:")
for cap in (2.0, 2.5, 3.0):
    for thr in (0.05, 0.10, 0.16, 0.25):
        rep(f"EV>{thr:.0%} odds<={cap}", G[(G.ev > thr) & (G.o <= cap)])
print("\nodds bands, EV>5%:")
for lo, hi in ((1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 100)):
    rep(f"odds {lo}-{hi}", G[(G.ev > 0.05) & (G.o > lo) & (G.o <= hi)])
