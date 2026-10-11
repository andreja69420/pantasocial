"""Fresh test, June-October 2026: matches after the end of the Sackmann archive (Flashscore results), results-only
model trained to 2024 and calibrated on 2025, priced at each bookmaker's OPENING odds from Flashscore (UK region),
settled on the result; the current price of a finished match is its closing price.
Rules: raw model EV and the betting-stage blend fitted on Bet365 2014-2024 (0.44 logit model + 0.64 logit Shin fair).

usage: eval_fs2026.py DATA_DIR      (expects live/out = out_live with preds3_{tour}_ens_r6_bxlive_lower_2026.parquet)
"""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
SEG = {"wta": [("Women ITF", r"^\d"), ("WTA Tour", r"^[PGIF]")], "atp": [("Men ITF", r"^\d"), ("Challenger", r"^C$"), ("ATP Tour", r"^A$")]}


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
rows = []
for tour, segs in SEG.items():
    P = pd.read_parquet(f"{D}/out_live/preds3_{tour}_ens_r6_bxlive_lower_2026.parquet")
    M = pd.read_parquet(f"{D}/out_live/{tour}_matches.parquet", columns=["tourney_date", "winner_id", "loser_id", "tourney_level", "date"])
    fs = pd.read_csv(f"{D}/fs_raw/{tour}__{tour}_matches_fs_2026.csv", usecols=["tourney_date", "winner_id", "loser_id", "fs_mid"])
    key = lambda d: d.tourney_date.astype(int).astype(str) + "|" + d.winner_id.astype(int).astype(str) + "|" + d.loser_id.astype(int).astype(str)
    M["fs_mid"] = key(M).map(dict(zip(key(fs), fs.fs_mid)))
    O = pd.read_parquet(f"{D}/fs_raw/{tour}_fs_odds_GB.parquet")
    P = P.join(M, how="inner").dropna(subset=["fs_mid"])
    J = P.reset_index().merge(O, on="fs_mid")
    # orient home/away prices to winner/loser
    ow = np.where(J.home_won, J.open_h, J.open_a); ol = np.where(J.home_won, J.open_a, J.open_h)
    cw = np.where(J.home_won, J.cur_h, J.cur_a); cl = np.where(J.home_won, J.cur_a, J.cur_h)
    lvl = J.tourney_level.astype(str)
    seg = np.full(len(J), "", dtype=object)
    for name, rx in segs:
        seg[lvl.str.match(rx).values & (seg == "")] = name
    for won in (True, False):
        q = np.where((J.y == 1) == won, J.p, 1 - J.p)
        o, oo, c, cc = (ow, ol, cw, cl) if won else (ol, ow, cl, cw)
        rows.append(pd.DataFrame({"seg": seg, "book": J.book.values, "mi": J["index"].values, "date": J.date.values,
                                  "q": q, "o": o, "oo": oo, "c": c, "cc": cc, "won": won,
                                  "pnl": np.where(J.stands.values, np.where(won, o - 1, -1.0), 0.0)}))
S = pd.concat(rows, ignore_index=True)
S = S[(S.seg != "") & (S.o > 1) & (S.oo > 1) & (S.c > 1) & (S.cc > 1)].copy()
S["shin"] = shin(1 / S.o.values, 1 / S.oo.values)
S["qb"] = 1 / (1 + np.exp(-(0.44 * lg(S.q.values) + 0.64 * lg(S.shin.values))))
S["ev"], S["evb"] = S.q * S.o - 1, S.qb * S.o - 1
S["clv"] = S.o / S.c - 1
fo, fc = (1 / S.o) / (1 / S.o + 1 / S.oo), (1 / S.c) / (1 / S.c + 1 / S.cc)
S["pv"] = np.where(fc - fo > 0.12, 0.0, S.pnl)
S["pnl_c"] = np.where(S.pnl == 0, 0.0, np.where(S.won, S.c - 1, -1.0))
print(f"fresh test {pd.Timestamp(S.date.min()):%Y-%m-%d}..{pd.Timestamp(S.date.max()):%Y-%m-%d}: {S.mi.nunique()} matches with odds; "
      f"books with most matches: {S.groupby('book').mi.nunique().sort_values(ascending=False).head(8).to_dict()}")


def rep(lab, b):
    if len(b) < 30:
        return
    m, se = b.pnl.mean(), b.pnl.std() / np.sqrt(len(b))
    segs = " ".join(f"{s[:5]} {g.pnl.mean():+.1%}({len(g)})" for s, g in b.groupby("seg"))
    print(f"  {lab:28s} {len(b):5d}  ROI@open {m:+6.2%} (lo {m - 1.645 * se:+5.1%})  @close {b.pnl_c.mean():+6.2%}  "
          f"void {b.pv.mean():+6.2%}  CLV {b.clv.mean():+5.2%} | {segs}")


for book in ["bet365", "BetMGM.uk", "William Hill", "Unibetuk", "Betfair", "Pinnacle"]:
    B_ = S[S.book == book]
    if B_.mi.nunique() < 200:
        continue
    print(f"\n== {book}: {B_.mi.nunique()} matches, median margin {np.median(1 / B_.o + 1 / B_.oo - 1):.1%}")
    rep("random sides odds<=2", B_[B_.o <= 2])
    rep("raw EV>5% odds<=2", B_[(B_.ev > 0.05) & (B_.o <= 2)])
    for cap in (2, 5):
        for thr in (0.05, 0.10, 0.15):
            rep(f"blend EV>{thr:.0%} odds<={cap}", B_[(B_.evb > thr) & (B_.o <= cap)])
# best opening price across books for each side
best = S.sort_values("o", ascending=False).drop_duplicates(["mi", "won"])
print(f"\n== best opening price across books: {best.mi.nunique()} matches")
rep("blend EV>5% odds<=2", best[(best.evb > 0.05) & (best.o <= 2)])
rep("blend EV>5% odds<=5", best[(best.evb > 0.05) & (best.o <= 5)])
