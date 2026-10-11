"""Whole-2026 test per tournament level at the best UK opening price: every 2026 prediction (model trained <= 2024,
calibrated on 2025, never saw 2026) matched to its Flashscore match and odds. Rows from the Sackmann archive (to June)
are matched on (winner_id, loser_id) with the match 3 days before .. 21 days after tourney_date, nearest first; tour
events from Flashscore (all labelled P / A there) get the level of the same event in the Sackmann archive by name.

usage: eval_levels2026.py DATA_DIR FS_DIR TAG:OUT_SUBDIR [TAG:OUT_SUBDIR ...]
  e.g. eval_levels2026.py $B $B/fs_full r6_v2:out_fx r6_v3:out_v3
  --bets=PATH: also save every side at its best opening price (tour, tag, level, model q, odds, blend EV, P&L)
  --book=NAME: that bookmaker's prices only; --close: closing prices (last before the start) instead of opening
  FS_DIR: fs_results.py / fs_odds.py output for 2026-01-01.. ({tour}__{tour}_matches_fs_2026.csv, {tour}_fs_odds_GB.parquet)
"""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

D, FS = sys.argv[1], sys.argv[2]
BETS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--bets=")), None)  # save every side priced
BOOK = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--book=")), None)  # one bookmaker only
CLOSE = "--close" in sys.argv  # the last price before the start instead of the opening price
RUNS = [a.split(":") for a in sys.argv[3:] if not a.startswith("--")]
NAMES = {"wta": {"G": "Grand Slam", "PM": "WTA 1000", "P": "WTA 500", "I": "WTA 250", "C": "WTA 125", "100": "W100",
                 "75": "W75", "50": "W50", "35": "W35", "15": "W15"},
         "atp": {"G": "Grand Slam", "M": "Masters 1000", "A": "ATP 250/500", "F": "Finals", "C": "Challenger",
                 "25": "M25", "15": "M15"}}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
norm = lambda s: re.sub(r"[^a-z]+", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())
# Flashscore and Sackmann spell some tour events differently
ALIAS = {"rolandgarros": "frenchopen", "usopen": "usopen", "australianopen": "australianopen", "wimbledon": "wimbledon",
         "cincinnati": "cincinnatimasters", "montecarlo": "montecarlomasters", "shanghai": "shanghaimasters",
         "paris": "parismasters", "madrid": "madridmasters", "rome": "romemasters", "indianwells": "indianwellsmasters",
         "miami": "miamimasters", "toronto": "canadamasters", "montreal": "canadamasters"}


def shin(a, b):
    lo, hi = np.zeros_like(a), np.full_like(a, 0.5)
    B = a + b
    f = lambda z, x: (np.sqrt(z ** 2 + 4 * (1 - z) * x ** 2 / B) - z) / (2 * (1 - z))
    for _ in range(50):
        z = (lo + hi) / 2
        s = f(z, a) + f(z, b)
        hi, lo = np.where(s < 1, z, hi), np.where(s < 1, lo, z)
    return f((lo + hi) / 2, a)


out, cover, saved = [], [], []
for tour in ("atp", "wta"):
    M = pd.read_parquet(f"{D}/{RUNS[0][1]}/{tour}_matches.parquet",
                        columns=["tourney_id", "tourney_name", "tourney_date", "tourney_level", "winner_id", "loser_id"])
    fs = pd.read_csv(f"{FS}/{tour}__{tour}_matches_fs_2026.csv", usecols=["fs_mid", "winner_id", "loser_id", "tourney_level", "tourney_name"])
    O = pd.read_parquet(f"{FS}/{tour}_fs_odds_GB.parquet")
    if BOOK:
        O = O[O.book == BOOK]
    if CLOSE:
        O = O.assign(open_h=O.cur_h, open_a=O.cur_a)
    fs = fs.merge(O.drop_duplicates("fs_mid")[["fs_mid", "start"]], on="fs_mid")
    fs["d"] = pd.to_datetime(fs.start, unit="s").dt.normalize()
    # level of Flashscore tour events from the archive's name -> level (2025-26)
    arch = M[~M.tourney_id.astype(str).str.startswith("fs-") & (M.tourney_date >= 20250101)]
    arch = arch[arch.tourney_level.astype(str).isin(["G", "M", "PM", "P", "I", "A", "F"])]
    lv_by_name = arch.assign(n=arch.tourney_name.map(norm)).groupby("n").tourney_level.agg(lambda s: s.astype(str).mode()[0]).to_dict()

    def level(row_lvl, name):
        if row_lvl not in ("P", "A"):
            return row_lvl
        n = norm(name)
        n = ALIAS.get(n, n)
        return lv_by_name.get(n, lv_by_name.get(n + "masters", row_lvl))

    preds = {}
    for tag, od in RUNS:
        preds[tag] = pd.read_parquet(f"{D}/{od}/preds3_{tour}_ens_{tag}_lower_2026.parquet")[["p", "y", "stands"]]
    idx = preds[RUNS[0][0]].index
    for tag in preds:
        idx = idx.intersection(preds[tag].index)
    X = M.loc[idx].rename_axis("index").reset_index()
    X["td"] = pd.to_datetime(X.tourney_date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    X["lvl"] = [level(str(l), n) for l, n in zip(X.tourney_level, X.tourney_name)]
    j = X[["index", "winner_id", "loser_id", "td"]].astype({"winner_id": str, "loser_id": str}).merge(
        fs.astype({"winner_id": str, "loser_id": str})[["fs_mid", "winner_id", "loser_id", "d"]], on=["winner_id", "loser_id"])
    j["gap"] = (j.d - j.td).dt.days
    j = j[j.gap.between(-3, 21)].assign(a=lambda t: t.gap.abs()).sort_values("a").drop_duplicates("index").drop_duplicates("fs_mid")
    X = X.merge(j[["index", "fs_mid"]], on="index", how="left")
    for l, g in X.groupby("lvl"):
        cover.append({"tour": tour, "lvl": l, "matches": len(g), "with_odds": int(g.fs_mid.notna().sum())})
    J = X.dropna(subset=["fs_mid"]).merge(O, on="fs_mid")
    ow = np.where(J.home_won, J.open_h, J.open_a)
    ol = np.where(J.home_won, J.open_a, J.open_h)
    for tag in preds:
        P = preds[tag]
        pX = P.loc[X["index"]]
        ll = -(pX.y.values * np.log(pX.p.values) + (1 - pX.y.values) * np.log(1 - pX.p.values))
        for l, v in pd.Series(ll).groupby(X.lvl.values):
            out.append({"tour": tour, "lvl": l, "tag": tag, "ll": v.mean()})
        pJ = P.loc[J["index"]]
        R = []
        for won in (True, False):
            q = np.where((pJ.y.values == 1) == won, pJ.p.values, 1 - pJ.p.values)
            o, oo = (ow, ol) if won else (ol, ow)
            R.append(pd.DataFrame({"lvl": J.lvl.values, "mi": J["index"].values, "q": q, "o": o, "oo": oo, "won": won,
                                   "pnl": np.where(pJ.stands.values, np.where(won, o - 1, -1.0), 0.0)}))
        S = pd.concat(R, ignore_index=True)
        S = S[(S.o > 1) & (S.oo > 1)].copy()
        S["evb"] = 1 / (1 + np.exp(-(0.44 * lg(S.q.values) + 0.64 * lg(shin(1 / S.o.values, 1 / S.oo.values))))) * S.o - 1
        best = S.sort_values("o", ascending=False).drop_duplicates(["mi", "won"])
        saved.append(best.assign(tour=tour, tag=tag, ev=best.q * best.o - 1))
        for cap in (2, 5):
            b = best[(best.evb > 0.05) & (best.o <= cap)]
            for l, g in list(b.groupby("lvl")) + [("ALL", b)]:
                m, se = g.pnl.mean(), g.pnl.std() / np.sqrt(len(g))
                out.append({"tour": tour, "lvl": l, "tag": tag, f"n{cap}": len(g), f"roi{cap}": m, f"lo{cap}": m - 1.645 * se})

C = pd.DataFrame(cover).set_index(["tour", "lvl"])
C.columns = pd.MultiIndex.from_tuples([(c, "") for c in C.columns])
T = pd.DataFrame(out).groupby(["tour", "lvl", "tag"]).first().unstack("tag")
T = C.join(T, how="outer")
order = {l: i for i, l in enumerate(["G", "F", "M", "PM", "A", "P", "I", "C", "100", "75", "50", "35", "25", "15", "ALL"])}
T = T.iloc[sorted(range(len(T)), key=lambda i: (T.index[i][0], order.get(T.index[i][1], 99)))]
T.index = [f"{t} {NAMES[t].get(l, l)}" for t, l in T.index]
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 40)
print(T.round(4).to_string())
if BETS:
    pd.concat(saved, ignore_index=True).to_parquet(BETS)
