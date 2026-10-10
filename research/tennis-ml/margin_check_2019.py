"""Re-settle the model's 2019 walk-forward predictions at Pinnacle and best-available odds (tennis-data.co.uk
2019 files shipped in the CRAN package sportsR). The model never sees odds; only the price we bet at changes."""
import sys

import numpy as np
import pandas as pd
import pyreadr

D, RDA, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
for tour in ["atp", "wta"]:
    P = pd.read_parquet(f"{D}/out/preds3_{tour}_ens_{TAG}_dev.parquet")
    P = P[P.year == 2019]
    f2 = pd.read_parquet(f"{D}/out/{tour}_f2.parquet", columns=["date", "odds_w", "odds_l"]).loc[P.index]
    s = list(pyreadr.read_r(f"{RDA}/{tour}_matches_2019.rda").values())[0]
    s["Date"] = pd.to_datetime(s["Date"], dayfirst=True)
    s = s.dropna(subset=["B365W", "B365L"])
    key = lambda w, l: np.round(w, 3).astype(str) + "|" + np.round(l, 3).astype(str)
    s["k"] = key(s.B365W, s.B365L)
    f2["k"] = key(f2.odds_w, f2.odds_l)
    f2.index.name = "index"
    j = f2.reset_index().merge(s[["k", "Date", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]], on="k")
    j = j[(j.Date - j.date).dt.days.between(-2, 20)].drop_duplicates("index").set_index("index")
    P = P.join(j[["PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]], how="inner")
    w_is_p1 = P.y == 1
    for b in ["PS", "Max", "Avg"]:
        P[f"{b}1"] = np.where(w_is_p1, P[f"{b}W"], P[f"{b}L"])
        P[f"{b}2"] = np.where(w_is_p1, P[f"{b}L"], P[f"{b}W"])
    P["B3651"], P["B3652"] = P.o1, P.o2
    print(f"\n{tour.upper()} 2019: {len(P)} matches matched to full odds")
    for b in ["B365", "Avg", "PS", "Max"]:
        o1, o2 = P[f"{b}1"], P[f"{b}2"]
        ok = o1.notna() & o2.notna()
        line = []
        for thr, mo in [(0.02, 100), (0.05, 100), (0.02, 3), (0.05, 3), (0.10, 3)]:
            pnl = []
            for s_, o in ((1, o1), (2, o2)):
                q = P.p if s_ == 1 else 1 - P.p
                win = (P.y == 1) if s_ == 1 else (P.y == 0)
                pick = ok & (q * o - 1 > thr) & (o <= mo)
                pnl.append(np.where(P.stands, np.where(win, o - 1, -1.0), 0.0)[pick])
            pnl = np.concatenate(pnl)
            se = pnl.std() / np.sqrt(max(len(pnl), 1))
            line.append(f"EV>{thr:.0%},odds<={mo:g}: n={len(pnl)} ROI {pnl.mean():+.1%}±{1.65 * se:.1%}")
        print(f"  {b:5s} margin {(1 / o1 + 1 / o2 - 1)[ok].median():5.2%} | " + " | ".join(line))
