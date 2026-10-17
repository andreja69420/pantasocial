"""Attach TennisExplorer-sourced odds (Kaggle 'Tennis Results and Betting Odds 2014-2025' and the 2026
'Pinnacle vs 1xBet' closing lines) to Sackmann lower-level and main matches.
Writes {tour}_te_odds.parquet indexed like {tour}_f2.parquet, with Bet365 (odds_w/odds_l), best, Pinnacle, 1xBet.

usage: attach_te_odds.py RAW_DIR OUT_DIR TOUR ALI_CSV SMARTODDS_CSV
"""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

RAW, OUT, TOUR, ALI, SO = sys.argv[1:6]
sys.argv = [sys.argv[0], RAW, OUT, TOUR, "1991"]
import features2  # noqa: E402  (argv must match the START used for {tour}_f2.parquet)


def key(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(re.sub(r"[^a-z]+", " ", s).split()))


m = features2.load_matches()
base = pd.DataFrame({"mi": m.index, "date": m.date, "w": m.winner_name.map(key).values, "l": m.loser_name.map(key).values})


def attach(src, a, b, date, cols_a, cols_b, names):
    """src rows have players a/b (any order); returns per-match odds oriented to winner/loser."""
    src = src.assign(ka=src[a].map(key), kb=src[b].map(key), d=pd.to_datetime(src[date], errors="coerce"))
    out = []
    for left, right, flip in (("ka", "kb", False), ("kb", "ka", True)):
        j = base.merge(src, left_on=["w", "l"], right_on=[left, right])
        j = j[((j.d - j.date).dt.days >= -3) & ((j.d - j.date).dt.days <= 21)]
        for n, ca, cb in zip(names, cols_a, cols_b):
            j[f"{n}_w"], j[f"{n}_l"] = (j[cb], j[ca]) if flip else (j[ca], j[cb])
        out.append(j)
    j = pd.concat(out).drop_duplicates("mi").set_index("mi")
    return j[[f"{n}_{s}" for n in names for s in ("w", "l")]]


ali = pd.read_csv(ALI, low_memory=False)
ali = ali[ali.tour.str.upper() == TOUR.upper()]
A = attach(ali, "player_a", "player_b", "match_date_formatted",
           ["bet365_odds_a", "best_odds_a", "Betfair_odds_a"], ["bet365_odds_b", "best_odds_b", "Betfair_odds_b"],
           ["b365", "best", "bf"])
so = pd.read_csv(SO)
so = so[so.tour.str.contains("wta|women" if TOUR == "wta" else "atp|challenger|men", case=False)]
if "home" in so.columns or "player_home" in so.columns:
    pa, pb = ("home", "away") if "home" in so.columns else ("player_home", "player_away")
else:
    pa, pb = None, None
out = A
if pa:
    S = attach(so, pa, pb, "date_utc", ["pinnacle_home", "onexbet_home"], ["pinnacle_away", "onexbet_away"], ["pin", "x1"])
    out = A.join(S, how="outer")
out["odds_w"], out["odds_l"] = out.get("b365_w"), out.get("b365_l")
out["agree"] = True
lvl = m.loc[out.index, "tourney_level"].astype(str)
print(f"{TOUR}: matches with any TE odds {len(out)}; Bet365 {out.b365_w.notna().sum()}; "
      f"Pinnacle {out['pin_w'].notna().sum() if 'pin_w' in out else 0}")
print("  by level:", lvl.value_counts().head(10).to_dict())
out.to_parquet(f"{OUT}/{TOUR}_te_odds.parquet")
