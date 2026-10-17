"""Attach Challenger / ITF moneyline odds (Kaggle 'A large Tennis dataset for ATP and ITF betting', 2009-2018)
to Sackmann matches. Writes {tour}_lower_odds.parquet indexed like {tour}_f2.parquet (same load_matches()).

usage: attach_lower_odds.py RAW_DIR OUT_DIR TOUR KAGGLE_DIR
"""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

RAW, OUT, TOUR, KG = sys.argv[1:5]
sys.argv = [sys.argv[0], RAW, OUT, TOUR, "1991"]
import features2  # noqa: E402  (reads argv at import; must match the START used for {tour}_f2.parquet)


def slug(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


m = features2.load_matches()
m = m[m.main == 0]
b = pd.read_csv(f"{KG}/betting_moneyline.csv")
b = b[~b.tournament.str.contains("junior", case=False)]
b["start"] = pd.to_datetime(b.start_date)
# one price per match: prefer Pinnacle, else the OddsPortal line (the only book with broad lower-level coverage)
b["pref"] = b.book_name.map({"Pinnacle Sports": 0, "OddsPortal": 1}).fillna(2)
b = b.sort_values("pref").drop_duplicates(["start_date", "tournament", "team1", "team2"])
b["o1"], b["o2"] = 1 / b.odds1, 1 / b.odds2  # 'odds' columns are implied probabilities in this file

ws, ls = m.winner_name.map(slug), m.loser_name.map(slug)
idx = pd.DataFrame({"mi": m.index, "date": m.date, "w": ws.values, "l": ls.values})
fwd = idx.merge(b[["start", "team1", "team2", "o1", "o2", "book_name"]], left_on=["w", "l"], right_on=["team1", "team2"])
fwd["odds_w"], fwd["odds_l"] = fwd.o1, fwd.o2
rev = idx.merge(b[["start", "team1", "team2", "o1", "o2", "book_name"]], left_on=["l", "w"], right_on=["team1", "team2"])
rev["odds_w"], rev["odds_l"] = rev.o2, rev.o1
j = pd.concat([fwd, rev])
j = j[(j.start - j.date).dt.days.abs() <= 7]
j = j.drop_duplicates("mi").set_index("mi")[["odds_w", "odds_l", "book_name"]]
j["agree"] = True
lvl = m.loc[j.index, "tourney_level"].astype(str)
print(f"{TOUR}: lower-level matches {len(m)}, with odds {len(j)}; by level:", lvl.value_counts().head(8).to_dict())
print("margin median", round(float((1 / j.odds_w + 1 / j.odds_l - 1).median()), 4), j.book_name.value_counts().to_dict())
j.to_parquet(f"{OUT}/{TOUR}_lower_odds.parquet")
