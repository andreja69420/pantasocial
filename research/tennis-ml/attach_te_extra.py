"""Attach the extra TennisExplorer markets to Sackmann matches: opening match odds and total-games lines
(Bet365 / Unibet / Ladbrokes, closing and opening). Writes {tour}_te_extra.parquet indexed like {tour}_f2.parquet.

usage: attach_te_extra.py RAW_DIR OUT_DIR TOUR ALI_CSV
"""
import re
import sys
import unicodedata

import pandas as pd

RAW, OUT, TOUR, ALI = sys.argv[1:5]
sys.argv = [sys.argv[0], RAW, OUT, TOUR, "1991"]
import features2  # noqa: E402


def key(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(re.sub(r"[^a-z]+", " ", s).split()))


m = features2.load_matches()
base = pd.DataFrame({"mi": m.index, "date": m.date, "w": m.winner_name.map(key).values, "l": m.loser_name.map(key).values})
a = pd.read_csv(ALI, low_memory=False)
a = a[a.tour.str.upper() == TOUR.upper()]
a = a.assign(ka=a.player_a.map(key), kb=a.player_b.map(key), d=pd.to_datetime(a.match_date_formatted, errors="coerce"))
SIDED = {"b365o": ("bet365_odds_a_opening", "bet365_odds_b_opening"), "besto": ("best_odds_a_opening", "best_odds_b_opening")}
FLAT = {}
for bk, pre in (("b365", "bet365"), ("ub", "Unibet"), ("lb", "Ladbrokes")):
    FLAT.update({f"{bk}_tline": f"{pre}_total_games_line", f"{bk}_over": f"{pre}_odds_over", f"{bk}_under": f"{pre}_odds_under",
                 f"{bk}_over_o": f"{pre}_odds_over_opening", f"{bk}_under_o": f"{pre}_odds_under_opening"})
FLAT["te_total_games"] = "total_games"
out = []
for left, right, flip in (("ka", "kb", False), ("kb", "ka", True)):
    j = base.merge(a, left_on=["w", "l"], right_on=[left, right])
    j = j[((j.d - j.date).dt.days >= -3) & ((j.d - j.date).dt.days <= 21)]
    r = pd.DataFrame({"mi": j.mi})
    for n, (ca, cb) in SIDED.items():
        r[f"{n}_w"], r[f"{n}_l"] = (j[cb], j[ca]) if flip else (j[ca], j[cb])
    for n, c in FLAT.items():
        r[n] = j[c].values
    out.append(r)
X = pd.concat(out).drop_duplicates("mi").set_index("mi").apply(pd.to_numeric, errors="coerce")
print(f"{TOUR}: {len(X)} matches; Bet365 totals {X.b365_tline.notna().sum()}, opening match odds {X.b365o_w.notna().sum()}")
X.to_parquet(f"{OUT}/{TOUR}_te_extra.parquet")
