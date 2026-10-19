"""Check a new build against the live one: no Flashscore id lost, what was added, which old rows changed.

usage: cmp_build.py OLD_DIR NEW_DIR TOUR
"""
import sys

import numpy as np
import pandas as pd

OLD, NEW, TOUR = sys.argv[1:4]
cols = ["tourney_id", "tourney_name", "tourney_level", "tourney_date", "winner_id", "loser_id", "round", "score", "fs_mid", "date"]
o = pd.read_parquet(f"{OLD}/{TOUR}_matches.parquet", columns=cols)
n = pd.read_parquet(f"{NEW}/{TOUR}_matches.parquet", columns=cols)
print(f"{TOUR}: old {len(o)} rows, new {len(n)} rows ({len(n) - len(o):+d})")
of, nf = set(o.fs_mid.dropna()), set(n.fs_mid.dropna())
lost = of - nf
print(f"Flashscore ids: old {len(of)}, new {len(nf)}; lost {len(lost)}; added {len(nf - of)}")
if lost:
    L = o[o.fs_mid.isin(lost)]
    print("  lost rows by date:", L.date.dt.strftime("%Y-%m-%d").value_counts().sort_index().tail(8).to_dict())
key = ["winner_id", "loser_id", "tourney_date", "round", "score"]
ok = o.assign(k=o[key].astype(str).agg("|".join, axis=1))
nk = n.assign(k=n[key].astype(str).agg("|".join, axis=1))
gone = ~ok.k.isin(set(nk.k))
print(f"old rows (by winner, loser, date, round, score) not in the new build: {gone.sum()}; "
      f"of them with a Flashscore id lost: {ok[gone].fs_mid.isin(lost).sum()}")
if gone.sum():
    print(ok[gone & ~ok.fs_mid.isin(lost)][key + ["tourney_name"]].head(8).to_string())
new = ~nk.k.isin(set(ok.k))
A = nk[new]
print(f"new rows: {new.sum()}; by year {A.date.dt.year.value_counts().sort_index().to_dict()}")
print("new rows by level:", A.tourney_level.astype(str).value_counts().head(10).to_dict())
dup = n.duplicated(["tourney_id", "winner_id", "loser_id", "round", "score"]).sum()
print("exact duplicates left in the new build:", dup)
