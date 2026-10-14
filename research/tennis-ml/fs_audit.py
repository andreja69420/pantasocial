"""How many matches does the build lack? Every singles match Flashscore lists for a season (fs_results.py, archive page
plus the paged results feed, qualifying included) is looked up in the build (`{tour}_matches.parquet`):
  1. by Flashscore match id (the build carries fs_mid for rows joined to Flashscore stats and for all Flashscore rows),
  2. by (winner_id, loser_id) with the event date within 30 days,
  3. by player names (word order and accents ignored) with the event date within 30 days.
The count is an upper bound: players Flashscore spells differently from Sackmann get other ids and names, so a match can be
counted missing while the build holds it (fs_fill.py removes those by score).
A match found by none of them is missing. The reverse direction (build rows of the same events that Flashscore does not
list) shows where Flashscore itself is incomplete.

usage: fs_audit.py BUILD_DIR FS_CSV TOUR OUT_PREFIX [FROM_DATE TO_DATE]   (dates YYYY-MM-DD, on the event date)
Writes OUT_PREFIX_missing.csv (Flashscore matches the build lacks), OUT_PREFIX_extra.csv (build rows Flashscore lacks)
and prints the counts per level and per event.
"""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

BUILD, FSCSV, TOUR, OUT = sys.argv[1:5]
FROM = pd.Timestamp(sys.argv[5]) if len(sys.argv) > 5 else None
TO = pd.Timestamp(sys.argv[6]) if len(sys.argv) > 6 else None
WIN = pd.Timedelta(days=30)
norm = lambda s: "".join(sorted(re.findall(r"[a-z]+", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())))  # order-free


def td(s):
    return pd.to_datetime(pd.to_numeric(s, errors="coerce").astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


F = pd.read_csv(FSCSV, low_memory=False)
F["td"] = td(F.tourney_date)
B = pd.read_parquet(f"{BUILD}/{TOUR}_matches.parquet",
                    columns=["tourney_id", "tourney_name", "tourney_level", "tourney_date", "winner_id", "loser_id",
                             "winner_name", "loser_name", "round", "score", "fs_mid"])
B["td"] = td(B.tourney_date)
lo = (FROM or F.td.min()) - WIN
hi = (TO or F.td.max()) + WIN
B = B[(B.td >= lo) & (B.td <= hi)].reset_index(drop=True)
if FROM is not None:
    F = F[F.td >= FROM]
if TO is not None:
    F = F[F.td <= TO]
F = F.drop_duplicates("fs_mid").reset_index(drop=True)
print(f"{TOUR}: Flashscore {len(F)} matches, build {len(B)} rows in the window", flush=True)


def keys(D):
    return (D.winner_id.astype(str) + "|" + D.loser_id.astype(str)).values, \
           (D.winner_name.map(norm) + "|" + D.loser_name.map(norm)).values


def lookup(A, Bd, kA, kB):
    """True where a row of A has a row of Bd with the same key and the event date within WIN."""
    idx = {}
    for k, d in zip(kB, Bd.td.values):
        idx.setdefault(k, []).append(d)
    out = np.zeros(len(A), bool)
    for i, (k, d) in enumerate(zip(kA, A.td.values)):
        for e in idx.get(k, ()):
            if abs(e - d) <= WIN:
                out[i] = True
                break
    return out


fa, fb = keys(F)
ba, bb = keys(B)
by_id = F.fs_mid.isin(set(B.fs_mid.dropna().astype(str)))
by_pair = lookup(F, B, fa, ba)
by_name = lookup(F, B, fb, bb)
found = by_id | by_pair | by_name
F["found"] = found
print(f"found: by id {by_id.sum()}, by id-pair {by_pair.sum()}, by names {by_name.sum()}; in none: {(~found).sum()} "
      f"({(~found).mean():.1%})")
# reverse: build rows of Flashscore's events that Flashscore does not list
ev = set(zip(F.tourney_name.map(norm), F.td.dt.to_period("M").astype(str)))
Bk = B.assign(e=list(zip(B.tourney_name.map(norm), B.td.dt.to_period("M").astype(str))))
Bk = Bk[Bk.e.isin(ev)].reset_index(drop=True)
ra, rb = keys(Bk)
rfound = Bk.fs_mid.astype(str).isin(set(F.fs_mid)) | lookup(Bk, F, ra, fa) | lookup(Bk, F, rb, fb)
extra = Bk[~rfound]
F[~found].drop(columns=["found"]).to_csv(f"{OUT}_missing.csv", index=False)
extra.drop(columns=["e"]).to_csv(f"{OUT}_extra.csv", index=False)
print(f"build rows of the same events (name, month) that Flashscore does not list: {len(extra)} of {len(Bk)}")
T = F.groupby("tourney_level").found.agg(matches="size", missing=lambda s: int((~s).sum()))
T["missing_%"] = (T.missing / T.matches * 100).round(1)
print(T.to_string())
W = F[~found].groupby(["tourney_name", F.td.dt.strftime("%Y-%m")]).size().sort_values(ascending=False)
print("events with the most missing matches:")
print(W.head(15).to_string())
