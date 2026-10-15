"""Give every Flashscore Challenger its own event id (fs-ch-<slug>-<year>) and the right start date.

Flashscore ids were fs-<slug>-<year>: a city that holds a tour event and a Challenger in one season (Shanghai: the
September Challenger and the October Masters; Mallorca: the June ATP 250 and the August Challenger) got ONE id, whose date
is the first match of both, so the Masters matches were dated 2026-09-06 and sorted before the Beijing and Davis Cup
matches the players had played earlier (rest days, form and ratings of the Shanghai Masters players were wrong).
fs_common.event_id now puts "ch" into Challenger ids; this script migrates the raw files built with the old ids.

usage: fs_fix_ids.py RAW_IN RAW_OUT DATES_ATP.csv DATES_WTA.csv
  RAW_IN/sackmann_raw: the raw dir to migrate; RAW_OUT/sackmann_raw gets symlinks to everything and a rewritten real copy of
  every Flashscore file (*_fs_*, *_fsbf_*, *_fsloop_*, *_fsfill_*). DATES_*: fs_results.py {tour}_match_dates.csv
  (fs_mid, tourney_id, tourney_level, date) of a fresh scrape with the new ids; the start date of an event whose old id was
  shared (below) is the first match day of its new id.
"""
import glob
import os
import re
import sys

import pandas as pd

RAW_IN, RAW_OUT, DATES = sys.argv[1], sys.argv[2], {"atp": sys.argv[3], "wta": sys.argv[4]}
SHARED = {"fs-shanghai-2026", "fs-mallorca-2026"}  # tour event and Challenger under one old id
os.makedirs(f"{RAW_OUT}/sackmann_raw", exist_ok=True)
for f in glob.glob(f"{RAW_IN}/*"):
    if os.path.basename(f) != "sackmann_raw" and not os.path.lexists(f"{RAW_OUT}/{os.path.basename(f)}"):
        os.symlink(os.path.realpath(f), f"{RAW_OUT}/{os.path.basename(f)}")
tot = {}
for f in sorted(glob.glob(f"{RAW_IN}/sackmann_raw/*")):
    b = os.path.basename(f)
    dst = f"{RAW_OUT}/sackmann_raw/{b}"
    m = re.match(r"(atp|wta)__\1_matches_(fs|fsbf|fsloop|fsfill)_\d{4}\.csv$", b)
    if not m:
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(f), dst)
        continue
    if os.path.lexists(dst):
        os.remove(dst)
    D = pd.read_csv(f, low_memory=False)
    ch = D.tourney_id.astype(str).str.startswith("fs-") & ~D.tourney_id.astype(str).str.startswith("fs-ch-") & (D.tourney_level.astype(str) == "C")
    old = D.tourney_id.copy()
    D.loc[ch, "tourney_id"] = D.loc[ch, "tourney_id"].str.replace(r"^fs-", "fs-ch-", regex=True)
    sh = old.isin(SHARED)
    if sh.any():
        T = pd.read_csv(DATES[m.group(1)], low_memory=False)
        start = pd.to_datetime(T.groupby("tourney_id").date.min()).dt.strftime("%Y%m%d").astype(int)
        new = D.loc[sh, "tourney_id"].map(start)
        D.loc[sh, "tourney_date"] = new.fillna(D.loc[sh, "tourney_date"]).astype(int)
        print(f"{b}: dates of {int(sh.sum())} rows of {sorted(set(old[sh]))} set from the fresh scrape "
              f"({int(new.isna().sum())} not found there)", flush=True)
    D.to_csv(dst, index=False)
    tot[b] = int(ch.sum())
print("rows moved to fs-ch- ids per file:", {k: v for k, v in tot.items() if v})
