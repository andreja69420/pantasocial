"""Copy a raw Sackmann-format directory and add the Flashscore match statistics (fs_stats.py) to it:
  - box score (ace, df, svpt, 1stIn, 1stWon, 2ndWon, SvGms, bpSaved, bpFaced) fills rows that have no serve statistics
    (Sackmann has none for ITF 2023-24 and for the Flashscore-sourced 2026 rows); rows with Sackmann stats are untouched
  - point-by-point serve counts go to separate columns w_pbp_* / l_pbp_* (svpt, svwon, SvGms, holds, bpFaced, bpSaved,
    tbpt, tbwon): they have no aces or first serves, so mixing them into the box columns would corrupt those features
  - surface normalised to Hard/Clay/Grass/Carpet (Sackmann's 2025 ITF files have lowercase 'clay'/'carpet')
Rows are matched on (winner_id, loser_id) with the Flashscore date 3 days before .. 21 days after tourney_date, nearest first.

usage: fs_join.py RAW_IN RAW_OUT STATS_DIR
"""
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

RAW_IN, RAW_OUT, ST = sys.argv[1:4]
BOX = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]
PBP = ["svpt", "svwon", "SvGms", "holds", "bpFaced", "bpSaved", "tbpt", "tbwon"]

os.makedirs(os.path.join(RAW_OUT, "sackmann_raw"), exist_ok=True)
for d in glob.glob(os.path.join(RAW_IN, "*")):
    if os.path.basename(d) != "sackmann_raw" and not os.path.exists(os.path.join(RAW_OUT, os.path.basename(d))):
        os.symlink(os.path.abspath(d), os.path.join(RAW_OUT, os.path.basename(d)))

for tour in ("wta", "atp"):
    S = pd.read_parquet(os.path.join(ST, f"{tour}_fs_stats.parquet"))
    S["d"] = S.date.dt.normalize()
    for f in sorted(glob.glob(os.path.join(RAW_IN, "sackmann_raw", f"{tour}__{tour}_matches_*.csv"))):
        out = os.path.join(RAW_OUT, "sackmann_raw", os.path.basename(f))
        M = pd.read_csv(f, low_memory=False)
        M["surface"] = M.surface.str.capitalize()
        for c in [f"{s}_{c}" for s in ("w", "l") for c in BOX]:
            if c not in M:
                M[c] = np.nan
        td = pd.to_datetime(M.tourney_date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        if td.max() < S.d.min() - pd.Timedelta(days=21):
            M.to_csv(out, index=False)
            continue
        x = M.reset_index()[["index", "winner_id", "loser_id"]].assign(td=td.values)
        x["winner_id"], x["loser_id"] = x.winner_id.astype(str), x.loser_id.astype(str)
        y = S.assign(winner_id=S.winner_id.astype(str), loser_id=S.loser_id.astype(str))
        j = x.merge(y, on=["winner_id", "loser_id"])
        j["gap"] = (j.d - j.td).dt.days
        j = j[j.gap.between(-3, 21)].assign(a=lambda t: t.gap.abs()).sort_values("a")
        j = j.drop_duplicates("index").drop_duplicates("fs_mid").set_index("index")
        has = M[["w_svpt", "l_svpt"]].apply(pd.to_numeric, errors="coerce").gt(0).all(axis=1)
        fb = j.index[j.w_SvGms.notna() & j.l_SvGms.notna()]
        fb = fb[~has.loc[fb].values]
        for s in ("w", "l"):
            for c in BOX:
                M.loc[fb, f"{s}_{c}"] = j.loc[fb, f"{s}_{c}"].values
            for c in PBP:
                M[f"{s}_pbp_{c}"] = np.nan
                M.loc[j.index, f"{s}_pbp_{c}"] = j[f"{s}_pbp_{c}"].values
        M.to_csv(out, index=False)
        print(f"{os.path.basename(f)}: {len(M)} rows, matched {len(j)}, box filled {len(fb)} "
              f"(had stats {int(has.sum())}), pbp {int(M.w_pbp_SvGms.notna().sum())}", flush=True)
