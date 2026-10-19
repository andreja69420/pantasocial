"""One-off repair of match files written before fs_common.ALIASES existed: rows of a player whose Flashscore name has an
alias carry a synthetic id (>= 9e8) and the Flashscore name; this gives them the real id and the history's name.

usage: fix_aliases.py RAW_DIR FILE [FILE ...]   (RAW_DIR = the Sackmann files for ids, e.g. DATA_DIR/live_raw; CSV or
parquet files with winner_/loser_ id and name columns, rewritten in place; symlinks are followed)
"""
import os
import sys

import pandas as pd

from fs_common import ALIASES, key, sackmann_ids_bio, synthetic_id

RAW, FILES = sys.argv[1], sys.argv[2:]
fix = {}
for tour, al in ALIASES.items():
    ids, _ = sackmann_ids_bio(RAW, tour)
    for a, b in al.items():
        rid = ids.get(key(b))
        if not rid or rid >= 9e8:
            sys.exit(f"{b}: no real id")
        fix[synthetic_id(a)] = (int(rid), b)
for f in FILES:
    f = os.path.realpath(f)
    pq = f.endswith(".parquet")
    d = pd.read_parquet(f) if pq else pd.read_csv(f, low_memory=False)
    n = 0
    for c in ("winner", "loser"):
        if f"{c}_id" not in d:
            continue
        ids = pd.to_numeric(d[f"{c}_id"], errors="coerce")
        hit = ids.isin(list(fix))
        if hit.any():
            d.loc[hit, f"{c}_name"] = ids[hit].map(lambda i: fix[i][1])
            d.loc[hit, f"{c}_id"] = ids[hit].map(lambda i: fix[i][0]).astype(d[f"{c}_id"].dtype)
            n += int(hit.sum())
    if n:
        d.to_parquet(f, index=False) if pq else d.to_csv(f, index=False)
        print(f"{f}: {n} player rows fixed")
