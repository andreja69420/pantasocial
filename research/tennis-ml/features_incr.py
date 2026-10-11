"""Incremental feature update: add new matches to an existing feature build in seconds instead of rebuilding
the whole history (features2.py, ~5 min per tour).

How: features2.py saves, next to {tour}_f2.parquet, the final rating state ({tour}_state.pkl) and the prepared match
table ({tour}_matches.parquet). For the new rows this script
  1. continues every Elo variant from the saved state (sequential, only over the new matches),
  2. takes the subset of history that the new rows' features can depend on - every earlier match of the players
     involved plus earlier matches at the same venues (venue speed features) - with those rows' stored pre-match
     ratings,
  3. runs the unchanged vectorized feature engine on that subset only, keeps the new rows,
  4. appends them to {tour}_f2.parquet / {tour}_matches.parquet and saves the new rating state.
Every feature of a row depends only on earlier rows of the same player / pair / venue, so the result equals a full
rebuild (verified in README) as long as the new matches are later than the ones already built.

usage: features_incr.py RAW_DIR OUT_DIR TOUR NEW_CSV [NEW_CSV ...]
env DRY_OUT=path: compute the rows (e.g. upcoming fixtures with placeholder results) and write them to that
parquet without changing the base build (DRY_MATCHES=path also writes the prepared rows with their batch number). Rows are split into batches in which no player appears twice and each batch
starts from the saved state, so one fixture's placeholder result never reaches another fixture's features.
"""
import os
import pickle
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW, OUT, TOUR, NEW = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.argv = [sys.argv[0], RAW, OUT, TOUR, "1991"]
import features2 as F2  # noqa: E402

t0 = time.time()
st = pickle.load(open(f"{OUT}/{TOUR}_state.pkl", "rb"))
base = pd.read_parquet(f"{OUT}/{TOUR}_matches.parquet")
DRY = os.environ.get("DRY_OUT")
NAMES = ["elo", "selo", "felo", "gelo", "melo", "welo", "wselo", "wsetelo"]
RCOLS = [f"{s}_{n}" for n in NAMES for s in ("w", "l")]
F2COLS = pq.read_schema(f"{OUT}/{TOUR}_f2.parquet").names
F2COLS = [c for c in F2COLS if not c.startswith("__")]
# a dry run only needs the stored ratings of the history subset: read 16 columns instead of the whole build
f2 = pd.read_parquet(f"{OUT}/{TOUR}_f2.parquet", columns=RCOLS if DRY else None)
new = []
for f in NEW:
    d = pd.read_csv(f, low_memory=False)
    d["main"] = int(bool(re.search(rf"{TOUR}_matches_\d{{4}}", os.path.basename(f))))
    new.append(d)
new = pd.concat(new, ignore_index=True)
key = lambda d: d.tourney_date.astype(int).astype(str) + "|" + d.winner_id.astype(int).astype(str) + "|" + d.loser_id.astype(int).astype(str)
if not DRY:
    new = new[~key(new).isin(set(key(base)))]
if new.empty:
    print("nothing new")
    sys.exit(0)
new = F2.prepare(new, first_order=st["last_order"] + 1)
late = (new.date < base.date.max()).sum()
if late:
    print(f"  note: {late} new rows are dated before the last built match; they are appended after it", flush=True)
first_mi = int(f2.index.max()) + 1
new.index = np.arange(first_mi, first_mi + len(new))



def features_for(new):
    # 1. ratings for the new rows, continued from the saved state
    state = tuple(defaultdict(lambda: 1500.0, d) for d in st["ratings"][:8]) + tuple(defaultdict(int, d) for d in st["ratings"][8:])
    sc = np.array([F2.parse_score(s) for s in new.score.values], dtype=float)
    gsh = np.where(sc[:, 0] + sc[:, 1] > 0, sc[:, 0] / np.maximum(sc[:, 0] + sc[:, 1], 1), np.nan)
    ssh = np.where(sc[:, 2] + sc[:, 3] > 0, sc[:, 2] / np.maximum(sc[:, 2] + sc[:, 3], 1), np.nan)
    R_new = F2.ratings(new.reset_index(drop=True), gsh, ssh, state)
    # 2. history the new rows depend on: same players, same venues
    players = set(new.winner_id) | set(new.loser_id)
    venues = set(new.tourney_name.fillna("").str.lower())
    sub = base[base.winner_id.isin(players) | base.loser_id.isin(players) | base.tourney_name.fillna("").str.lower().isin(venues)]
    R_sub = f2.loc[sub.index, RCOLS].values
    # 3. unchanged feature engine on subset + new rows
    S = pd.concat([sub, new])
    gmi = S.index.values
    S = S.reset_index(drop=True)
    _, W, Lo = F2.build(m=S, R=np.vstack([R_sub, R_new]), wcmap=st["wcmap"])
    keep = np.flatnonzero(np.isin(gmi, new.index))
    W, Lo = W.loc[keep], Lo.loc[keep]
    W.index = Lo.index = gmi[keep]
    ctx = new[["date", "surface", "tourney_level", "rord", "best_of", "draw_size", "main", "retired", "score", "tourney_name"]].copy()
    for c in ("odds_w", "odds_l", "agree"):
        ctx[c] = np.nan if c != "agree" else False
    return W.add_prefix("w_").join(Lo.add_prefix("l_")).join(ctx), len(sub)


if DRY:
    batch, seen = np.zeros(len(new), dtype=int), []
    for i, (w, l) in enumerate(zip(new.winner_id.values, new.loser_id.values)):
        b = 0
        while b < len(seen) and (w in seen[b] or l in seen[b]):
            b += 1
        if b == len(seen):
            seen.append(set())
        seen[b] |= {w, l}
        batch[i] = b
    rows = pd.concat([features_for(new[batch == b])[0] for b in range(len(seen))]).loc[new.index]
    print(f"  {len(seen)} batch(es) without a repeated player", flush=True)
else:
    rows, n_sub = features_for(new)

if DRY:
    out = rows[F2COLS].copy()
    for c in ("winner_id", "loser_id", "winner_name", "loser_name", "fs_mid"):
        if c in new:
            out[c] = new[c].values
    out.to_parquet(DRY)
    if os.environ.get("DRY_MATCHES"):  # the prepared rows themselves (matches-table format), for fx_live.py
        nm = new.drop(columns=[c for c in new.columns if new[c].dtype == object and c not in F2.KEEP_OBJ | {"fs_mid"}])
        for c in (F2.KEEP_OBJ | {"fs_mid"}) & set(nm.columns):
            nm[c] = nm[c].astype(str).replace("nan", np.nan)
        nm.assign(batch=batch).to_parquet(os.environ["DRY_MATCHES"])
    print(f"{TOUR}: features for {len(out)} rows (dry run) in {time.time() - t0:.1f}s -> {DRY}", flush=True)
    sys.exit(0)

# 4. append and save the new state
f2 = pd.concat([f2, rows[F2COLS]])
f2.to_parquet(f"{OUT}/{TOUR}_f2.parquet")
nt = new.drop(columns=[c for c in new.columns if new[c].dtype == object and c not in F2.KEEP_OBJ])
for c in F2.KEEP_OBJ & set(nt.columns):
    nt[c] = nt[c].astype(str).replace("nan", np.nan)
pd.concat([base, nt[[c for c in base.columns if c in nt.columns]]]).to_parquet(f"{OUT}/{TOUR}_matches.parquet")
with open(f"{OUT}/{TOUR}_state.pkl", "wb") as f:
    pickle.dump({"ratings": [dict(d) for d in F2.ratings.state], "wcmap": st["wcmap"], "last_order": int(new.order.max())}, f)
print(f"{TOUR}: +{len(new)} matches (history subset {n_sub} of {len(base)} rows) in {time.time() - t0:.1f}s; "
      f"features now {len(f2)} rows up to {f2.date.max():%Y-%m-%d}", flush=True)
