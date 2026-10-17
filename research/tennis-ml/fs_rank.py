"""ATP / WTA singles rankings from Flashscore (feed ra_<ranking id>, the whole list with points), saved as weekly
snapshots so the rank and ranking points of every player are known when a match is added. The Flashscore result rows carry
no rank (NaN in 100% of the rows since 2026-06-08 when Sackmann stopped), and Flashscore only shows the current list,
so the history is built from today on, one file per list date.

usage: fs_rank.py DATA_DIR                      fetch both lists, save new snapshots
  DATA_DIR/fs_raw/rank/{tour}_{YYYYMMDD}.parquet   id (Sackmann id / synthetic), name, rank, points, pid (Flashscore id)
  import: ranks_asof(D, tour, ids, dates) -> (rank, points) arrays for players ids at match dates
"""
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

from fs_common import UA, Limiter, key, sackmann_ids_bio, synthetic_id

PAGES = {"atp": "https://www.flashscore.com/tennis/rankings/atp/", "wta": "https://www.flashscore.com/tennis/rankings/wta/"}


def fetch(L, tour):
    """(list date, DataFrame rank / points / name / pid) of the current singles ranking."""
    rid = re.search(r'rankingId = "(\w+)"', L.get(PAGES[tour], UA)).group(1)
    text = L.get(f"https://www.flashscore.com/x/feed/ra_{rid}_1_en_1", UA)
    recs = [dict(x.split("÷", 1) for x in r.split("¬") if "÷" in x) for r in text.split("~")]
    stamp = next(int(r["RRC"]) for r in recs if "RRC" in r)
    rows = [(int(r["RRR"].rstrip(".")), int(r["RRL"]), r["RRP"], r["RPI"]) for r in recs if "RRR" in r and r["RRR"].rstrip(".").isdigit()]
    df = pd.DataFrame(rows, columns=["rank", "points", "name", "pid"])
    # the list is stamped at local midnight of the Monday it is valid from
    return (pd.Timestamp(stamp, unit="s") + pd.Timedelta(hours=12)).normalize(), df


def snapshots(D, tour):
    return sorted(glob.glob(os.path.join(D, "fs_raw", "rank", f"{tour}_*.parquet")))


def run(D, L=None):
    L = L or Limiter(20)
    out = os.path.join(D, "fs_raw", "rank")
    os.makedirs(out, exist_ok=True)
    for tour in ("atp", "wta"):
        date, df = fetch(L, tour)
        f = os.path.join(out, f"{tour}_{date:%Y%m%d}.parquet")
        if os.path.exists(f):
            print(f"{tour}: list of {date:%Y-%m-%d} already saved ({len(df)} players)", flush=True)
            continue
        ids, _ = sackmann_ids_bio(os.path.join(D, "live_raw"), tour)
        df["id"] = [ids.get(key(n)) or synthetic_id(n) for n in df.name]
        df = df.sort_values("rank", kind="stable").drop_duplicates("id")  # two Flashscore players under one name: the better rank
        df.to_parquet(f)
        print(f"{tour}: saved the list of {date:%Y-%m-%d}: {len(df)} players, {int((df.id < 900000000).sum())} with a history id", flush=True)


_CACHE = {}


def ranks_asof(D, tour, pids, days, max_before=10):
    """Rank and points of players (ids) on match days: the latest list dated on or before the day; a day up to max_before days
    before the first list uses that first list; earlier days have NaN. pids / days: arrays of equal length."""
    files = snapshots(D, tour)
    pids = np.asarray(pids, dtype=np.int64)
    rank = np.full(len(pids), np.nan)
    pts = np.full(len(pids), np.nan)
    if not files:
        return rank, pts
    days = pd.to_datetime(pd.Series(days)).dt.normalize().values
    dates = [pd.Timestamp(re.search(r"_(\d{8})\.parquet", f).group(1)) for f in files]
    for i, (f, d) in enumerate(zip(files, dates)):
        if f not in _CACHE:
            t = pd.read_parquet(f).set_index("id")
            _CACHE[f] = t
        lo = d if i else d - pd.Timedelta(days=max_before)
        hi = dates[i + 1] if i + 1 < len(dates) else pd.Timestamp.max
        m = (days >= np.datetime64(lo)) & (days < np.datetime64(hi))
        if m.any():
            t = _CACHE[f]
            rank[m] = t["rank"].reindex(pids[m]).values
            pts[m] = t["points"].reindex(pids[m]).values
    return rank, pts


if __name__ == "__main__":
    t0 = time.time()
    run(sys.argv[1])
    print(f"done in {time.time() - t0:.0f}s")
