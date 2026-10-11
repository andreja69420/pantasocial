"""Daily results update in seconds: finished singles matches of the last DAYS days from Flashscore's daily feeds, the ones
not yet in the feature build are converted to Sackmann rows and appended with features_incr.py (no full rebuild).

usage: fs_update.py DATA_DIR [DAYS=7]
  DATA_DIR/live_raw (Sackmann files for ids), DATA_DIR/out_live (feature build), DATA_DIR/fs_raw/updates (written CSVs)
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fs_common import CATS, Limiter, add_bio, records, result_row, sackmann_ids_bio

D = sys.argv[1]
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
HERE = os.path.dirname(os.path.abspath(__file__))
UPD = os.path.join(D, "fs_raw", "updates")
os.makedirs(UPD, exist_ok=True)
t0 = time.time()
L = Limiter(float(os.environ.get("RATE", "150")))
with ThreadPoolExecutor(8) as ex:
    feeds = list(ex.map(lambda off: L.get(f"https://www.flashscore.com/x/feed/f_2_{off}_1_en_1"), range(-DAYS, 1)))
pairs = [(h, d) for text in feeds for h, d in records(text)]
for tour in ("wta", "atp"):
    base = pd.read_parquet(os.path.join(D, "out_live", f"{tour}_matches.parquet"),
                           columns=["tourney_id", "tourney_date", "winner_id", "loser_id", "fs_mid"])
    have = set(base.fs_mid.dropna().astype(str))
    ids, BIO = sackmann_ids_bio(os.path.join(D, "live_raw"), tour)
    rows = [r for h, d in pairs if CATS.get(h["ZA"].split(":")[0].strip(), ("",))[0] == tour and d["AA"] not in have
            and (r := result_row(h, d, ids))]
    if not rows:
        print(f"{tour}: no new finished matches", flush=True)
        continue
    O = add_bio(pd.DataFrame(rows), BIO)
    O.pop("unmapped")
    # tournament start date: from the build when the tournament is already in it, else its first new match day
    start = base.assign(tid=base.tourney_id.astype(str)).groupby("tid").tourney_date.min().astype(int)
    first = pd.to_datetime(O.groupby("tourney_id").date.transform("min")).dt.strftime("%Y%m%d").astype(int)
    O["tourney_date"] = O.tourney_id.map(start).fillna(first).astype(int)
    O = O.sort_values(["date", "tourney_id"])
    O["match_num"] = 900 + O.groupby("tourney_id").cumcount()
    f = os.path.join(UPD, f"{tour}__{tour}_matches_fs_upd_{time.strftime('%Y%m%d%H%M')}.csv")
    O.drop(columns=["date"]).to_csv(f, index=False)
    r = subprocess.run([sys.executable, os.path.join(HERE, "features_incr.py"), os.path.join(D, "live_raw"), os.path.join(D, "out_live"),
                        tour, f], capture_output=True, text=True)
    print(f"{tour}: {len(O)} new finished matches {O.date.min()}..{O.date.max()} -> {r.stdout.strip().splitlines()[-1:] or r.stderr[-500:]}",
          flush=True)
print(f"update done in {time.time() - t0:.0f}s", flush=True)
