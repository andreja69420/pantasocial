"""Up-to-date singles results from Flashscore in Sackmann CSV format, to extend the Sackmann archive (ends June 2026).
Each tournament's results page embeds its full results feed (cjs.initialFeeds['results']); one request per tournament.
Tournament slugs: Flashscore's category page plus the BetExplorer slug lists already cached (same slugs, same company).

usage: fs_results.py RAW_DIR OUT_DIR TOUR FROM_DATE [TO_DATE]
  TOUR atp -> itf-men-singles, challenger-men-singles, atp-singles; wta -> itf-women-, challenger-women- (WTA 125), wta-singles
Writes OUT_DIR/{tour}__{tour}_matches_fs_{year}.csv; page cache in OUT_DIR/fs_cache_{tour}.
Serve stats, rankings, seeds, entry and minutes stay empty (the results-only model does not use them).
Fixed rate RATE (150/s): a slow answer only lowers it a little, a real error pauses (fs_common.Limiter). THREADS (128).
"""
import glob
import os
import re
import sys
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fs_common import Limiter, add_bio, records, page_feed, result_row, sackmann_ids_bio

RAW, OUT, TOUR, FROM = sys.argv[1:5]
TO = sys.argv[5] if len(sys.argv) > 5 else "2099-12-31"
CATS = {"atp": ["itf-men-singles", "challenger-men-singles", "atp-singles"],
        "wta": ["itf-women-singles", "challenger-women-singles", "wta-singles"]}[TOUR]
BX_SLUG_DIRS = {"itf-men-singles": ["bxo_mitf", "bxo25_mitf"], "challenger-men-singles": ["bxo_chal", "bxo25_chal"],
                "itf-women-singles": ["bxo_witf", "bxo25_witf"]}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
CACHE = os.path.join(OUT, f"fs_cache_{TOUR}")
os.makedirs(CACHE, exist_ok=True)
THREADS = int(os.environ.get("THREADS", "128"))
L = Limiter(float(os.environ.get("RATE", "150")))


def fetch(url, cache=True):
    path = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9]+", "_", url.split(".com/")[1])[:180] + ".html")
    # cached pages older than MAX_AGE_H hours are refetched, so tournaments in progress pick up new results
    if cache and os.path.exists(path) and time.time() - os.path.getmtime(path) < 3600 * float(os.environ.get("MAX_AGE_H", "6")):
        return open(path, encoding="utf-8").read()
    body = L.get(url, UA)
    if cache and body:
        open(path, "w", encoding="utf-8").write(body)
    return body


slugs = []
for cat in CATS:
    s = set(re.findall(rf'/tennis/{cat}/([a-z0-9\-]+)/', fetch(f"https://www.flashscore.com/tennis/{cat}/", cache=False)))
    for d in BX_SLUG_DIRS.get(cat, []):
        s |= {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(RAW, d, "res", "*.html"))}
    if cat in ("atp-singles", "wta-singles"):  # tour level: the full slug list from BetExplorer's category page
        try:
            with urllib.request.urlopen(urllib.request.Request(f"https://www.betexplorer.com/tennis/{cat}/", headers=UA), timeout=90) as r:
                s |= set(re.findall(rf'href="/tennis/{cat}/([a-z0-9\-]+)/"', r.read().decode("utf-8", "ignore")))
        except Exception as e:
            print(f"  BetExplorer slug list for {cat} unavailable ({e}); using Flashscore's current list only", flush=True)
    if cat in ("atp-singles", "wta-singles"):  # plus slugified names of recent Sackmann tour-level events
        T = pd.concat(pd.read_csv(f, usecols=["tourney_name", "tourney_level"], low_memory=False)
                      for f in glob.glob(f"{RAW}/sackmann_raw/{TOUR}__{TOUR}_matches_202[456].csv"))
        T = T[~T.tourney_level.astype(str).isin(["D", "O"])]
        s |= {re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()).strip("-")
              for n in T.tourney_name.dropna().unique()}
    s = {x for x in s if not re.search(r"-\d{4}$", x) and x not in ("results", "fixtures", "standings", "archive", "draw")}
    slugs += [(cat, x) for x in sorted(s)]
    print(f"{cat}: {len(s)} tournament slugs", flush=True)
t0 = time.time()
with ThreadPoolExecutor(THREADS) as ex:
    pages = list(ex.map(lambda cs: fetch(f"https://www.flashscore.com/tennis/{cs[0]}/{cs[1]}/results/"), slugs))
print(f"pages fetched in {time.time() - t0:.0f}s (rate {L.rate:.0f}/s)", flush=True)

ids, BIO = sackmann_ids_bio(RAW, TOUR)
out = [r for page in pages for head, d in records(page_feed(page)) if (r := result_row(head, d, ids))]
O = pd.DataFrame(out)
O = O[(O.date >= FROM) & (O.date <= TO)].drop_duplicates("fs_mid")
unmapped = int(O.pop("unmapped").sum())
O = add_bio(O, BIO)
O["tourney_date"] = pd.to_datetime(O.groupby("tourney_id").date.transform("min")).dt.strftime("%Y%m%d")
O = O.sort_values(["tourney_date", "tourney_id", "date"])
O["match_num"] = O.groupby("tourney_id").cumcount()
for y, g in O.groupby(O.date.str[:4]):
    g.drop(columns=["date"]).to_csv(os.path.join(OUT, f"{TOUR}__{TOUR}_matches_fs_{y}.csv"), index=False)
print(f"{len(O)} matches {O.date.min()}..{O.date.max()} written; by level {O.tourney_level.value_counts().head(8).to_dict()}; "
      f"names not in Sackmann {unmapped} of {2 * len(O)}", flush=True)
