"""Match statistics from Flashscore for every finished singles match of the given seasons: the box score (aces, double
faults, 1st serve in/won, 2nd serve won, service games, break points; Challenger, WTA 125 and tour level) and, where
there is no box score, the point-by-point log (most ITF matches from ~2024-25 on, W35+ more often than W15), from which
service points played/won, service games, holds and break points faced/saved are counted exactly (tiebreak points too).

usage: fs_stats.py DATA_DIR YEARS            e.g. fs_stats.py $B 2023,2024,2025,2026
  COUNTRY_ONLY=1: only read the season pages and add the venues' countries to event_country.json (fs_country.py)
  slugs: Flashscore category pages + BetExplorer slug caches (DATA_DIR/bxo*) + Flashscore tourney ids in the build;
  season pages {slug}-{year}/results/ (the current season is the bare slug). Writes DATA_DIR/fs_raw/stats/{tour}_fs_stats.parquet
  with winner/loser ids (Sackmann ids via NameIds), Sackmann stat columns w_/l_ (box score) and w_pbp_/l_pbp_ (point by point).
Fixed rate RATE (150/s, one limiter for everything), THREADS (128).
"""
import glob
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from fs_common import UA, Limiter, canonical, event_id, key, level, page_feed, player_name, records, sackmann_ids_bio, synthetic_id, tournament
import fs_country

D, YEARS = sys.argv[1], [int(y) for y in sys.argv[2].split(",")]
OUT = os.path.join(D, "fs_raw", "stats")
os.makedirs(OUT, exist_ok=True)
L = Limiter(float(os.environ.get("RATE", "150")))
THREADS = int(os.environ.get("THREADS", "128"))
TOURS = {"atp": ["itf-men-singles", "challenger-men-singles", "atp-singles"],
         "wta": ["itf-women-singles", "challenger-women-singles", "wta-singles"]}
BX = {"itf-men-singles": ["bxo_mitf", "bxo25_mitf"], "challenger-men-singles": ["bxo_chal", "bxo25_chal"],
      "itf-women-singles": ["bxo_witf", "bxo25_witf"]}
from fs_statlib import BOX, PT, box, fields, match_stats, pbp  # noqa: E402,F401


def season_pages(tour):
    slugs = set()
    for cat in TOURS[tour]:
        page = L.get(f"https://www.flashscore.com/tennis/{cat}/", UA)
        s = set(re.findall(rf'/tennis/{cat}/([a-z0-9\-]+)/', page))
        for dd in BX.get(cat, []):
            s |= {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(D, dd, "res", "*.html"))}
        s = {re.sub(r"-\d{4}$", "", x) for x in s} - {"results", "fixtures", "standings", "archive", "draw"}
        slugs |= {(cat, x) for x in s}
    M = pd.read_parquet(os.path.join(D, "out_live", f"{tour}_matches.parquet"), columns=["tourney_id", "tourney_level"])
    for tid, lv in M[M.tourney_id.astype(str).str.startswith("fs-")].drop_duplicates("tourney_id").itertuples(index=False):
        slug = re.sub(r"^fs-(ch-)?|-\d{4}$", "", tid)
        cat = TOURS[tour][0] if lv[:1].isdigit() else (TOURS[tour][1] if lv == "C" else TOURS[tour][2])
        slugs.add((cat, slug))
    cur = max(YEARS)
    return [f"https://www.flashscore.com/tennis/{cat}/{slug}{'' if y == cur else f'-{y}'}/results/" for cat, slug in sorted(slugs) for y in YEARS]


t0 = time.time()
for tour in ("wta", "atp"):
    urls = season_pages(tour)
    with ThreadPoolExecutor(THREADS) as ex:
        feeds = list(ex.map(lambda u: page_feed(L.get(u, UA)), urls))
    ms = {}
    heads = []
    for f in feeds:
        for head, d in records(f):
            heads.append((head, d))
            if d.get("AB") != "3" or d.get("AC") not in ("3", "8") or tournament(head)[0] is None:
                continue
            hs, as_ = int(d.get("AG", 0) or 0), int(d.get("AH", 0) or 0)
            cat, slug, tname, surf = tournament(head)
            ms[d["AA"]] = {"fs_mid": d["AA"], "date": pd.Timestamp(int(d["AD"]), unit="s").normalize(), "cat": cat, "tourney_name": tname,
                           "level": level(cat, tname), "home": player_name(d, "h"), "away": player_name(d, "a"),
                           "home_won": hs > as_ if hs != as_ else d.get("AS") == "1"}
    n_new = fs_country.update(heads, lambda h: (lambda t: event_id(t[0], t[1], "0000") if t[0] else None)(tournament(h)))
    print(f"{tour}: event countries, {n_new} new venues ({len(fs_country.load())} in the map)", flush=True)
    if os.environ.get("COUNTRY_ONLY"):
        continue
    M = pd.DataFrame(ms.values())
    M = M[M.date.dt.year.isin(YEARS)].reset_index(drop=True)
    print(f"{tour}: {len(urls)} season pages, {len(M)} finished matches {M.date.min():%Y-%m-%d}..{M.date.max():%Y-%m-%d} "
          f"({time.time() - t0:.0f}s, rate {L.rate:.0f}/s)", flush=True)

    with ThreadPoolExecutor(THREADS) as ex:
        S = list(ex.map(lambda mid: match_stats(L, mid), M.fs_mid))
    ids, _ = sackmann_ids_bio(os.path.join(D, "live_raw"), tour)
    rows = []
    for m, (b, p) in zip(M.to_dict("records"), S):
        w = 0 if m["home_won"] else 1
        names = (m["home"], m["away"])
        r = {k: m[k] for k in ("fs_mid", "date", "cat", "tourney_name", "level")}
        for side, pre in ((w, "winner"), (1 - w, "loser")):
            r[f"{pre}_name"] = canonical(names[side])
            r[f"{pre}_id"] = ids.get(key(names[side])) or synthetic_id(names[side])
            s = pre[0]
            for st in ("ace", "df", "1stIn", "1stWon", "2ndIn", "2ndWon", "SvGms", "SvGmsWon", "bpSaved", "bpFaced"):
                r[f"{s}_{st}"] = b.get((side, st), np.nan)
            for j, st in enumerate(("svpt", "svwon", "SvGms", "holds", "bpFaced", "bpSaved", "tbpt", "tbwon")):
                r[f"{s}_pbp_{st}"] = p[side, j] if p is not None else np.nan
        for s in ("w", "l"):
            r[f"{s}_svpt"] = r[f"{s}_1stIn"] + r[f"{s}_2ndIn"]
        rows.append(r)
    O = pd.DataFrame(rows)
    O.to_parquet(os.path.join(OUT, f"{tour}_fs_stats.parquet"))
    lv = O.level.where(O.level.isin(["C", "A", "P"]), "ITF")
    cov = pd.DataFrame({"box": O.w_SvGms.notna(), "pbp": O.w_pbp_SvGms.notna(), "pbp_points": O.w_pbp_svpt > 0})
    print(f"{tour}: {len(O)} matches with stats lookup in {time.time() - t0:.0f}s; coverage by year and level:\n"
          + cov.groupby([O.date.dt.year, lv]).mean().round(2).assign(n=O.groupby([O.date.dt.year, lv]).size()).to_string(), flush=True)
