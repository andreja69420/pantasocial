"""Opening and current (closing, once the match has started) home/away odds of every bookmaker Flashscore lists, for
matches in the fs_results.py page cache (or the daily fixtures feed). One GraphQL request per match and region.

usage: fs_odds.py OUT_DIR TOUR [GEO=GB] [--fixtures DAYS | --done DAYS_BACK]
  --done N: finished singles of today and the N days before (daily results feeds) with their result -> {tour}_fs_done_odds_{geo}.parquet
  without --fixtures: all matches in OUT_DIR/fs_cache_{tour} (results pages), for the history test
  --matches ODDS_PARQUET: history test for the matches of an existing odds file (e.g. GB) in another region
  --fixtures N: every scheduled singles match Flashscore lists, for paper trading: the daily feeds for today .. +N days
    plus the fixtures page of every tournament on the category pages (matches scheduled further ahead show up there)
Writes OUT_DIR/{tour}_fs_odds_{geo}.parquet (or {tour}_fs_fixture_odds_{geo}.parquet): one row per match and bookmaker
with fs_mid, book, open_h, open_a, cur_h, cur_a (+ match info for fixtures). Fixed rate (fs_common.Limiter), THREADS connections.
"""
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fs_common import Limiter, page_feed

OUT, TOUR = sys.argv[1], sys.argv[2]
GEO = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "GB"
FIX = int(sys.argv[sys.argv.index("--fixtures") + 1]) if "--fixtures" in sys.argv else 0
DONE = int(sys.argv[sys.argv.index("--done") + 1]) if "--done" in sys.argv else -1
NOCACHE = "--matches" in sys.argv or bool(os.environ.get("NOCACHE"))  # another region's history: one pass, no page cache (the GB cache alone is 3.5 GB)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Origin": "https://www.flashscore.com", "Referer": "https://www.flashscore.com/", "x-fsign": "SW9D1eZo"}
CATS = {"atp": ("ITF MEN - SINGLES", "CHALLENGER MEN - SINGLES", "ATP - SINGLES"),
        "wta": ("ITF WOMEN - SINGLES", "CHALLENGER WOMEN - SINGLES", "WTA - SINGLES")}[TOUR]
URLCATS = {"atp": ("itf-men-singles", "challenger-men-singles", "atp-singles"),
           "wta": ("itf-women-singles", "challenger-women-singles", "wta-singles")}[TOUR]


L = Limiter(float(os.environ.get("RATE", "150")))
get = lambda url: L.get(url, UA, timeout=float(os.environ.get("TIMEOUT", "10")))  # answers take <1 s; a hung one is retried


def feed_records(text):
    head = None
    for rec in text.split("~"):
        d = dict(x.split("÷", 1) for x in rec.split("¬") if "÷" in x)
        if "ZA" in d:
            head = d
        elif "AA" in d and head is not None:
            yield head, d


matches = {}
if DONE >= 0:
    # finished singles of today and the DONE days before (daily feeds), with opening and closing prices: replays of the
    # paper rule on matches that are no longer listed as fixtures
    FIX = -1  # never read or write the history cache
    with ThreadPoolExecutor(16) as ex:
        texts = list(ex.map(get, [f"https://www.flashscore.com/x/feed/f_2_{-off}_1_en_1" for off in range(0, DONE + 1)]))
    for text in texts:
        for head, d in feed_records(text):
            if head["ZA"].split(":")[0].strip() in CATS and d.get("AB") == "3" and d["AA"] not in matches:
                hs, as_ = int(d.get("AG", 0) or 0), int(d.get("AH", 0) or 0)
                matches[d["AA"]] = {"fs_mid": d["AA"], "home_pid": d.get("JA"), "away_pid": d.get("JB"),
                                    "home": d.get("WU", d.get("AE")), "away": d.get("WV", d.get("AF")),
                                    "start": int(d["AD"]), "tournament": head["ZA"], "path": head.get("ZL", ""),
                                    "round": d.get("ER", ""), "result": "void" if d.get("AC") != "3" else ("home" if hs > as_ else "away")}
elif FIX:
    with ThreadPoolExecutor(16) as ex:
        texts = list(ex.map(get, [f"https://www.flashscore.com/x/feed/f_2_{off}_1_en_1" for off in range(0, FIX + 1)]))
        cats = list(ex.map(get, [f"https://www.flashscore.com/tennis/{c}/" for c in URLCATS]))
        tours = sorted({(c, t) for c, page in zip(URLCATS, cats) for t in re.findall(rf'/tennis/{c}/([a-z0-9\-]+)/', page)})
        texts += [page_feed(p, "fixtures") for p in ex.map(get, [f"https://www.flashscore.com/tennis/{c}/{t}/fixtures/" for c, t in tours])]
    for text in texts:
        for head, d in feed_records(text):
            if d.get("AA") in matches and not matches[d["AA"]]["round"] and d.get("ER"):
                # the daily feed has no round; the tournament's fixtures page (read after it) has
                matches[d["AA"]].update(round=d["ER"], tournament=head["ZA"])
            if head["ZA"].split(":")[0].strip() in CATS and d.get("AB") == "1" and d["AA"] not in matches:
                matches[d["AA"]] = {"fs_mid": d["AA"], "home_pid": d.get("JA"), "away_pid": d.get("JB"),
                                    "home": d.get("WU", d.get("AE")), "away": d.get("WV", d.get("AF")),
                                    "start": int(d["AD"]), "tournament": head["ZA"], "path": head.get("ZL", ""),
                                    "round": d.get("ER", "")}
elif "--matches" in sys.argv:
    # the match list of an existing odds file (another region's history), e.g. once the results page cache is gone
    src = pd.read_parquet(sys.argv[sys.argv.index("--matches") + 1]).drop_duplicates("fs_mid")
    matches = {r.fs_mid: {"fs_mid": r.fs_mid, "home_pid": r.home_pid, "away_pid": r.away_pid, "start": int(r.start),
                          "home_won": bool(r.home_won)} for r in src.itertuples()}
else:
    for f in glob.glob(os.path.join(OUT, f"fs_cache_{TOUR}", "*.html")):
        page = open(f, encoding="utf-8").read()
        m = re.search(r"cjs\.initialFeeds\[['\"]results['\"]\]\s*=\s*\{\s*data:\s*`(.*?)`", page, re.S)
        for head, d in feed_records(m.group(1) if m else ""):
            matches[d["AA"]] = {"fs_mid": d["AA"], "home_pid": d.get("JA"), "away_pid": d.get("JB"), "start": int(d.get("AD", 0)),
                                "home_won": (int(d.get("AG", 0) or 0) > int(d.get("AH", 0) or 0)) if d.get("AG") != d.get("AH")
                                else d.get("AS") == "1"}
    lo = int(pd.Timestamp(os.environ.get("FROM", "2026-06-08")).timestamp())
    matches = {k: v for k, v in matches.items() if v["start"] >= lo}
print(f"{TOUR}: {len(matches)} matches to price ({'finished' if DONE >= 0 else 'fixtures' if FIX else 'history'}), region {GEO}", flush=True)


CACHE = os.path.join(OUT, f"fs_odds_cache_{GEO}")
os.makedirs(CACHE, exist_ok=True)


def odds(mid):
    path = os.path.join(CACHE, f"{mid}.json")
    if not FIX and not NOCACHE and os.path.exists(path):
        s = open(path, encoding="utf-8").read()
    else:
        s = get(f"https://global.ds.lsapp.eu/odds/pq_graphql?_hash=oce&eventId={mid}&projectId=2&geoIpCode={GEO}&geoIpSubdivisionCode={GEO}")
        if s and not FIX and not NOCACHE:
            open(path, "w", encoding="utf-8").write(s)
    try:
        o = json.loads(s)["data"]["findOddsByEventId"]
    except Exception:
        return []
    names = {b["bookmaker"]["id"]: b["bookmaker"]["name"] for b in o["settings"]["bookmakers"]}
    info, rows = matches[mid], []
    for e in o.get("odds") or []:
        if e["bettingType"] != "HOME_AWAY" or e["bettingScope"] != "FULL_TIME":
            continue
        side = {x["eventParticipantId"]: x for x in e["odds"]}
        h, a = side.get(info["home_pid"]), side.get(info["away_pid"])
        if not (h and a):
            continue
        f = lambda v: float(v) if v not in (None, "", "-") else float("nan")
        rows.append({**info, "book": names.get(e["bookmakerId"], str(e["bookmakerId"])), "open_h": f(h.get("opening")),
                     "open_a": f(a.get("opening")), "cur_h": f(h.get("value")), "cur_a": f(a.get("value"))})
    return rows


t0 = time.time()
with ThreadPoolExecutor(int(os.environ.get("THREADS", "48"))) as ex:
    res = list(ex.map(odds, list(matches)))
O = pd.DataFrame([r for rs in res for r in rs])
name = f"{TOUR}_fs_done_odds_{GEO}" if DONE >= 0 else f"{TOUR}_fs_fixture_odds_{GEO}" if FIX else f"{TOUR}_fs_odds_{GEO}"
O.to_parquet(os.path.join(OUT, f"{name}.parquet"))
print(f"{len(O)} bookmaker rows for {O.fs_mid.nunique() if len(O) else 0} matches in {time.time() - t0:.0f}s (rate reached {L.rate:.0f}/s); "
      f"books: {O.book.value_counts().head(12).to_dict() if len(O) else {}}", flush=True)
