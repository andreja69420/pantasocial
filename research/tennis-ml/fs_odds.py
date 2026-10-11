"""Opening and current (closing, once the match has started) home/away odds of every bookmaker Flashscore lists, for
matches in the fs_results.py page cache (or the daily fixtures feed). One GraphQL request per match and region.

usage: fs_odds.py OUT_DIR TOUR [GEO=GB] [--fixtures DAYS]
  without --fixtures: all matches in OUT_DIR/fs_cache_{tour} (results pages), for the history test
  --fixtures N: the next N days of singles fixtures from the daily feed (for paper trading)
Writes OUT_DIR/{tour}_fs_odds_{geo}.parquet (or {tour}_fs_fixture_odds_{geo}.parquet): one row per match and bookmaker
with fs_mid, book, open_h, open_a, cur_h, cur_a (+ match info for fixtures). Adaptive rate, 128 connections.
"""
import glob
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

OUT, TOUR = sys.argv[1], sys.argv[2]
GEO = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "GB"
FIX = int(sys.argv[sys.argv.index("--fixtures") + 1]) if "--fixtures" in sys.argv else 0
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Origin": "https://www.flashscore.com", "Referer": "https://www.flashscore.com/", "x-fsign": "SW9D1eZo"}
CATS = {"atp": ("ITF MEN - SINGLES", "CHALLENGER MEN - SINGLES", "ATP - SINGLES"),
        "wta": ("ITF WOMEN - SINGLES", "CHALLENGER WOMEN - SINGLES", "WTA - SINGLES")}[TOUR]


class Limiter:
    """Fixed target rate (RATE, default 150/s). A slow answer (>5 s) only lowers the rate by STEP (no pause) and it
    climbs back after UP clean requests; only a real error (HTTP 429/5xx, connection failure) pauses."""
    STEP, UP, PAUSE = 10.0, 200, 30

    def __init__(self, rate):
        self.rate = self.target = rate
        self.lock, self.next_t, self.ok_n, self.pause_until, self.last_cut = threading.Lock(), 0.0, 0, 0.0, 0.0

    def wait(self):
        with self.lock:
            t = max(time.time(), self.next_t, self.pause_until)
            self.next_t = t + 1.0 / self.rate
        time.sleep(max(0.0, t - time.time()))

    def ok(self):
        with self.lock:
            self.ok_n += 1
            if self.ok_n % self.UP == 0 and self.rate < self.target:
                self.rate = min(self.target, self.rate + self.STEP)

    def slow(self):
        # many requests are in flight at once, so one slow spell reports many times: cut at most once per 3 s
        with self.lock:
            if time.time() - self.last_cut > 3:
                self.rate, self.ok_n, self.last_cut = max(10.0, self.rate - self.STEP), 0, time.time()

    def error(self):
        with self.lock:
            if time.time() < self.pause_until:
                return
            self.rate, self.ok_n, self.last_cut = max(10.0, self.rate - self.STEP), 0, time.time()
            self.pause_until = time.time() + self.PAUSE
            print(f"  ! error -> {self.rate:.0f}/s, pause {self.PAUSE}s", flush=True)


L = Limiter(float(os.environ.get("RATE", "150")))


def get(url):
    for _ in range(5):
        L.wait()
        t = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                body = r.read().decode("utf-8", "ignore")
            (L.slow if time.time() - t > 5 else L.ok)()
            return body
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                L.ok()
                return ""
            L.error()
        except Exception:
            L.error()
    return ""


def feed_records(text):
    head = None
    for rec in text.split("~"):
        d = dict(x.split("÷", 1) for x in rec.split("¬") if "÷" in x)
        if "ZA" in d:
            head = d
        elif "AA" in d and head is not None:
            yield head, d


matches = {}
if FIX:
    for off in range(0, FIX + 1):
        for head, d in feed_records(get(f"https://www.flashscore.com/x/feed/f_2_{off}_1_en_1")):
            if head["ZA"].split(":")[0].strip() in CATS and d.get("AB") == "1":
                matches[d["AA"]] = {"fs_mid": d["AA"], "home_pid": d.get("JA"), "away_pid": d.get("JB"),
                                    "home": d.get("WU", d.get("AE")), "away": d.get("WV", d.get("AF")),
                                    "start": int(d["AD"]), "tournament": head["ZA"]}
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
print(f"{TOUR}: {len(matches)} matches to price ({'fixtures' if FIX else 'history'}), region {GEO}", flush=True)


CACHE = os.path.join(OUT, f"fs_odds_cache_{GEO}")
os.makedirs(CACHE, exist_ok=True)


def odds(mid):
    path = os.path.join(CACHE, f"{mid}.json")
    if not FIX and os.path.exists(path):
        s = open(path, encoding="utf-8").read()
    else:
        s = get(f"https://global.ds.lsapp.eu/odds/pq_graphql?_hash=oce&eventId={mid}&projectId=2&geoIpCode={GEO}&geoIpSubdivisionCode={GEO}")
        if s and not FIX:
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
with ThreadPoolExecutor(128) as ex:
    res = list(ex.map(odds, list(matches)))
O = pd.DataFrame([r for rs in res for r in rs])
name = f"{TOUR}_fs_fixture_odds_{GEO}" if FIX else f"{TOUR}_fs_odds_{GEO}"
O.to_parquet(os.path.join(OUT, f"{name}.parquet"))
print(f"{len(O)} bookmaker rows for {O.fs_mid.nunique() if len(O) else 0} matches in {time.time() - t0:.0f}s (rate reached {L.rate:.0f}/s); "
      f"books: {O.book.value_counts().head(12).to_dict() if len(O) else {}}", flush=True)
