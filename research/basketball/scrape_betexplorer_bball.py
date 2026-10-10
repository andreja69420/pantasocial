"""Scrape BetExplorer basketball results (all seasons) and, from ODDS_FROM on, every bookmaker's opening and closing
home/away odds (match table + odds history per bookmaker side). Cached on disk; adaptive request rate.

usage: scrape_betexplorer_bball.py OUT_DIR LEAGUE FIRST LAST [ODDS_FROM]
  LEAGUE e.g. usa/nba or europe/euroleague; FIRST/LAST = first year of the season (2008 -> 2008-2009)
Writes OUT_DIR/games.csv (season, stage, date, home, away, home_pts, away_pts, mid, avg_home, avg_away).
"""
import csv
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT, LEAGUE, FIRST, LAST = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ODDS_FROM = int(sys.argv[5]) if len(sys.argv) > 5 else 2020
SKIP_STAGES = ("pre-season", "all stars", "all-star")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "X-Requested-With": "XMLHttpRequest", "Referer": "https://www.betexplorer.com/"}
for sub in ("res", "match", "arch"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)


class Limiter:
    """+STEP/s after every UP clean requests; on an error back to the last clean rate, pause, retry; after STRIKES
    errors there step down. A failed rate is a ceiling until REPROBE clean requests."""
    STEP, UP, STRIKES, PAUSE, REPROBE = 4.0, 300, 3, 90, 5000

    def __init__(self, rate):
        self.rate = self.good = rate
        self.ceiling, self.lock, self.next_t, self.ok_streak, self.pause_until, self.strikes = 1e9, threading.Lock(), 0.0, 0, 0.0, 0

    def wait(self):
        with self.lock:
            t = max(time.time(), self.next_t, self.pause_until)
            self.next_t = t + 1.0 / self.rate
        time.sleep(max(0.0, t - time.time()))

    def error(self):
        with self.lock:
            if time.time() < self.pause_until:
                return
            if self.rate > self.good:
                self.ceiling, self.rate, self.strikes = self.rate, self.good, 1
            else:
                self.strikes += 1
                if self.strikes > self.STRIKES:
                    self.ceiling, self.rate = self.rate, max(1.0, self.rate - self.STEP)
                    self.good, self.strikes = self.rate, 1
            self.ok_streak = 0
            self.pause_until = time.time() + self.PAUSE
            print(f"  ! error -> rate {self.rate:.0f}/s (ceiling {self.ceiling:.0f}), pausing {self.PAUSE}s", flush=True)

    def ok(self):
        with self.lock:
            self.ok_streak += 1
            if self.ok_streak >= self.REPROBE and self.ceiling < 1e9:
                self.ceiling = 1e9
            if self.ok_streak % self.UP == 0:
                self.good, self.strikes = self.rate, 0
                if self.rate + self.STEP < self.ceiling:
                    self.rate += self.STEP


L = Limiter(float(os.environ.get("RATE", "16")))


def fetch(url, tries=5):
    for _ in range(tries):
        L.wait()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                body = r.read().decode("utf-8", "ignore")
            L.ok()
            return body
        except urllib.error.HTTPError as e:
            if e.code == 404:
                L.ok()
                return None
            L.error()
        except Exception:
            L.error()
    return None


def cached(path, url):
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    body = fetch(url) or ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return body


def run(jobs, fn, label):
    t0, done = time.time(), [0]

    def w(j):
        fn(j)
        done[0] += 1
        if done[0] % 1000 == 0:
            print(f"  {label}: {done[0]}/{len(jobs)} {time.time() - t0:.0f}s rate {L.rate:.0f}/s", flush=True)
    with ThreadPoolExecutor(96) as ex:
        list(ex.map(w, jobs))


slug = LEAGUE.split("/")[-1]
games = []
for y in range(FIRST, LAST + 1):
    season = f"{y}-{y + 1}"
    base = f"https://www.betexplorer.com/basketball/{LEAGUE}-{season}/results/"
    h = cached(os.path.join(OUT, "res", f"{season}.html"), base)
    stages = [(a, re.sub(r"<[^>]+>", "", b).strip()) for a, b in re.findall(r'<a[^>]*stage=([A-Za-z0-9]+)[^>]*>(.*?)</a>', h, re.S)]
    stages = [(a, b) for a, b in dict(stages).items() if b and not b.lower().startswith(SKIP_STAGES) and "version" not in b.lower()]
    n = 0
    for sid, sname in stages:
        sh = cached(os.path.join(OUT, "res", f"{season}_{sid}.html"), f"{base}?stage={sid}&month=all")
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", sh, re.S):
            link = re.search(rf'href="/basketball/{LEAGUE}-{season}/[a-z0-9\-]+/([A-Za-z0-9]{{8}})/"', tr)
            teams = re.findall(r"<span>(?:<strong>)?([^<]+)(?:</strong>)?</span>", tr)
            score = re.search(r">\s*(\d+):(\d+)(?:\s*[A-Z.]*)?\s*<", tr)
            date = re.search(r">\s*(\d\d)\.(\d\d)\.(\d{4})\s*<", tr)
            if not (link and len(teams) >= 2 and score and date):
                continue
            odds = re.findall(r'data-odd="([0-9.]+)"', tr)
            games.append({"season": y, "stage": sname, "date": f"{date.group(3)}-{date.group(2)}-{date.group(1)}",
                          "home": teams[0].strip(), "away": teams[1].strip(), "home_pts": score.group(1), "away_pts": score.group(2),
                          "mid": link.group(1), "avg_home": odds[0] if len(odds) >= 2 else "", "avg_away": odds[1] if len(odds) >= 2 else ""})
            n += 1
    print(f"{LEAGUE} {season}: {n} games, stages {[b for a, b in stages]}", flush=True)
with open(os.path.join(OUT, "games.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(games[0]))
    w.writeheader()
    w.writerows(games)
mids = [g["mid"] for g in games if g["season"] >= ODDS_FROM and g["avg_home"]]
print(f"games {len(games)}; bookmaker tables to fetch (seasons >= {ODDS_FROM}): {len(mids)}", flush=True)
run(mids, lambda m: cached(os.path.join(OUT, "match", f"{m}.json"), f"https://www.betexplorer.com/match-odds-old/{m}/1/ha/1/en/"),
    "bookmaker tables")
arch = []
for m in mids:
    s = open(os.path.join(OUT, "match", f"{m}.json"), encoding="utf-8").read()
    for td in re.findall(r"<td[^>]*js-has-aodds[^>]*>", s.replace('\\"', '"')):
        g = lambda k: (re.search(rf'{k}="([^"]*)"', td) or [None, None])[1]
        if g("data-oid"):
            arch.append((g("data-oid"), g("data-bid"), g("data-bt"), g("data-sc"), g("data-hcp")))
print(f"odds histories to fetch: {len(arch)}", flush=True)
run(arch, lambda a: cached(os.path.join(OUT, "arch", f"{a[0]}.json"),
                           f"https://www.betexplorer.com/archive-odds/{a[0]}/{a[1]}/{a[2]}/{a[3]}/{a[4]}/"), "odds histories")
print("done", flush=True)
