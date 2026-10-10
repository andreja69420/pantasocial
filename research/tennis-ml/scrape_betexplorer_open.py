"""Scrape one BetExplorer category-season with per-bookmaker opening and closing odds.

Step 1: results pages of every tournament (match id, players slug, date, result, average odds).
Step 2: per match, the bookmaker table (closing odds + archive ids).
Step 3: per bookmaker side, the odds history; the oldest entry is the opening price.
Everything is cached on disk; rate limited like scrape_betexplorer_itf.py (RATE env, default 8/s).

usage: scrape_betexplorer_open.py OUT_DIR CATEGORY SEASON   (e.g. itf-women-singles 2026; current season = base slug)
"""
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT, CAT, SEASON = sys.argv[1], sys.argv[2], int(sys.argv[3])
BASE = f"https://www.betexplorer.com/tennis/{CAT}/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "X-Requested-With": "XMLHttpRequest", "Referer": "https://www.betexplorer.com/"}
CURRENT = 2026
for sub in ("res", "match", "arch"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)


class Limiter:
    def __init__(self, rate):
        self.rate, self.max_rate, self.lock, self.next_t, self.ok_streak, self.pause_until = rate, rate, threading.Lock(), 0.0, 0, 0.0

    def wait(self):
        with self.lock:
            t = max(time.time(), self.next_t, self.pause_until)
            self.next_t = t + 1.0 / self.rate
        time.sleep(max(0.0, t - time.time()))

    def error(self):
        with self.lock:
            self.rate = max(0.5, self.rate / 2)
            self.pause_until = time.time() + 30
            self.ok_streak = 0
            print(f"  ! error -> rate {self.rate:.1f}/s, pausing 30s", flush=True)

    def ok(self):
        with self.lock:
            self.ok_streak += 1
            if self.ok_streak >= 50 and self.rate < self.max_rate:
                self.rate, self.ok_streak = min(self.max_rate, self.rate * 1.5), 0


L = Limiter(float(os.environ.get("RATE", "8")))


def fetch(url, tries=5):
    for _ in range(tries):
        L.wait()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                body, final = r.read().decode("utf-8", "ignore"), r.geturl()
            L.ok()
            return body, final
        except urllib.error.HTTPError as e:
            if e.code == 404:
                L.ok()
                return None, url
            L.error()
        except Exception:
            L.error()
    return None, url


def cached(path, url, keep=lambda b, f: b):
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    body, final = fetch(url)
    body = (keep(body, final) if body else "") or ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return body


def run(jobs, fn, label):
    t0, done = time.time(), [0]

    def w(j):
        fn(j)
        done[0] += 1
        if done[0] % 500 == 0:
            print(f"  {label}: {done[0]}/{len(jobs)} {time.time() - t0:.0f}s rate {L.rate:.1f}/s", flush=True)
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(w, jobs))


listing, _ = fetch(BASE)
slugs = sorted(set(re.findall(rf'href="/tennis/{CAT}/([a-z0-9\-]+)/"', listing or "")))
slugs = [s for s in slugs if not re.search(r"-\d{4}$", s)]
print(f"{CAT} {SEASON}: {len(slugs)} tournament slugs", flush=True)


def res_job(slug):
    url = f"{BASE}{slug}/results/" if SEASON == CURRENT else f"{BASE}{slug}-{SEASON}/results/"
    cached(os.path.join(OUT, "res", f"{slug}.html"), url, lambda b, f: b if "/results/" in f and "data-odd" in b else "")


run(slugs, res_job, "results pages")
matches = {}
for slug in slugs:
    h = open(os.path.join(OUT, "res", f"{slug}.html"), encoding="utf-8").read()
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S):
        link = re.search(rf'href="/tennis/{CAT}/[a-z0-9\-]+/([a-z0-9\-]+)/([A-Za-z0-9]+)/"', tr)
        if link and "data-odd" in tr:
            matches[link.group(2)] = link.group(1)
print(f"matches with odds: {len(matches)}", flush=True)
run(list(matches), lambda mid: cached(os.path.join(OUT, "match", f"{mid}.json"),
                                      f"https://www.betexplorer.com/match-odds-old/{mid}/1/ha/1/en/"), "bookmaker tables")
arch = []
for mid in matches:
    s = open(os.path.join(OUT, "match", f"{mid}.json"), encoding="utf-8").read()
    if not s:
        continue
    h = json.loads(s).get("odds", "")
    for td in re.findall(r"<td[^>]*js-has-aodds[^>]*>", h):
        g = lambda k: (re.search(rf'{k}="([^"]*)"', td) or [None, None])[1]
        if g("data-oid"):
            arch.append((g("data-oid"), g("data-bid"), g("data-bt"), g("data-sc"), g("data-hcp")))
print(f"odds histories to fetch: {len(arch)}", flush=True)
run(arch, lambda a: cached(os.path.join(OUT, "arch", f"{a[0]}.json"),
                           f"https://www.betexplorer.com/archive-odds/{a[0]}/{a[1]}/{a[2]}/{a[3]}/{a[4]}/"), "odds histories")
print("done", flush=True)
