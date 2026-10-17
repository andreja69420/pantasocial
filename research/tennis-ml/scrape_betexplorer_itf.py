"""Scrape men's ITF results pages (with average odds) from BetExplorer for the given seasons.
Rate: up to RATE (env, default 8) requests/second; on errors (429/403/5xx/timeouts) halve the rate and pause, then recover slowly.
Pages are cached on disk so the run can be resumed.

usage: scrape_betexplorer_itf.py OUT_DIR SEASON [SEASON ...]   (current season = base slug, older = slug-YYYY)
"""
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT, SEASONS = sys.argv[1], [int(s) for s in sys.argv[2:]]
BASE = "https://www.betexplorer.com/tennis/itf-men-singles/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
CURRENT = 2026
os.makedirs(OUT, exist_ok=True)


class Limiter:
    def __init__(self, rate=8.0):
        self.rate, self.max_rate, self.lock, self.next_t, self.ok_streak = rate, rate, threading.Lock(), 0.0, 0
        self.pause_until = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next_t, self.pause_until)
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
                self.rate = min(self.max_rate, self.rate * 1.5)
                self.ok_streak = 0


L = Limiter(float(os.environ.get("RATE", "8")))


def fetch(url, tries=5):
    for _ in range(tries):
        L.wait()
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
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


listing, _ = fetch(BASE)
slugs = sorted(set(re.findall(r'href="/tennis/itf-men-singles/([a-z0-9\-]+)/"', listing or "")))
slugs = [s for s in slugs if not re.search(r"-\d{4}$", s)]
print(f"{len(slugs)} tournament slugs; seasons {SEASONS}", flush=True)
jobs = [(s, y) for y in SEASONS for s in slugs]
done = [0]
t0 = time.time()


def work(job):
    slug, year = job
    path = os.path.join(OUT, f"{slug}__{year}.html")
    if os.path.exists(path):
        return
    url = f"{BASE}{slug}/results/" if year == CURRENT else f"{BASE}{slug}-{year}/results/"
    body, final = fetch(url)
    # missing seasons redirect to the home page; store an empty marker so a resume skips them
    keep = body if body and "/results/" in final and "data-odd" in body else ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(keep)
    done[0] += 1
    if done[0] % 250 == 0:
        print(f"  {done[0]}/{len(jobs)} pages, {time.time() - t0:.0f}s, rate {L.rate:.1f}/s", flush=True)


with ThreadPoolExecutor(32) as ex:
    list(ex.map(work, jobs))
n_ok = sum(1 for f in os.listdir(OUT) if os.path.getsize(os.path.join(OUT, f)) > 0)
print(f"done: {n_ok} pages with odds in {time.time() - t0:.0f}s", flush=True)
