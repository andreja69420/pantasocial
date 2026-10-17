"""Fresh tennis results from BetExplorer in Sackmann CSV format, so features2.py can extend the archive (which ends
in June 2026) up to today. One row per finished singles match: tournament, level, round, date, players (mapped to
Sackmann ids by name; unseen players get synthetic ids >= 9e8), score as "6-2 5-7 6-2" (+ " RET").
Serve statistics, rankings, seeds, entry and minutes stay empty - the results-only model does not use them.

usage: bx_live_results.py RAW_DIR OUT_DIR TOUR FROM_DATE [TO_DATE]
  TOUR atp -> challenger-men-singles, itf-men-singles, atp-singles; wta -> itf-women-singles, wta-singles
Writes OUT_DIR/{tour}__{tour}_matches_bx_{year}.csv (cache of pages in OUT_DIR/bx_cache_{tour}).
"""
import csv
import glob
import threading
import time
import os
import re
import sys
import unicodedata
import zlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

RAW, OUT, TOUR, FROM = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
TO = sys.argv[5] if len(sys.argv) > 5 else "2099-12-31"
CURRENT = 2026
CATS = {"atp": ["challenger-men-singles", "itf-men-singles", "atp-singles"], "wta": ["itf-women-singles", "wta-singles"]}[TOUR]
ROUNDS = {"final": "F", "semi-finals": "SF", "quarter-finals": "QF", "1/8-finals": "R16", "1/16-finals": "R32",
          "1/32-finals": "R64", "1/64-finals": "R128", "round robin": "RR"}
CACHE = os.path.join(OUT, f"bx_cache_{TOUR}")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Referer": "https://www.betexplorer.com/"}
os.makedirs(CACHE, exist_ok=True)
os.environ.setdefault("RATE", "32")


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
                if self.rate + self.STEP < min(self.ceiling, float(os.environ.get("MAX_RATE", "64"))):
                    self.rate += self.STEP





def key(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(re.sub(r"[^a-z]+", " ", s).split()))


L = Limiter(float(os.environ["RATE"]))


def fetch(url):
    import urllib.error
    import urllib.request
    path = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9]+", "_", url.split("betexplorer.com/")[1])[:180] + ".html")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    for _ in range(5):
        L.wait()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                body = r.read().decode("utf-8", "ignore")
            L.ok()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                L.ok()
                body = ""
                break
            L.error()
        except Exception:
            L.error()
    else:
        return ""
    open(path, "w", encoding="utf-8").write(body)
    return body


def level(cat, tname):
    m = re.match(r"[WM](\d+)", tname)
    if cat.startswith("itf"):
        return m.group(1) if m else "15"
    return {"challenger-men-singles": "C", "atp-singles": "A", "wta-singles": "P"}[cat]


rows = []
for cat in CATS:
    base = f"https://www.betexplorer.com/tennis/{cat}/"
    slugs = sorted({s for s in re.findall(rf'href="/tennis/{cat}/([a-z0-9\-]+)/"', fetch(base)) if not re.search(r"-\d{{4}}$", s)})
    with ThreadPoolExecutor(64) as ex:
        pages = dict(zip(slugs, ex.map(lambda s: fetch(f"{base}{s}/results/"), slugs)))
    for slug, h in pages.items():
        rnd = "R32"
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S):
            th = re.search(r"<th[^>]*>(.*?)</th>", tr, re.S)
            if th and "in-match" not in tr:
                rnd = ROUNDS.get(re.sub(r"<[^>]+>", "", th.group(1)).strip().lower(), "R32")
                continue
            link = re.search(rf'href="(/tennis/{cat}/{slug}/[a-z0-9\-]+/([A-Za-z0-9]+)/)"', tr)
            date = re.search(r">\s*(\d\d)\.(\d\d)\.(\d{4})?\s*<", tr)
            if not (link and date):
                continue
            d = f"{date.group(3) or CURRENT}-{date.group(2)}-{date.group(1)}"
            if FROM <= d <= TO and "in-match" in tr:
                rows.append({"cat": cat, "slug": slug, "url": link.group(1), "mid": link.group(2), "date": d, "round": rnd,
                             "ret": bool(re.search(r"ret\.|retired", tr, re.I)), "wo": bool(re.search(r"w\.o\.|awarded", tr, re.I))})
    print(f"{cat}: {len(slugs)} tournaments, finished matches {FROM}..{TO}: {sum(r['cat'] == cat for r in rows)}", flush=True)
rows = [r for r in rows if not r["wo"]]
with ThreadPoolExecutor(64) as ex:
    pages = list(ex.map(lambda r: fetch("https://www.betexplorer.com" + r["url"]), rows))

# Sackmann name -> id map (latest id per normalised name)
ids = {}
for f in sorted(glob.glob(f"{RAW}/sackmann_raw/{TOUR}__{TOUR}_matches_*.csv")):
    d = pd.read_csv(f, usecols=["winner_id", "winner_name", "loser_id", "loser_name"], low_memory=False)
    for c in ("winner", "loser"):
        ids.update(zip(d[f"{c}_name"].map(key), d[f"{c}_id"]))
out, unmapped = [], 0
for r, h in zip(rows, pages):
    names = re.findall(r'class="list-details__item__title"[^>]*>(.*?)</', h, re.S)
    sets = re.search(r'id="js-partial"[^>]*>\((.*?)\)<', h)
    score = re.search(r'js-score[^>]*>(\d):(\d)<', h)
    tname = re.findall(r'list-breadcrumb__item__in"[^>]*>(.*?)<', h)
    if len(names) < 2 or not score or not sets:
        continue
    home, away = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", n)).strip() for n in names[:2]]
    hw = int(score.group(1)) > int(score.group(2))
    games = re.findall(r"(\d+):(\d+)(?:<sup>(\d+)</sup>)?", sets.group(1))
    sc = []
    for a, b, tb in games:
        a, b = (a, b) if hw else (b, a)
        sc.append(f"{a}-{b}" + (f"({tb})" if tb else ""))
    w, l = (home, away) if hw else (away, home)
    wid, lid = ids.get(key(w)), ids.get(key(l))
    unmapped += (wid is None) + (lid is None)
    syn = lambda n: 900000000 + zlib.crc32(key(n).encode()) % 90000000
    t = re.sub(r"\s+\d{4}$", "", tname[3]).strip() if len(tname) > 3 else r["slug"]
    out.append({"tourney_id": f"bx-{r['slug']}-{r['date'][:4]}", "tourney_name": t, "surface": "", "draw_size": "",
                "tourney_level": level(r["cat"], t), "tourney_date": "", "match_num": 0, "winner_id": wid or syn(w),
                "winner_name": w, "loser_id": lid or syn(l), "loser_name": l, "score": " ".join(sc) + (" RET" if r["ret"] else ""),
                "best_of": 3, "round": r["round"], "date": r["date"], "bx_mid": r["mid"]})
O = pd.DataFrame(out)
# Sackmann dates matches by the tournament's start: use the first match day of each tournament
O["tourney_date"] = pd.to_datetime(O.groupby("tourney_id").date.transform("min")).dt.strftime("%Y%m%d")
# surface: the most recent Sackmann edition of a tournament with the same name, else Hard
S = pd.concat(pd.read_csv(f, usecols=["tourney_name", "surface", "tourney_date"], low_memory=False)
              for f in sorted(glob.glob(f"{RAW}/sackmann_raw/{TOUR}__{TOUR}_matches_*202[45].csv")))
surf = S.sort_values("tourney_date").groupby(S.tourney_name.map(key)).surface.last()
O["surface"] = O.tourney_name.map(key).map(surf).fillna("Hard")
O["match_num"] = O.groupby("tourney_id").cumcount()
for y, g in O.groupby(O.date.str[:4]):
    g.drop(columns=["date"]).to_csv(os.path.join(OUT, f"{TOUR}__{TOUR}_matches_bx_{y}.csv"), index=False, quoting=csv.QUOTE_MINIMAL)
print(f"{len(O)} matches written; player names not in Sackmann: {unmapped} of {2 * len(O)}", flush=True)
