"""Up-to-date singles results from Flashscore in Sackmann CSV format, to extend the Sackmann archive (ends June 2026).
Each tournament's results page embeds its full results feed (cjs.initialFeeds['results']); one request per tournament.
Tournament slugs: Flashscore's category page plus the BetExplorer slug lists already cached (same slugs, same company).

usage: fs_results.py RAW_DIR OUT_DIR TOUR FROM_DATE [TO_DATE]
  TOUR atp -> itf-men-singles, challenger-men-singles, atp-singles; wta -> itf-women-singles, wta-singles
Writes OUT_DIR/{tour}__{tour}_matches_fs_{year}.csv; page cache in OUT_DIR/fs_cache_{tour}.
Serve stats, rankings, seeds, entry and minutes stay empty (the results-only model does not use them).
Adaptive rate: starts at RATE (16/s), +8/s every 200 clean requests; an error or a >5 s answer backs off to the last
clean rate and pauses 90 s, repeated trouble steps down. THREADS (128) parallel connections.
"""
import glob
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

RAW, OUT, TOUR, FROM = sys.argv[1:5]
TO = sys.argv[5] if len(sys.argv) > 5 else "2099-12-31"
CATS = {"atp": ["itf-men-singles", "challenger-men-singles", "atp-singles"], "wta": ["itf-women-singles", "wta-singles"]}[TOUR]
BX_SLUG_DIRS = {"itf-men-singles": ["bxo_mitf", "bxo25_mitf"], "challenger-men-singles": ["bxo_chal", "bxo25_chal"],
                "itf-women-singles": ["bxo_witf", "bxo25_witf"]}
ROUNDS = {"final": "F", "semi-finals": "SF", "quarter-finals": "QF", "1/8-finals": "R16", "1/16-finals": "R32",
          "1/32-finals": "R64", "1/64-finals": "R128", "round robin": "RR", "group": "RR"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
CACHE = os.path.join(OUT, f"fs_cache_{TOUR}")
os.makedirs(CACHE, exist_ok=True)
THREADS = int(os.environ.get("THREADS", "128"))
stats = {"ok": 0, "err": 0, "slow": 0}


class Limiter:
    """+STEP/s after every UP clean requests; on an error back to the last clean rate, pause, retry; after STRIKES
    errors there step down. A failed rate is a ceiling until REPROBE clean requests."""
    STEP, UP, STRIKES, PAUSE, REPROBE = 8.0, 200, 3, 90, 5000

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
                if self.rate + self.STEP < min(self.ceiling, float(os.environ.get("MAX_RATE", "1000"))):
                    self.rate += self.STEP
                    print(f"  rate up -> {self.rate:.0f}/s", flush=True)




L = Limiter(float(os.environ.get("RATE", "16")))


def fetch(url, cache=True):
    path = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9]+", "_", url.split(".com/")[1])[:180] + ".html")
    # cached pages older than MAX_AGE_H hours are refetched, so tournaments in progress pick up new results
    if cache and os.path.exists(path) and time.time() - os.path.getmtime(path) < 3600 * float(os.environ.get("MAX_AGE_H", "6")):
        return open(path, encoding="utf-8").read()
    for attempt in range(5):
        L.wait()
        t = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                body = r.read().decode("utf-8", "ignore")
            stats["ok"] += 1
            if time.time() - t > 5:  # slow answers are the first sign of throttling
                stats["slow"] += 1
                L.error()
            else:
                L.ok()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                L.ok()
                body = ""
                break
            stats["err"] += 1
            L.error()
        except Exception:
            stats["err"] += 1
            L.error()
    else:
        return ""
    if cache:
        open(path, "w", encoding="utf-8").write(body)
    return body


def key(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(re.sub(r"[^a-z]+", " ", s).split()))


def records(page):
    m = re.search(r"cjs\.initialFeeds\[['\"]results['\"]\]\s*=\s*\{\s*data:\s*`(.*?)`", page, re.S)
    if not m:
        return None, []
    head, rows = None, []
    for rec in m.group(1).split("~"):
        d = dict(x.split("÷", 1) for x in rec.split("¬") if "÷" in x)
        if "ZA" in d:
            head = d
        elif "AA" in d and head is not None:
            rows.append((head, d))
    return head, rows


def level(cat, tname):
    m = re.match(r"[WM](\d+)", tname)
    if cat.startswith("itf"):
        return m.group(1) if m else "15"
    return {"challenger-men-singles": "C", "atp-singles": "A", "wta-singles": "P"}[cat]


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
print(f"pages fetched in {time.time() - t0:.0f}s; ok {stats['ok']} errors {stats['err']} slow(>5s) {stats['slow']}", flush=True)

ids, bio = {}, []
for f in sorted(glob.glob(f"{RAW}/sackmann_raw/{TOUR}__{TOUR}_matches_*.csv")):
    if "_fs_" in f or "_bx_" in f:
        continue
    d = pd.read_csv(f, low_memory=False)
    for c in ("winner", "loser"):
        ids.update(zip(d[f"{c}_name"].map(key), d[f"{c}_id"]))
        bio.append(pd.DataFrame({"id": d[f"{c}_id"], "date": d.tourney_date, "hand": d[f"{c}_hand"], "ht": d[f"{c}_ht"],
                                 "ioc": d[f"{c}_ioc"], "age": d[f"{c}_age"]}))
# static player data from each player's latest Sackmann match; age is carried forward by the elapsed time
BIO = pd.concat(bio).dropna(subset=["id"]).sort_values("date").groupby("id").last()
out, unmapped, skipped = [], 0, {"status": 0, "date": 0}
for (cat, slug), page in zip(slugs, pages):
    _, rows = records(page)
    for head, d in rows:
        day = pd.Timestamp(int(d["AD"]), unit="s").strftime("%Y-%m-%d")
        if not (FROM <= day <= TO):
            skipped["date"] += 1
            continue
        if d.get("AB") != "3" or d.get("AC") not in ("3", "8"):  # 3 finished, 8 retired; walkovers / cancelled skipped
            skipped["status"] += 1
            continue
        hs, as_ = int(d.get("AG", 0) or 0), int(d.get("AH", 0) or 0)
        ret = d.get("AC") == "8"
        hw = hs > as_ if hs != as_ else d.get("AS") == "1"
        sets = []
        for a, b in (("BA", "BB"), ("BC", "BD"), ("BE", "BF"), ("BG", "BH"), ("BI", "BJ")):
            if a in d and b in d and (d[a] or d[b]):
                x, y = (d[a], d[b]) if hw else (d[b], d[a])
                sets.append(f"{x}-{y}")
        if not sets:
            skipped["status"] += 1
            continue
        hname = re.sub(r"(\s\d+)+$", "", d.get("WU", d.get("AE", "")).replace("-", " "))  # slugs carry "-2000" style suffixes
        aname = re.sub(r"(\s\d+)+$", "", d.get("WV", d.get("AF", "")).replace("-", " "))
        w, l = (hname, aname) if hw else (aname, hname)
        wid, lid = ids.get(key(w)), ids.get(key(l))
        unmapped += (wid is None) + (lid is None)
        syn = lambda n: 900000000 + zlib.crc32(key(n).encode()) % 90000000
        za = head["ZA"].split(":", 1)[-1].strip()
        tname = re.sub(r"\s*\(.*$", "", za).strip()
        surf = za.rsplit(",", 1)[-1].strip().capitalize() if "," in za else "Hard"
        surf = {"Clay": "Clay", "Grass": "Grass", "Carpet": "Carpet"}.get(surf.split()[0] if surf else "", "Hard")
        out.append({"tourney_id": f"fs-{slug}-{day[:4]}", "tourney_name": tname, "surface": surf, "draw_size": "",
                    "tourney_level": level(cat, tname), "tourney_date": "", "match_num": 0,
                    "winner_id": wid or syn(w), "winner_name": w.title(), "loser_id": lid or syn(l), "loser_name": l.title(),
                    "score": " ".join(sets) + (" RET" if ret else ""), "best_of": 5 if hs + as_ > 3 and not ret else 3,
                    "round": ROUNDS.get(d.get("ER", "").strip().lower(), "Q1" if "qualif" in d.get("ER", "").lower() else "R32"),
                    "date": day, "fs_mid": d["AA"]})
O = pd.DataFrame(out).drop_duplicates("fs_mid")
for c in ("winner", "loser"):
    b = BIO.reindex(O[f"{c}_id"].values)
    O[f"{c}_hand"], O[f"{c}_ht"], O[f"{c}_ioc"] = b.hand.values, b.ht.values, b.ioc.values
    last = pd.to_datetime(pd.Series(b.date.values).astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    elapsed = (pd.to_datetime(O.date).values - last.values) / pd.Timedelta(days=365.25)
    O[f"{c}_age"] = b.age.values + elapsed
O["tourney_date"] = pd.to_datetime(O.groupby("tourney_id").date.transform("min")).dt.strftime("%Y%m%d")
O = O.sort_values(["tourney_date", "tourney_id", "date"])
O["match_num"] = O.groupby("tourney_id").cumcount()
for y, g in O.groupby(O.date.str[:4]):
    g.drop(columns=["date"]).to_csv(os.path.join(OUT, f"{TOUR}__{TOUR}_matches_fs_{y}.csv"), index=False)
print(f"{len(O)} matches {O.date.min()}..{O.date.max()} written; by level {O.tourney_level.value_counts().head(8).to_dict()}; "
      f"names not in Sackmann {unmapped} of {2 * len(O)}; skipped {skipped}", flush=True)
