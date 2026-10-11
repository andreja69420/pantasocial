"""Shared Flashscore helpers: feed parsing, player-name keys, Sackmann-format rows from feed records."""
import glob
import re
import unicodedata
import zlib

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "x-fsign": "SW9D1eZo"}
# Flashscore category header -> (tour, url category); WTA 125 events are level "C" in the Sackmann WTA files
CATS = {"ITF MEN - SINGLES": ("atp", "itf-men-singles"), "CHALLENGER MEN - SINGLES": ("atp", "challenger-men-singles"),
        "ATP - SINGLES": ("atp", "atp-singles"), "ITF WOMEN - SINGLES": ("wta", "itf-women-singles"),
        "CHALLENGER WOMEN - SINGLES": ("wta", "challenger-women-singles"), "WTA - SINGLES": ("wta", "wta-singles")}
ROUNDS = {"final": "F", "semi-finals": "SF", "quarter-finals": "QF", "1/8-finals": "R16", "1/16-finals": "R32",
          "1/32-finals": "R64", "1/64-finals": "R128", "round robin": "RR", "group": "RR"}


def key(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(re.sub(r"[^a-z]+", " ", s).split()))


def records(text):
    """(header, match) pairs of a Flashscore feed: ~ separates records, ¬ fields, ÷ key and value."""
    head = None
    for rec in text.split("~"):
        d = dict(x.split("÷", 1) for x in rec.split("¬") if "÷" in x)
        if "ZA" in d:
            head = d
        elif "AA" in d and head is not None:
            yield head, d


def page_feed(page, name="results"):
    m = re.search(rf"cjs\.initialFeeds\[['\"]{name}['\"]\]\s*=\s*\{{\s*data:\s*`(.*?)`", page, re.S)
    return m.group(1) if m else ""


def level(cat, tname):
    m = re.match(r"[WM](\d+)", tname)
    if cat.startswith("itf"):
        return m.group(1) if m else "15"
    return {"challenger-men-singles": "C", "challenger-women-singles": "C", "atp-singles": "A", "wta-singles": "P"}[cat]


def tournament(head):
    """(category, slug, name, surface) from a feed header, e.g. 'CHALLENGER MEN - SINGLES: Maia (Portugal) - Qualification,
    clay (indoor)' with ZL '/tennis/challenger-men-singles/maia/'."""
    za = head["ZA"].split(":", 1)[-1].strip()
    cat = CATS.get(head["ZA"].split(":")[0].strip(), (None, None))[1]
    zl = head.get("ZL", "").strip("/").split("/")
    tname = re.sub(r"\s*\(.*$", "", za).strip()
    slug = zl[-1] if len(zl) >= 3 else re.sub(r"[^a-z0-9]+", "-", tname.lower()).strip("-")
    surf = za.rsplit(",", 1)[-1].strip().lower() if "," in za else "hard"
    surf = {"clay": "Clay", "grass": "Grass", "carpet": "Carpet"}.get(surf.split()[0] if surf else "", "Hard")
    return cat, slug, tname, surf


def player_name(d, side):
    raw = d.get("WU" if side == "h" else "WV", d.get("AE" if side == "h" else "AF", ""))
    return re.sub(r"(\s\d+)+$", "", raw.replace("-", " "))  # slugs carry "-2000" style suffixes


def synthetic_id(name):
    return 900000000 + zlib.crc32(key(name).encode()) % 90000000


def round_code(d):
    er = d.get("ER", "").strip().lower()
    return ROUNDS.get(er, "Q1" if "qualif" in er else "R32")


def sackmann_ids_bio(raw, tour, extra=()):
    """name key -> player id, and each player's latest static data (hand, height, country, age at that date)."""
    ids, bio = {}, []
    files = [f for f in sorted(glob.glob(f"{raw}/sackmann_raw/{tour}__{tour}_matches_*.csv")) if "_fs_" not in f and "_bx_" not in f]
    for d in [pd.read_csv(f, low_memory=False) for f in files] + list(extra):
        for c in ("winner", "loser"):
            ids.update(zip(d[f"{c}_name"].map(key), d[f"{c}_id"]))
            bio.append(pd.DataFrame({"id": d[f"{c}_id"], "date": pd.to_numeric(d.tourney_date, errors="coerce"),
                                     "hand": d[f"{c}_hand"], "ht": d[f"{c}_ht"], "ioc": d[f"{c}_ioc"], "age": d[f"{c}_age"]}))
    return ids, pd.concat(bio).dropna(subset=["id", "date"]).sort_values("date").groupby("id").last()


def add_bio(O, BIO, day_col="date"):
    for c in ("winner", "loser"):
        b = BIO.reindex(O[f"{c}_id"].values)
        O[f"{c}_hand"], O[f"{c}_ht"], O[f"{c}_ioc"] = b.hand.values, b.ht.values, b.ioc.values
        last = pd.to_datetime(pd.Series(b.date.values).astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        elapsed = (pd.to_datetime(O[day_col]).values - last.values) / pd.Timedelta(days=365.25)
        O[f"{c}_age"] = b.age.values + elapsed
    return O


def result_row(head, d, ids):
    """Sackmann-format row of a finished (or retired) match, None for walkovers / cancelled / unfinished."""
    if d.get("AB") != "3" or d.get("AC") not in ("3", "8"):
        return None
    cat, slug, tname, surf = tournament(head)
    if cat is None:
        return None
    hs, as_ = int(d.get("AG", 0) or 0), int(d.get("AH", 0) or 0)
    ret = d.get("AC") == "8"
    hw = hs > as_ if hs != as_ else d.get("AS") == "1"
    sets = []
    for a, b in (("BA", "BB"), ("BC", "BD"), ("BE", "BF"), ("BG", "BH"), ("BI", "BJ")):
        if a in d and b in d and (d[a] or d[b]):
            x, y = (d[a], d[b]) if hw else (d[b], d[a])
            sets.append(f"{x}-{y}")
    if not sets:
        return None
    hname, aname = player_name(d, "h"), player_name(d, "a")
    w, l = (hname, aname) if hw else (aname, hname)
    day = pd.Timestamp(int(d["AD"]), unit="s").strftime("%Y-%m-%d")
    return {"tourney_id": f"fs-{slug}-{day[:4]}", "tourney_name": tname, "surface": surf, "draw_size": "",
            "tourney_level": level(cat, tname), "tourney_date": "", "match_num": 0,
            "winner_id": ids.get(key(w)) or synthetic_id(w), "winner_name": w.title(),
            "loser_id": ids.get(key(l)) or synthetic_id(l), "loser_name": l.title(),
            "score": " ".join(sets) + (" RET" if ret else ""), "best_of": 5 if hs + as_ > 3 and not ret else 3,
            "round": round_code(d), "date": day, "fs_mid": d["AA"], "unmapped": (key(w) not in ids) + (key(l) not in ids)}


class Limiter:
    """Fixed target rate (Flashscore works at ~150/s). A slow answer (>5 s) only lowers the rate by STEP, no pause, and it
    climbs back after UP clean requests; only a real error (HTTP 429/5xx, connection failure) pauses."""
    STEP, UP, PAUSE = 10.0, 200, 30

    def __init__(self, rate):
        import threading
        self.rate = self.target = rate
        self.lock, self.next_t, self.ok_n, self.pause_until, self.last_cut = threading.Lock(), 0.0, 0, 0.0, 0.0

    def wait(self):
        import time
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
        import time
        # many requests are in flight at once, so one slow spell reports many times: cut at most once per 3 s
        with self.lock:
            if time.time() - self.last_cut > 3:
                self.rate, self.ok_n, self.last_cut = max(10.0, self.rate - self.STEP), 0, time.time()

    def error(self, why=""):
        import time
        with self.lock:
            if time.time() < self.pause_until:
                return
            self.rate, self.ok_n, self.last_cut = max(10.0, self.rate - self.STEP), 0, time.time()
            self.pause_until = time.time() + self.PAUSE
            print(f"  ! error ({why}) -> {self.rate:.0f}/s, pause {self.PAUSE}s", flush=True)

    def get(self, url, headers=None, timeout=30):
        """Body of url ('' for 400/404 or after 5 failed tries). 408/5xx = backend gave up on one slow query: retry, no pause."""
        import time
        import urllib.error
        import urllib.request
        for _ in range(5):
            self.wait()
            t = time.time()
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers or UA), timeout=timeout) as r:
                    body = r.read().decode("utf-8", "ignore")
                (self.slow if time.time() - t > 5 else self.ok)()
                return body
            except urllib.error.HTTPError as e:
                if e.code in (400, 404):
                    self.ok()
                    return ""
                if e.code in (408, 500, 502, 503, 504):
                    self.slow()
                    continue
                self.error(f"HTTP {e.code}")
            except Exception as e:
                self.error(type(e).__name__)
        return ""
