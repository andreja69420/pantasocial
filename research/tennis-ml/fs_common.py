"""Shared Flashscore helpers: feed parsing, player-name keys, Sackmann-format rows from feed records."""
import glob
import os
import pickle
import re
import unicodedata
import zlib

import numpy as np
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


class NameIds(dict):
    """name key -> id. A Flashscore name without an exact match falls back to
    1. the player whose name contains all its tokens or whose tokens it all contains (middle names, second surnames:
       'Knutson Gabriela' = 'Gabriela Andrea Knutson', 'Alcaraz Garfia Carlos' = 'Carlos Alcaraz'),
    2. a close spelling of the same name in any token order (Kousuke/Kosuke, Moeller/Moller, 'D Agostino' =
       'Dagostino', 'Wang Xinyu' = 'Xin Yu Wang'): whole-name similarity >= 0.92, every token of the shorter name close
       (>= 0.75) to a token of the other, and clearly the best candidate.
    Fallback matches must have played since 2023 (a Flashscore name in 2026 is an active player).
    When several players qualify, the most recently active one is taken only if the others were last seen 2+ years
    earlier; otherwise the name stays unmapped (a new id), which is safer than merging two players."""

    def __init__(self):
        super().__init__()
        self.last, self.memo = {}, {}

    def add(self, keys, idv, dates):
        """the same as setting self[k] = i row by row (the last id of a key wins) and keeping each id's latest date, vectorised"""
        f = pd.DataFrame({"k": np.asarray(keys, dtype=object), "i": np.asarray(idv),
                          "d": pd.to_numeric(pd.Series(np.asarray(dates)), errors="coerce")})
        last = f.drop_duplicates("k", keep="last")
        self.update(zip(last.k, last.i))
        for i, d in f.groupby("i", dropna=False, sort=False).d.max().fillna(0).items():
            self.last[i] = max(self.last.get(i, 0), d)

    def _index(self):
        self._tok, self._names = {}, {}
        for kk, i in self.items():
            self._names.setdefault(i, set()).add(kk)
            for t in kk.split():
                self._tok.setdefault(t, set()).add(i)
                self._tok.setdefault(t[:4], set()).add(i)

    def _pick(self, cand):
        ranked = sorted(cand, key=lambda i: -self.last.get(i, 0))
        if len(ranked) == 1 or self.last.get(ranked[0], 0) - self.last.get(ranked[1], 0) > 20000:
            return ranked[0]
        return None

    def get(self, k, default=None):
        if k in self:
            return self[k]
        if k in self.memo:
            return self.memo[k] if self.memo[k] is not None else default
        import difflib
        from itertools import permutations
        if not hasattr(self, "_tok"):
            self._index()
        toks, res = set(k.split()), None
        if len(toks) >= 2:
            near = set().union(*[self._tok.get(t, set()) | self._tok.get(t[:4], set()) for t in toks])
            near = {i for i in near if self.last.get(i, 0) >= 20230101}
            sub = {i for i in near for n in self._names[i] if min(len(toks), len(n.split())) >= 2
                   and (toks <= set(n.split()) or set(n.split()) <= toks)}
            if sub:
                res = self._pick(sub)
            else:
                flat = "".join(sorted(toks))
                score = {}
                for i in near:
                    for n in self._names[i]:
                        nt = n.split()
                        if len(nt) > 5:
                            continue
                        short, other = (sorted(toks), nt) if len(toks) <= len(nt) else (nt, sorted(toks))
                        if not all(max(difflib.SequenceMatcher(None, a, b).ratio() for b in other) >= 0.75 for a in short):
                            continue
                        best = max(difflib.SequenceMatcher(None, "".join(p), x).ratio()
                                   for p in permutations(nt) for x in ("".join(k.split()), flat))
                        score[i] = max(score.get(i, 0), best)
                top = sorted(score.items(), key=lambda x: -x[1])
                if top and top[0][1] >= 0.92:
                    close = {i for i, v in top if v >= top[0][1] - 0.04}
                    res = self._pick(close)
        self.memo[k] = res
        return res if res is not None else default


def sackmann_ids_bio(raw, tour, extra=()):
    """name key -> player id, and each player's latest static data (hand, height, country, age at that date).
    Without extra frames the result is cached next to the files (rebuilt when any file's name, size or time changes)."""
    ids, bio = NameIds(), []
    files = [f for f in sorted(glob.glob(f"{raw}/sackmann_raw/{tour}__{tour}_matches_*.csv")) if "_fs_" not in f and "_bx_" not in f]
    cache = os.path.join(raw, f".ids_bio_{tour}.pkl")
    sig = [(os.path.basename(f), os.stat(f).st_size, os.stat(f).st_mtime_ns) for f in files]
    if not extra and os.path.exists(cache):
        with open(cache, "rb") as fh:
            c = pickle.load(fh)
        if c["sig"] == sig:
            ids.update(c["ids"])
            ids.last.update(c["last"])
            return ids, c["bio"]
    for d in [pd.read_csv(f, low_memory=False) for f in files] + list(extra):
        for c in ("winner", "loser"):
            ids.add(d[f"{c}_name"].map({n: key(n) for n in d[f"{c}_name"].unique()}), d[f"{c}_id"],
                    pd.to_numeric(d.tourney_date, errors="coerce").values)
            bio.append(pd.DataFrame({"id": d[f"{c}_id"], "date": pd.to_numeric(d.tourney_date, errors="coerce"),
                                     "hand": d[f"{c}_hand"], "ht": d[f"{c}_ht"], "ioc": d[f"{c}_ioc"], "age": d[f"{c}_age"]}))
    bio = pd.concat(bio).dropna(subset=["id", "date"]).sort_values("date").groupby("id").last()
    if not extra:
        with open(cache + ".tmp", "wb") as fh:
            pickle.dump({"sig": sig, "ids": dict(ids), "last": ids.last, "bio": bio}, fh)
        os.replace(cache + ".tmp", cache)
    return ids, bio


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
            "round": round_code(d), "date": day, "fs_mid": d["AA"],
            "unmapped": (ids.get(key(w)) is None) + (ids.get(key(l)) is None)}


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
        """Body of url ('' for 400/404, an endless redirect or after 5 failed tries). 408/5xx and a timeout = one slow
        query: retry, no pause."""
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
                # 400/404, and a redirect that never ends (a moved tournament page that points to itself): a page that
                # is not there, not a server in trouble, so no pause and no retry
                if e.code in (400, 404) or 300 <= e.code < 400:
                    self.ok()
                    return ""
                if e.code in (408, 500, 502, 503, 504):
                    self.slow()
                    continue
                self.error(f"HTTP {e.code}")
            except Exception as e:
                # no answer within the timeout is a slow answer: lower the rate and ask again, no pause
                if isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError):
                    self.slow()
                    continue
                self.error(type(e).__name__)
        return ""
