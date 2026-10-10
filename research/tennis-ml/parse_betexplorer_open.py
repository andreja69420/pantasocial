"""Parse a scrape_betexplorer_open.py directory: per match, BetMGM.us opening price (oldest entry of its odds history;
the current price when the line never moved) and closing price, oriented winner/loser by the result, and attach to
Sackmann matches by sorted name tokens + date window. Writes {tour}_bxo_odds.parquet indexed like {tour}_f2.parquet.

usage: parse_betexplorer_open.py RAW_DIR OUT_DIR TOUR BXO_DIR [BOOK_ID=851]
"""
import json
import os
import re
import sys
import unicodedata

import pandas as pd

RAW, OUT, TOUR, BX = sys.argv[1:5]
BOOK = sys.argv[5] if len(sys.argv) > 5 else "851"
sys.argv = [sys.argv[0], RAW, OUT, TOUR, "1991"]
import features2  # noqa: E402


def toks(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]+", " ", s).split()


def attr(td, k):
    m = re.search(rf'{k}="([^"]*)"', td)
    return m.group(1) if m else None


def opening(td, match_date):
    """(opening odds, opening time, number of price changes) for one bookmaker cell."""
    close = float(attr(td, "data-odd"))
    oid = attr(td, "data-oid") if "js-has-aodds" in td else None
    path = os.path.join(BX, "arch", f"{oid}.json") if oid else None
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        c = attr(td, "data-created") or "1,1,1970,0,0"
        dd, mo, yy, hh, mi = [int(x) for x in c.split(",")]
        t = pd.Timestamp(yy, mo, dd, hh, mi) if c else pd.NaT
        return close, t, 0
    hist = []
    for e in json.loads(open(path, encoding="utf-8").read()):
        m = re.match(r"(\d+)\.(\d+)\.\s*(\d+):(\d+)", e.get("date", ""))
        if not m:
            continue
        d, mo, hh, mm = map(int, m.groups())
        y = match_date.year - (1 if mo > match_date.month + 1 else 0)
        hist.append((pd.Timestamp(y, mo, d, hh, mm), float(e["odd"])))
    if not hist:
        return close, pd.NaT, 0
    hist.sort()
    return hist[0][1], hist[0][0], len(hist) - 1


rows = []
for fn in os.listdir(os.path.join(BX, "res")):
    h = open(os.path.join(BX, "res", fn), encoding="utf-8", errors="ignore").read()
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S):
        link = re.search(r'href="/tennis/[a-z\-]+/[a-z0-9\-]+/([a-z0-9\-]+)/([A-Za-z0-9]+)/"', tr)
        res = re.search(r">\s*(\d):(\d)\s*<", tr)
        date = re.search(r">\s*(\d\d)\.(\d\d)\.(\d{4})?\s*<", tr)
        if not (link and res and date) or res.group(1) == res.group(2):
            continue
        d, mth, yr = date.groups()
        md = pd.Timestamp(int(yr or 2026), int(mth), int(d))
        path = os.path.join(BX, "match", f"{link.group(2)}.json")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        tds = [t for t in re.findall(r"<td[^>]*>", json.loads(open(path, encoding="utf-8").read()).get("odds", ""))
               if "data-odd" in t and attr(t, "data-bookmaker-id") == BOOK]
        if len(tds) != 2:
            continue
        (oh, th, nh), (oa, ta, na) = opening(tds[0], md), opening(tds[1], md)
        ch, ca = float(attr(tds[0], "data-odd")), float(attr(tds[1], "data-odd"))
        hw = int(res.group(1)) > int(res.group(2))
        rows.append({"slug_toks": " ".join(sorted(link.group(1).replace("-", " ").split())), "date": md,
                     "open_w": oh if hw else oa, "open_l": oa if hw else oh,
                     "close_w": ch if hw else ca, "close_l": ca if hw else ch,
                     "open_time": min(th, ta), "moves": nh + na, "retired": "ret" in tr.lower()})
B = pd.DataFrame(rows).drop_duplicates(["slug_toks", "date"])
print(f"parsed matches with book {BOOK}: {len(B)} ({B.date.min():%Y-%m-%d}..{B.date.max():%Y-%m-%d}); "
      f"with line movement {(B.moves > 0).mean():.0%}", flush=True)

m = features2.load_matches()
m = m[m.date >= B.date.min() - pd.Timedelta(days=14)]
key = (m.winner_name.map(toks) + m.loser_name.map(toks)).map(lambda t: " ".join(sorted(t)))
S = pd.DataFrame({"mi": m.index, "date": m.date, "slug_toks": key.values})
j = S.merge(B, on="slug_toks", suffixes=("", "_bx"))
j = j[((j.date_bx - j.date).dt.days >= -2) & ((j.date_bx - j.date).dt.days <= 14)]
j = j.drop_duplicates("mi").set_index("mi")
print(f"matched to Sackmann: {len(j)}; margin open {(1 / j.open_w + 1 / j.open_l - 1).median():.2%}, "
      f"close {(1 / j.close_w + 1 / j.close_l - 1).median():.2%}", flush=True)
j[["open_w", "open_l", "close_w", "close_l", "open_time", "date_bx", "moves", "retired"]].to_parquet(f"{OUT}/{TOUR}_bxo_odds.parquet")
