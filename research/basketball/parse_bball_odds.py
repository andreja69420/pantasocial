"""Per game and bookmaker: opening (oldest entry of the odds history, or the current price if it never moved) and
closing home/away odds from a scrape_betexplorer_bball.py directory. Writes LEAGUE_DIR/odds.parquet indexed by mid
with columns {book}_oh, {book}_oa, {book}_ch, {book}_ca for bet365.us (b365), BetMGM.us (mgm), Stake.com (stake).

usage: parse_bball_odds.py LEAGUE_DIR
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

DIR = sys.argv[1]
BOOKS = {"549": "b365", "851": "mgm", "997": "stake"}
attr = lambda td, k: (re.search(rf'{k}="([^"]*)"', td) or [None, None])[1]


def opening(td, d):
    oid = attr(td, "data-oid") if "js-has-aodds" in td else None
    path = os.path.join(DIR, "arch", f"{oid}.json") if oid else None
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return float(attr(td, "data-odd"))
    hist = []
    for e in json.loads(open(path, encoding="utf-8").read()):
        m = re.match(r"(\d+)\.(\d+)\.\s*(\d+):(\d+)", e.get("date", ""))
        if m:
            dd, mo, hh, mi = map(int, m.groups())
            y = d.year - (1 if mo > d.month + 1 else 0)
            hist.append((pd.Timestamp(y, mo, dd, hh, mi), float(e["odd"])))
    return min(hist)[1] if hist else float(attr(td, "data-odd"))


G = pd.read_csv(f"{DIR}/games.csv", dtype={"mid": str}, parse_dates=["date"]).drop_duplicates("mid")
rows = []
for g in G.itertuples():
    path = os.path.join(DIR, "match", f"{g.mid}.json")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        continue
    h = json.loads(open(path, encoding="utf-8").read()).get("odds", "")
    tds = [t for t in re.findall(r"<td[^>]*>", h) if "data-odd" in t]
    r = {"mid": g.mid}
    for bid, name in BOOKS.items():
        b = [t for t in tds if attr(t, "data-bookmaker-id") == bid]
        if len(b) == 2:
            r[f"{name}_ch"], r[f"{name}_ca"] = float(attr(b[0], "data-odd")), float(attr(b[1], "data-odd"))
            r[f"{name}_oh"], r[f"{name}_oa"] = opening(b[0], g.date), opening(b[1], g.date)
    rows.append(r)
O = pd.DataFrame(rows).set_index("mid")
print(f"{DIR}: {len(O)} games with a bookmaker table; " + ", ".join(
    f"{n} {O[f'{n}_oh'].notna().sum()} (margin open {np.nanmedian(1 / O[f'{n}_oh'] + 1 / O[f'{n}_oa'] - 1):.1%})"
    for n in BOOKS.values() if f"{n}_oh" in O))
O.to_parquet(f"{DIR}/odds.parquet")
