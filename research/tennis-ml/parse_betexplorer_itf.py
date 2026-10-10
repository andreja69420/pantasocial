"""Parse cached BetExplorer men's ITF results pages and attach the average odds to Sackmann futures/ITF matches.
Writes atp_bx_odds.parquet indexed like atp_f2.parquet.

usage: parse_betexplorer_itf.py RAW_DIR OUT_DIR BX_DIR
"""
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

RAW, OUT, BX = sys.argv[1:4]
sys.argv = [sys.argv[0], RAW, OUT, "atp", "1991"]
import features2  # noqa: E402


def toks(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]+", " ", s).split()


rows = []
for fn in os.listdir(BX):
    path = os.path.join(BX, fn)
    if os.path.getsize(path) == 0:
        continue
    season = int(fn.rsplit("__", 1)[1][:4])
    h = open(path, encoding="utf-8", errors="ignore").read()
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S):
        link = re.search(r'href="/tennis/itf-men-singles/[a-z0-9\-]+/([a-z0-9\-]+)/[A-Za-z0-9]+/"', tr)
        odds = re.findall(r'data-odd="([0-9.]+)"', tr)
        res = re.search(r">\s*(\d):(\d)\s*<", tr)
        date = re.search(r">\s*(\d\d)\.(\d\d)\.(\d{4})?\s*<", tr)
        if not (link and len(odds) >= 2 and res and date):
            continue
        d, mth, yr = date.groups()
        hs, as_ = int(res.group(1)), int(res.group(2))
        if hs == as_:
            continue
        o_home, o_away = float(odds[0]), float(odds[1])
        home_won = hs > as_
        rows.append({"slug_toks": " ".join(sorted(link.group(1).replace("-", " ").split())),
                     "date": pd.Timestamp(int(yr or season), int(mth), int(d)),
                     "odds_w": o_home if home_won else o_away, "odds_l": o_away if home_won else o_home,
                     "retired": "ret" in tr.lower()})
B = pd.DataFrame(rows).drop_duplicates(["slug_toks", "date"])
print("parsed matches with odds:", len(B), B.date.min(), B.date.max(), flush=True)

m = features2.load_matches()
lv = m.tourney_level.astype(str)
m = m[(lv.str.match(r"^\d") | (lv == "S")) & (m.date >= "2023-12-01")]
m_key = (m.winner_name.map(toks) + m.loser_name.map(toks)).map(lambda t: " ".join(sorted(t)))
S = pd.DataFrame({"mi": m.index, "date": m.date, "slug_toks": m_key.values})
j = S.merge(B, on="slug_toks", suffixes=("", "_bx"))
j = j[((j.date_bx - j.date).dt.days >= -2) & ((j.date_bx - j.date).dt.days <= 14)]
j = j.drop_duplicates("mi").set_index("mi")
j["agree"] = True
print(f"Sackmann men's ITF since Dec 2023: {len(m)}; matched with odds: {len(j)} "
      f"({len(j) / max(len(m), 1):.0%}); margin median {(1 / j.odds_w + 1 / j.odds_l - 1).median():.2%}", flush=True)
print(j.groupby(j.date.dt.year).size().to_dict())
j[["odds_w", "odds_l", "agree"]].to_parquet(f"{OUT}/atp_bx_odds.parquet")
