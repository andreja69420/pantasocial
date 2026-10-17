"""Flashscore match statistics parsers shared by fs_stats.py (seasons), fs_update.py (new matches in the loop) and
fs_stats_fill.py (holes): the box score feed df_st_1_<id> (aces, double faults, 1st / 2nd serve points won, break points
saved, service games won) and the point-by-point feed df_mh_1_<id> (service points played / won, service games, holds, break
points faced / saved, tiebreak points, counted exactly), for one match. Box score columns follow the Sackmann names
(ace, df, svpt, 1stIn, 1stWon, 2ndWon, SvGms, bpSaved, bpFaced); point-by-point counts go to separate pbp_ columns.
"""
import re

import numpy as np

from fs_common import UA

BOX = {"Aces": "ace", "Double Faults": "df", "1st serve points won": ("1stWon", "1stIn"), "2nd serve points won": ("2ndWon", "2ndIn"),
       "Break Points Saved": ("bpSaved", "bpFaced"), "Service games won": ("SvGmsWon", "SvGms")}
PT = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4}


def fields(text):
    return [dict(x.split("÷", 1) for x in rec.split("¬") if "÷" in x) for rec in text.split("~")]


def box(text):
    """{(side, stat): value} of the whole-match section, side 0 = home, 1 = away."""
    out = {}
    for d in fields(text):
        if d.get("SE", "Match") != "Match":
            break
        st = BOX.get(d.get("SG"))
        if not st:
            continue
        for side, v in enumerate((d.get("SH", ""), d.get("SI", ""))):
            m = re.search(r"\((\d+)/(\d+)\)", v) or re.fullmatch(r"(\d+)/(\d+)", v.strip())
            if isinstance(st, tuple) and m:
                out[side, st[0]], out[side, st[1]] = int(m.group(1)), int(m.group(2))
            elif isinstance(st, str) and v.strip().isdigit():
                out[side, st] = int(v)
    return out


def pbp(text):
    """Serve counts per side from the point-by-point log: svpt, svwon, SvGms, holds, bpFaced, bpSaved, tbpt, tbwon."""
    c = np.zeros((2, 8), dtype=int)
    tb = False
    for d in fields(text):
        if "HB" in d:
            tb = d["HB"].startswith("Tiebreak")
        if "HG" not in d or "HK" not in d:
            continue
        srv, win = int(d["HG"]) - 1, int(d["HK"]) - 1
        if tb:  # one row per tiebreak point
            c[srv, 6] += 1
            c[srv, 7] += win == srv
            continue
        c[srv, 2] += 1
        c[srv, 3] += win == srv
        if "HL" not in d:
            continue
        states = [(0, 0)]
        for s in d["HL"].split(","):
            h, _, a = re.sub(r"\|B\d\|", "", s).strip().partition(":")
            if h.strip() not in PT or a.strip() not in PT:
                states = None
                break
            states.append((PT[h.strip()], PT[a.strip()]))
        if not states:
            continue
        for i, (h, a) in enumerate(states):
            if i + 1 < len(states):
                h2, a2 = states[i + 1]
                pw = 0 if (h2 > h or a2 < a) else 1
            else:
                pw = win
            sv, rc = (h, a) if srv == 0 else (a, h)
            bp = (rc == 3 and sv < 3) or rc == 4
            c[srv, 0] += 1
            c[srv, 1] += pw == srv
            c[srv, 4] += bp
            c[srv, 5] += bp and pw == srv
    return c if c[:, 2].sum() else None


BOX_COLS = ("ace", "df", "1stIn", "1stWon", "2ndIn", "2ndWon", "SvGms", "SvGmsWon", "bpSaved", "bpFaced")
PBP_COLS = ("svpt", "svwon", "SvGms", "holds", "bpFaced", "bpSaved", "tbpt", "tbwon")


def match_stats(L, mid):
    """(box, pbp) of one finished match: box = {(side, stat): value} (side 0 = home), pbp = 2x8 array or None. The
    point-by-point log is only fetched when there is no box score."""
    b = box(L.get(f"https://www.flashscore.com/x/feed/df_st_1_{mid}", UA))
    return b, (None if "SvGms" in {k for _, k in b} else pbp(L.get(f"https://www.flashscore.com/x/feed/df_mh_1_{mid}", UA)))


def stat_columns(b, p, home_won):
    """w_/l_ columns (winner / loser of the match) from match_stats; NaN where Flashscore has nothing."""
    w = 0 if home_won else 1
    r = {}
    for side, s in ((w, "w"), (1 - w, "l")):
        for st in BOX_COLS:
            r[f"{s}_{st}"] = b.get((side, st), np.nan)
        for j, st in enumerate(PBP_COLS):
            r[f"{s}_pbp_{st}"] = p[side, j] if p is not None else np.nan
        r[f"{s}_svpt"] = r[f"{s}_1stIn"] + r[f"{s}_2ndIn"]
    return r
