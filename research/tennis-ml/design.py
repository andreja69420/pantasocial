"""Turn the per-player feature table into a symmetric match design matrix (player 1 vs player 2, random
orientation), plus a Barnett-Clarke style Markov serve model. No bookmaker odds enter X."""
from math import comb

import glob
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RATINGS = ["elo", "selo", "felo", "gelo", "melo", "welo", "wselo", "wsetelo"]
DERIVED_FROM = ["venue_sv_won", "sv_won_30", "rt_won_30", "sv_won_10", "rt_won_10", "surf_sv_won_30", "surf_rt_won_30"]


def f2_parts(path):
    """Rows appended by features_incr.py live in small files next to the build: {tour}_f2_parts/*.parquet."""
    return sorted(glob.glob(path[:-len(".parquet")] + "_parts/*.parquet"))


def read_f2(path, columns=None):
    """The whole feature table (build + appended parts), only COLUMNS."""
    return pd.concat([pd.read_parquet(f, columns=columns) for f in [path] + f2_parts(path)])


def load_f2(path, only=None, start="2010-01-01"):
    """The rows from START on and only the columns build_X needs for the features matching the regex ONLY (all of them
    if None): with ~630 columns the whole file next to a training run does not fit in 15 GB."""
    names = pq.read_schema(path).names
    need = {c[2:] for c in names if c.startswith("w_") and (not only or any(re.search(only, f"{p}_{c[2:]}") for p in ("p1", "p2", "d")))}
    need |= set(RATINGS) | set(DERIVED_FROM)
    cols = [c for c in names if not c.startswith(("w_", "l_", "__")) or c[2:] in need]
    parts = [pd.read_parquet(f, columns=cols) for f in f2_parts(path)]
    if not start:
        return pd.concat([pd.read_parquet(path, columns=cols)] + parts)
    df = pd.read_parquet(path, columns=cols, filters=[("date", ">=", pd.Timestamp(start))])
    # a filtered read numbers the rows from 0: put the stored index (match ids) back
    idx = pd.read_parquet(path, columns=["date"]).date.ge(pd.Timestamp(start))
    assert int(idx.sum()) == len(df)
    df.index = idx.index[idx.values]
    return pd.concat([df] + [p[p.date >= pd.Timestamp(start)] for p in parts]) if parts else df


def take(X, idx):
    """X.iloc[idx] without a copy when idx is one contiguous run (the usual case: rows are in date order)."""
    idx = np.asarray(idx)
    if idx.dtype == bool:
        idx = np.flatnonzero(idx)
    if len(idx) and idx[-1] - idx[0] == len(idx) - 1 and (np.diff(idx) == 1).all():
        return X.iloc[idx[0]:idx[-1] + 1]
    return X.iloc[idx]


def game_p(p):
    q = 1 - p
    return p ** 4 * (1 + 4 * q + 10 * q ** 2) + 20 * (p * q) ** 3 * p ** 2 / (1 - 2 * p * q)


def tb_p(r):
    q = 1 - r
    return sum(comb(6 + k, k) * r ** 7 * q ** k for k in range(6)) + comb(12, 6) * r ** 6 * q ** 6 * r ** 2 / (1 - 2 * r * q)


def set_p(pa, pb):
    """P(A wins a set) given serve-point win probabilities pa, pb; A serves first."""
    ga, gb = game_p(pa), game_p(pb)
    tb = tb_p((pa + 1 - pb) / 2)
    P = {(0, 0): np.ones_like(pa)}
    win = np.zeros_like(pa)
    for n in range(12):
        for i in range(n + 1):
            j = n - i
            if (i, j) not in P:
                continue
            pr = P.pop((i, j))
            w = ga if n % 2 == 0 else 1 - gb
            for (a, b), pp in (((i + 1, j), pr * w), ((i, j + 1), pr * (1 - w))):
                if (a == 6 and b <= 4) or a == 7:
                    win += pp
                elif not ((b == 6 and a <= 4) or b == 7):
                    P[(a, b)] = P.get((a, b), 0) + pp
    return win + P.get((6, 6), 0) * tb


def match_p(s, bo):
    return np.where(bo == 5, s ** 3 * (1 + 3 * (1 - s) + 6 * (1 - s) ** 2), s ** 2 * (1 + 2 * (1 - s)))


def build_X(df, tour, seed=7, keep=None):
    """X (float32), y, flip. keep: optional predicate on column names; only those columns are built (same values and
    order as building everything and selecting them afterwards). Memory: one float32 matrix is allocated and filled
    column by column, so the peak is about the size of X instead of several float64 copies of it."""
    rng = np.random.default_rng(seed)
    flip = rng.random(len(df)) < 0.5
    y = (~flip).astype(int)
    want = keep or (lambda c: True)
    base = [c[2:] for c in df.columns if c.startswith("w_")]

    def pair(f):  # p1, p2 in the stored dtype (differences of float32 inputs stay exact in float32)
        a, b = df[f"w_{f}"].values, df[f"l_{f}"].values
        return np.where(flip, b, a), np.where(flip, a, b)

    extra = {}
    probs = {r: 1 / (1 + 10 ** (-np.subtract(*pair(r)) / 400)) for r in RATINGS if r in base}
    for r, v in probs.items():
        extra[f"prob_{r}"] = v
    extra["prob_blend"] = pd.DataFrame(probs).mean(axis=1).values if probs else np.full(len(df), np.nan)
    tour_sv = 0.635 if tour == "atp" else 0.565
    base_sv = np.nan_to_num(pair("venue_sv_won")[0], nan=tour_sv)
    for tag, sv, rt in [("30", "sv_won_30", "rt_won_30"), ("10", "sv_won_10", "rt_won_10"),
                        ("surf", "surf_sv_won_30", "surf_rt_won_30")]:
        f1, f2 = (np.nan_to_num(x, nan=tour_sv) for x in pair(sv))
        g1, g2 = (np.nan_to_num(x, nan=1 - tour_sv) for x in pair(rt))
        pa = np.clip(base_sv + (f1 - tour_sv) - (g2 - (1 - tour_sv)), 0.3, 0.9)
        pb = np.clip(base_sv + (f2 - tour_sv) - (g1 - (1 - tour_sv)), 0.3, 0.9)
        s_ab = 0.5 * (set_p(pa, pb) + 1 - set_p(pb, pa))
        extra[f"mk_pa_{tag}"], extra[f"mk_pb_{tag}"], extra[f"mk_set_{tag}"] = pa, pb, s_ab
        extra[f"mk_match_{tag}"] = match_p(s_ab, df.best_of.fillna(3).values)
    extra["surface"] = df.surface.map({"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}).fillna(0).values
    extra["level"] = df.tourney_level.astype("category").cat.codes.values
    extra["rord"], extra["best_of"], extra["main"] = df.rord.values, df.best_of.fillna(3).values, df.main.values
    extra["draw"] = pd.to_numeric(df.draw_size, errors="coerce").values
    extra["month"] = df.date.dt.month.values

    names = [c for f in base for c in (f"p1_{f}", f"p2_{f}", f"d_{f}") if want(c)] + [c for c in extra if want(c)]
    X = np.empty((len(df), len(names)), dtype=np.float32)
    j = 0
    for f in base:
        cs = [want(f"p1_{f}"), want(f"p2_{f}"), want(f"d_{f}")]
        if not any(cs):
            continue
        p1, p2 = pair(f)
        for use, v in zip(cs, (p1, p2, p1 - p2)):
            if use:
                X[:, j] = v
                j += 1
    for c, v in extra.items():
        if want(c):
            X[:, j] = v
            j += 1
    return pd.DataFrame(X, index=df.index, columns=names, copy=False), y, flip
