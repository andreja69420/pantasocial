"""Turn the per-player feature table into a symmetric match design matrix (player 1 vs player 2, random
orientation), plus a Barnett-Clarke style Markov serve model. No bookmaker odds enter X."""
from math import comb

import numpy as np
import pandas as pd

RATINGS = ["elo", "selo", "felo", "gelo", "melo", "welo", "wselo", "wsetelo"]


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


def build_X(df, tour, seed=7):
    rng = np.random.default_rng(seed)
    flip = rng.random(len(df)) < 0.5
    y = (~flip).astype(int)
    base = [c[2:] for c in df.columns if c.startswith("w_")]
    cols = {}
    for f in base:
        a, b = df[f"w_{f}"].values, df[f"l_{f}"].values
        p1, p2 = np.where(flip, b, a), np.where(flip, a, b)
        cols[f"p1_{f}"], cols[f"p2_{f}"], cols[f"d_{f}"] = p1, p2, p1 - p2
    X = pd.DataFrame(cols, index=df.index)
    for r in [r for r in RATINGS if f"d_{r}" in X]:
        X[f"prob_{r}"] = 1 / (1 + 10 ** (-X[f"d_{r}"] / 400))
    X["prob_blend"] = X[[c for c in X.columns if c.startswith("prob_")]].mean(axis=1)

    tour_sv = 0.635 if tour == "atp" else 0.565
    base_sv = X["p1_venue_sv_won"].fillna(tour_sv).values
    for tag, sv, rt in [("30", "sv_won_30", "rt_won_30"), ("10", "sv_won_10", "rt_won_10"),
                        ("surf", "surf_sv_won_30", "surf_rt_won_30")]:
        f1, f2 = (np.nan_to_num(X[f"{p}_{sv}"].values, nan=tour_sv) for p in ("p1", "p2"))
        g1, g2 = (np.nan_to_num(X[f"{p}_{rt}"].values, nan=1 - tour_sv) for p in ("p1", "p2"))
        pa = np.clip(base_sv + (f1 - tour_sv) - (g2 - (1 - tour_sv)), 0.3, 0.9)
        pb = np.clip(base_sv + (f2 - tour_sv) - (g1 - (1 - tour_sv)), 0.3, 0.9)
        s_ab = 0.5 * (set_p(pa, pb) + 1 - set_p(pb, pa))
        X[f"mk_pa_{tag}"], X[f"mk_pb_{tag}"], X[f"mk_set_{tag}"] = pa, pb, s_ab
        X[f"mk_match_{tag}"] = match_p(s_ab, df.best_of.fillna(3).values)
    X["surface"] = df.surface.map({"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}).fillna(0).values
    X["level"] = df.tourney_level.astype("category").cat.codes.values
    X["rord"], X["best_of"], X["main"] = df.rord.values, df.best_of.fillna(3).values, df.main.values
    X["draw"] = pd.to_numeric(df.draw_size, errors="coerce").values
    X["month"] = df.date.dt.month.values
    return X.astype("float32"), y, flip
