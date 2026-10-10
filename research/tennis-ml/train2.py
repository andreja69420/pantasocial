"""Fast experiment loop on the >150-feature set. Odds are never features: they only decide value and settle bets.

usage: train2.py DATA_DIR TOUR [dev|holdout]
  dev     : train 2010-2019 (+2020 for early stopping), evaluate 2021-2024
  holdout : train 2010-2023 (+2024 for early stopping), evaluate 2025-2026  (run once, at the end)
"""
import re
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

D, TOUR = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "dev"
MAIN_W = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--mainw=")), 1))
SPLITS = {"dev": (2010, 2020, 2021, 2024), "holdout": (2010, 2024, 2025, 2026)}
TR0, VAL, TE0, TE1 = SPLITS[MODE]

df = pd.read_parquet(f"{D}/out/{TOUR}_f2.parquet")
df["year"] = df.date.dt.year
df = df[df.year >= TR0].copy()
rng = np.random.default_rng(7)
flip = rng.random(len(df)) < 0.5
y = (~flip).astype(int)
base = [c[2:] for c in df.columns if c.startswith("w_")]

cols = {}
for f in base:
    a, b = df[f"w_{f}"].values, df[f"l_{f}"].values
    p1, p2 = np.where(flip, b, a), np.where(flip, a, b)
    cols[f"p1_{f}"], cols[f"p2_{f}"], cols[f"d_{f}"] = p1, p2, p1 - p2
X = pd.DataFrame(cols, index=df.index)
for r in ["elo", "selo", "felo", "gelo", "melo"]:
    X[f"prob_{r}"] = 1 / (1 + 10 ** (-X[f"d_{r}"] / 400))
X["prob_blend"] = X[["prob_elo", "prob_selo", "prob_gelo"]].mean(axis=1)


def game_p(p):
    q = 1 - p
    return p ** 4 * (1 + 4 * q + 10 * q ** 2) + 20 * (p * q) ** 3 * p ** 2 / (1 - 2 * p * q)


def tb_p(r):
    from math import comb
    q = 1 - r
    return sum(comb(6 + k, k) * r ** 7 * q ** k for k in range(6)) + comb(12, 6) * r ** 6 * q ** 6 * r ** 2 / (1 - 2 * r * q)


def set_p(pa, pb):
    """Probability A wins a set; pa/pb = serve-point win probabilities of A and B. A serves first."""
    ga, gb = game_p(pa), game_p(pb)
    tb = tb_p((pa + 1 - pb) / 2)
    P = {(0, 0): np.ones_like(pa)}
    win = np.zeros_like(pa)
    for n in range(12):
        for i in range(n + 1):
            j = n - i
            if (i, j) not in P or i > 6 or j > 6:
                continue
            pr = P.pop((i, j))
            w = ga if n % 2 == 0 else 1 - gb
            for (a, b), pp in (((i + 1, j), pr * w), ((i, j + 1), pr * (1 - w))):
                if (a == 6 and b <= 4) or a == 7:
                    win += pp
                elif (b == 6 and a <= 4) or b == 7:
                    pass
                else:
                    P[(a, b)] = P.get((a, b), 0) + pp
    return win + P.get((6, 6), 0) * tb


def match_p(s, bo):
    return np.where(bo == 5, s ** 3 * (1 + 3 * (1 - s) + 6 * (1 - s) ** 2), s ** 2 * (1 + 2 * (1 - s)))


tour_sv = 0.635 if TOUR == "atp" else 0.565
for tag, sv1, sv2, rt1, rt2 in [("30", "sv_won_30", "sv_won_30", "rt_won_30", "rt_won_30"),
                                ("10", "sv_won_10", "sv_won_10", "rt_won_10", "rt_won_10"),
                                ("surf", "surf_sv_won_30", "surf_sv_won_30", "surf_rt_won_30", "surf_rt_won_30")]:
    base_sv = pd.Series(X["p1_venue_sv_won"]).fillna(tour_sv).values
    f1, f2 = np.nan_to_num(X[f"p1_{sv1}"].values, nan=tour_sv), np.nan_to_num(X[f"p2_{sv2}"].values, nan=tour_sv)
    g1, g2 = np.nan_to_num(X[f"p1_{rt1}"].values, nan=1 - tour_sv), np.nan_to_num(X[f"p2_{rt2}"].values, nan=1 - tour_sv)
    pa = np.clip(base_sv + (f1 - tour_sv) - (g2 - (1 - tour_sv)), 0.3, 0.9)
    pb = np.clip(base_sv + (f2 - tour_sv) - (g1 - (1 - tour_sv)), 0.3, 0.9)
    s_ab = 0.5 * (set_p(pa, pb) + 1 - set_p(pb, pa))
    X[f"mk_pa_{tag}"], X[f"mk_pb_{tag}"] = pa, pb
    X[f"mk_set_{tag}"] = s_ab
    X[f"mk_match_{tag}"] = match_p(s_ab, df.best_of.fillna(3).values)
X["surface"] = df.surface.map({"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}).fillna(0).values
X["level"] = df.tourney_level.astype("category").cat.codes.values
X["rord"], X["best_of"], X["main"] = df.rord.values, df.best_of.fillna(3).values, df.main.values
X["draw"] = pd.to_numeric(df.draw_size, errors="coerce").values
X["month"] = df.date.dt.month.values
FEATS = list(X.columns)
print(f"{TOUR} {MODE}: {len(FEATS)} features", flush=True)

tr = (df.year >= TR0) & (df.year < VAL)
va = df.year == VAL
has_odds = df.odds_w.notna() & df.agree.fillna(False).astype(bool)
te = (df.year >= TE0) & (df.year <= TE1) & has_odds

clf = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=6, min_child_weight=50, subsample=0.8,
                        colsample_bytree=0.5, reg_lambda=5.0, tree_method="hist", max_bin=128, n_jobs=4,
                        early_stopping_rounds=100, eval_metric="logloss")
w = np.where(df.main.values[tr.values] == 1, MAIN_W, 1.0)
clf.fit(X[tr], y[tr.values], sample_weight=w, eval_set=[(X[va], y[va.values])], verbose=False)
print("trees", clf.best_iteration, flush=True)
p_raw = clf.predict_proba(X[te])[:, 1]
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(clf.predict_proba(X[va])[:, 1], y[va.values])
p = iso.predict(p_raw) if "--calib" in sys.argv else p_raw

o1 = np.where(flip, df.odds_l, df.odds_w)[te.values]
o2 = np.where(flip, df.odds_w, df.odds_l)[te.values]
yt = y[te.values]


def stands(score, retired):
    if not retired:
        return True
    sets = re.findall(r"(\d+)-(\d+)", score.split("RET")[0])
    return any(max(int(a), int(b)) >= 6 and (abs(int(a) - int(b)) >= 2 or max(int(a), int(b)) == 7) for a, b in sets)


st = np.array([stands(s, r) for s, r in zip(df.score[te], df.retired[te])])
P = pd.DataFrame({"p": p, "y": yt, "o1": o1, "o2": o2, "stands": st, "year": df.year[te].values,
                  "level": df.tourney_level[te].values}, index=df.index[te])
inv = 1 / P.o1 + 1 / P.o2
P["p_mkt"] = (1 / P.o1) / inv
P.to_parquet(f"{D}/out/{TOUR}_{MODE}_preds2.parquet")


def ll(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return -np.mean(P.y * np.log(q) + (1 - P.y) * np.log(1 - q))


print(f"matches {len(P)}  margin {(inv - 1).median():.2%}")
if np.mean((P.p > .5) == P.y) > np.mean((P.p_mkt > .5) == P.y) + 0.03:
    print("!!! LEAKAGE SUSPECTED: model accuracy is >3 points above the market. Results below are not trustworthy.")
print(f"logloss  market {ll(P.p_mkt):.4f}   model {ll(P.p):.4f}   | accuracy market {np.mean((P.p_mkt > .5) == P.y):.3f}"
      f"  model {np.mean((P.p > .5) == P.y):.3f}")


def bets(thr, lo=1.0, hi=100.0):
    out = []
    for s in (1, 2):
        o = P.o1 if s == 1 else P.o2
        q = P.p if s == 1 else 1 - P.p
        win = (P.y == 1) if s == 1 else (P.y == 0)
        pick = (q * o - 1 > thr) & (o >= lo) & (o <= hi)
        out.append(pd.DataFrame({"pnl": np.where(P.stands, np.where(win, o - 1, -1.0), 0.0), "odds": o,
                                 "year": P.year})[pick])
    return pd.concat(out)


print("value bets at Bet365 prices (flat 1 unit):")
for thr in [0.0, 0.05, 0.10, 0.20]:
    for lo, hi in [(1.0, 100), (1.0, 2.0), (2.0, 100)]:
        b = bets(thr, lo, hi)
        if len(b) < 20:
            continue
        se = b.pnl.std() / np.sqrt(len(b))
        yrs = b.groupby("year").pnl.mean()
        print(f"  EV>{thr:4.0%} odds {lo:.1f}-{hi:<5.1f} bets {len(b):5d}  ROI {b.pnl.mean():+6.1%} ± {1.96 * se:4.1%}  "
              f"by year: " + " ".join(f"{yy}:{v:+.1%}" for yy, v in yrs.items()))
imp = pd.Series(clf.feature_importances_, index=FEATS).sort_values(ascending=False)
print("top features:", ", ".join(imp.index[:15]))
