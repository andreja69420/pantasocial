"""Walk-forward XGBoost on tennis matches (trained on all levels), evaluated as a betting strategy against
Bet365 prices on tour-level matches. Margin is removed for the 'fair' market probability, but bets are
always settled at the real (margin-included) odds."""
import re
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

D, TOUR = sys.argv[1], sys.argv[2]
FIRST_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 2008
TRAIN_YEARS = 12
rng = np.random.default_rng(42)

m = pd.read_parquet(f"{D}/out/{TOUR}_features.parquet")
m = m[~m["walkover"]].copy()
m["year"] = m["date"].dt.year
base = ["elo", "selo", "n", "ns", "rest", "form10", "win52", "n52", "n14", "surf52", "svpct", "rtpct", "min7", "h2h"]

# Randomly orient each match so the label is not always "player 1 wins".
flip = rng.random(len(m)) < 0.5
y = (~flip).astype(int)


def side(a, b):
    return np.where(flip, m[b].values, m[a].values)


X = pd.DataFrame(index=m.index)
for f in base:
    p1, p2 = side(f"w_{f}", f"l_{f}"), side(f"l_{f}", f"w_{f}")
    X[f"d_{f}"] = p1 - p2
    if f in ("n", "rest", "n52", "n14", "svpct", "rtpct", "form10"):
        X[f"p1_{f}"], X[f"p2_{f}"] = p1, p2
for a, b, name, fn in [("winner_rank", "loser_rank", "lrank", lambda v: np.log(np.nan_to_num(v, nan=3000))),
                       ("winner_rank_points", "loser_rank_points", "lpts", lambda v: np.log1p(np.nan_to_num(v, nan=0))),
                       ("winner_age", "loser_age", "age", lambda v: v), ("winner_ht", "loser_ht", "ht", lambda v: v)]:
    p1, p2 = fn(side(a, b).astype(float)), fn(side(b, a).astype(float))
    X[f"d_{name}"] = p1 - p2
    if name in ("lrank", "age"):
        X[f"p1_{name}"], X[f"p2_{name}"] = p1, p2
X["p1_left"] = (side("winner_hand", "loser_hand") == "L").astype(int)
X["p2_left"] = (side("loser_hand", "winner_hand") == "L").astype(int)
elo_p = lambda a, b: 1 / (1 + 10 ** ((b - a) / 400))
X["elo_prob"] = elo_p(side("w_elo", "l_elo"), side("l_elo", "w_elo"))
X["selo_prob"] = elo_p(side("w_selo", "l_selo"), side("l_selo", "w_selo"))
X["blend_prob"] = 0.5 * X["elo_prob"] + 0.5 * X["selo_prob"]
X["surface"] = m["surface"].map({"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}).fillna(0).values
X["level"] = m["tourney_level"].astype("category").cat.codes.values
X["main"] = (m["src"] == "main").astype(int).values
X["rord"] = m["rord"].values
X["best_of"] = m["best_of"].fillna(3).values
FEATS = list(X.columns)

odds1 = np.where(flip, m["odds_l"], m["odds_w"])
odds2 = np.where(flip, m["odds_w"], m["odds_l"])


def bet_stands(score, retired):
    """Bet365/Pinnacle-style: a retirement voids bets unless the first set was completed."""
    if not retired:
        return True
    sets = re.findall(r"(\d+)-(\d+)", score.split("RET")[0])
    return any(max(int(a), int(b)) >= 6 and (abs(int(a) - int(b)) >= 2 or max(int(a), int(b)) == 7) for a, b in sets)


stands = np.array([bet_stands(s, r) for s, r in zip(m["score"], m["retired"])])
has_odds = m["odds_w"].notna().values & m["winner_agrees"].fillna(False).astype(bool).values

preds = []
years = sorted(m.loc[has_odds, "year"].unique())
for Y in [yy for yy in years if yy >= FIRST_TEST]:
    tr = (m["year"] < Y).values & (m["year"] >= Y - TRAIN_YEARS).values
    te = (m["year"] == Y).values & has_odds
    clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=5, min_child_weight=50,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", n_jobs=4)
    clf.fit(X.loc[tr, FEATS], y[tr])
    p = clf.predict_proba(X.loc[te, FEATS])[:, 1]
    preds.append(pd.DataFrame({"year": Y, "p_model": p, "p_elo": X.loc[te, "blend_prob"].values,
                               "y": y[te], "o1": odds1[te], "o2": odds2[te], "stands": stands[te],
                               "level": m.loc[te, "tourney_level"].values, "date": m.loc[te, "date"].values},
                              index=m.index[te]))
    print(Y, "train", tr.sum(), "test", te.sum(), flush=True)
P = pd.concat(preds)
inv = 1 / P.o1 + 1 / P.o2
P["margin"] = inv - 1
P["p_mkt"] = (1 / P.o1) / inv  # proportional margin removal

# Stacking: does the model carry information the price does not? Fit on prior years only.
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
P["p_stack"] = np.nan
coefs = {}
for Y in sorted(P.year.unique()):
    prev, cur = P.year < Y, P.year == Y
    if prev.sum() < 3000:
        continue
    lr = LogisticRegression(C=1e6).fit(np.c_[logit(P.loc[prev, "p_mkt"]), logit(P.loc[prev, "p_model"])], P.loc[prev, "y"])
    coefs[Y] = lr.coef_[0]
    P.loc[cur, "p_stack"] = lr.predict_proba(np.c_[logit(P.loc[cur, "p_mkt"]), logit(P.loc[cur, "p_model"])])[:, 1]


def ll(p, yv):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))


print(f"\n=== {TOUR.upper()} {P.year.min()}-{P.year.max()}: {len(P)} matches with Bet365 odds; median margin {P.margin.median():.2%}")
print("Prediction quality (lower log-loss / Brier = better):")
for name in ["p_mkt", "p_model", "p_elo", "p_stack"]:
    s = P.dropna(subset=[name])
    print(f"  {name:8s} logloss {ll(s[name], s.y):.4f}  brier {np.mean((s[name] - s.y) ** 2):.4f}  "
          f"accuracy {np.mean((s[name] > .5) == s.y):.3f}  (n={len(s)})")
print("  stacking coefficients [market, model] by year:", {k: np.round(v, 2).tolist() for k, v in list(coefs.items())[-5:]})


def simulate(P, pcol, thr, min_odds=1.0, max_odds=99):
    rows = []
    for side_, o, win in [(1, P.o1, P.y == 1), (2, P.o2, P.y == 0)]:
        p = P[pcol] if side_ == 1 else 1 - P[pcol]
        ev = p * o - 1
        pick = (ev > thr) & (o >= min_odds) & (o <= max_odds) & P[pcol].notna()
        pnl = np.where(P.stands, np.where(win, o - 1, -1.0), 0.0)
        rows.append(pd.DataFrame({"year": P.year, "odds": o, "pnl": pnl, "ev": ev})[pick])
    return pd.concat(rows)


def summarize(b):
    if len(b) == 0:
        return {"bets": 0}
    boot = [b.pnl.sample(frac=1, replace=True, random_state=i).mean() for i in range(300)]
    return {"bets": len(b), "avg_odds": b.odds.mean(), "ROI": b.pnl.mean(), "profit_units": b.pnl.sum(),
            "ROI_CI95": f"[{np.percentile(boot, 2.5):+.1%}, {np.percentile(boot, 97.5):+.1%}]",
            "winning_years": f"{(b.groupby('year').pnl.sum() > 0).sum()}/{b.year.nunique()}"}


rows = {}
for name, pcol in [("market fair prob (control: should lose ~margin)", "p_mkt"), ("XGBoost model", "p_model"),
                   ("Elo only", "p_elo"), ("Stacked market+model", "p_stack")]:
    for thr in [0.0, 0.05, 0.10, 0.20]:
        rows[f"{name} | EV>{thr:.0%}"] = summarize(simulate(P, pcol, thr))
rows["bet every favourite"] = summarize(simulate(P.assign(fav=(P.o1 < P.o2).astype(float)), "fav", -0.99, 1, 99).query("odds < 2"))
rows["bet every underdog"] = summarize(simulate(P.assign(dog=(P.o1 > P.o2).astype(float)), "dog", -0.99, 1, 99).query("odds > 2"))
res = pd.DataFrame(rows).T
pd.set_option("display.width", 220)
print("\nFlat 1-unit bets at Bet365 prices:")
print(res.to_string(float_format=lambda v: f"{v:.3f}"))
b = simulate(P, "p_model", 0.05)
print("\nXGBoost EV>5% by year:")
print(b.groupby("year").agg(bets=("pnl", "size"), ROI=("pnl", "mean"), profit=("pnl", "sum")).round(3).T.to_string())
P.to_parquet(f"{D}/out/{TOUR}_preds.parquet")
res.to_csv(f"{D}/out/{TOUR}_betting.csv")
