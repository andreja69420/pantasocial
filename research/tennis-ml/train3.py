"""Walk-forward from 2010: for each test year Y the model trains on 2010..Y-2, early-stops and calibrates on
Y-1, and predicts Y's tour-level matches that have Bet365 odds. Odds are never features.

usage: train3.py DATA_DIR TOUR MODEL TAG [--holdout] [--only=REGEX]
  MODEL in {xgb, lgbm, ens}; dev = test years 2014-2024, --holdout = test 2025-2026 (run once, at the end)
Every run appends one line per betting rule to experiments.csv, so the number of things tried stays visible.
"""
import os
import re
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from design import build_X

D, TOUR, MODEL, TAG = sys.argv[1:5]
HOLDOUT = "--holdout" in sys.argv
ONLY = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
NO_FUTURES = "--no-futures" in sys.argv  # futures/ITF rows still feed the ratings, just not the training set
HALFLIFE = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--halflife=")), 0))
STEP = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--step=")), 1))  # retrain every STEP years
TRAIN_FROM = 2010
LOWER = "--lower" in sys.argv  # evaluate on Challenger/ITF/qualifying matches with Kaggle odds (2014-2018)
TEST_YEARS = [2025, 2026] if HOLDOUT else (list(range(2014, 2019)) if LOWER else list(range(2014, 2025)))
YEARS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--years=")), None)
if YEARS:
    TEST_YEARS = [int(v) for v in YEARS.split(",")]
# boosting budget: learning-rate multiplier, max trees, early-stopping patience (defaults = the tuned round-6 setup)
LR_MULT = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--lr-mult=")), 1.0))
TREES = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--trees=")), 1500))
PATIENCE = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--patience=")), 60))
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.csv")

df = pd.read_parquet(f"{D}/out/{TOUR}_f2.parquet")
df["year"] = df.date.dt.year
df = df[df.year >= TRAIN_FROM].copy()
X, y, flip = build_X(df, TOUR)
if ONLY:
    X = X[[c for c in X.columns if re.search(ONLY, c)]]
FEATS = list(X.columns)
lvl = df.tourney_level.astype(str)
futures = (lvl.str.match(r"^\d") | (lvl == "S")).values
if LOWER:
    ODDS_FILE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--odds-file=")), "lower_odds")
    lo = pd.read_parquet(f"{D}/out/{TOUR}_{ODDS_FILE}.parquet")
    lo = lo.reindex(df.index)
    df["odds_w"], df["odds_l"] = lo.odds_w.astype(float), lo.odds_l.astype(float)
    df["agree"] = lo.agree.fillna(False).astype(bool)
has_odds = (df.odds_w.notna() & df.agree.fillna(False).astype(bool)).values
year = df.year.values
print(f"{TOUR} {MODEL} {TAG}: {len(FEATS)} features, rows {len(X)}", flush=True)


def fit_predict(tr, va, te, Y):
    preds_va, preds_te = [], []
    sw = 0.5 ** ((Y - year[tr]) / HALFLIFE) if HALFLIFE > 0 else np.ones(len(tr))
    if MODEL in ("xgb", "ens"):
        m = xgb.XGBClassifier(n_estimators=TREES, learning_rate=0.06 * LR_MULT, max_depth=6, min_child_weight=50, subsample=0.8,
                              colsample_bytree=0.4, reg_lambda=5.0, tree_method="hist", max_bin=64, n_jobs=2,
                              early_stopping_rounds=PATIENCE, eval_metric="logloss")
        m.fit(X.iloc[tr], y[tr], sample_weight=sw, eval_set=[(X.iloc[va], y[va])], verbose=False)
        preds_va.append(m.predict_proba(X.iloc[va])[:, 1]); preds_te.append(m.predict_proba(X.iloc[te])[:, 1])
        print(f"    xgb best iteration {m.best_iteration}", flush=True)
    if MODEL in ("lgbm", "ens"):
        m = lgb.LGBMClassifier(n_estimators=TREES, learning_rate=0.05 * LR_MULT, num_leaves=48, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0, max_bin=63,
                               n_jobs=2, verbose=-1)
        m.fit(X.iloc[tr], y[tr], sample_weight=sw, eval_set=[(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        preds_va.append(m.predict_proba(X.iloc[va])[:, 1]); preds_te.append(m.predict_proba(X.iloc[te])[:, 1])
        print(f"    lgbm best iteration {m.best_iteration_}", flush=True)
    if MODEL == "ens":
        # simple, well-calibrated anchor: logistic regression on the rating/Markov probabilities
        cols = [c for c in FEATS if c.startswith(("prob_", "mk_match"))]
        lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        Z = lambda idx: np.nan_to_num(lg(X.iloc[idx][cols].values.astype(float)))
        m = LogisticRegression(C=1.0, max_iter=500).fit(Z(tr), y[tr], sample_weight=sw)
        preds_va.append(m.predict_proba(Z(va))[:, 1]); preds_te.append(m.predict_proba(Z(te))[:, 1])
    pv, pt = np.mean(preds_va, axis=0), np.mean(preds_te, axis=0)
    # Platt scaling on the validation year (isotonic on a single year overfit in round 3)
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))[:, None]
    cal = LogisticRegression(C=1e4).fit(lg(pv), y[va])
    return pt, cal.predict_proba(lg(pt))[:, 1]


def stands(score, retired):
    if not retired:
        return True
    sets = re.findall(r"(\d+)-(\d+)", score.split("RET")[0])
    return any(max(int(a), int(b)) >= 6 and (abs(int(a) - int(b)) >= 2 or max(int(a), int(b)) == 7) for a, b in sets)


t0 = time.time()
rows = []
for Y in TEST_YEARS[::STEP]:
    tr = (year >= TRAIN_FROM) & (year <= Y - 2)
    if NO_FUTURES:
        tr &= ~futures
    va = year == Y - 1
    te = (year >= Y) & (year < Y + STEP) & (year <= max(TEST_YEARS)) & has_odds
    if te.sum() == 0:
        continue
    raw, cal = fit_predict(np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te), Y)
    sub = df.iloc[np.flatnonzero(te)]
    o1 = np.where(flip[te], sub.odds_l, sub.odds_w)
    o2 = np.where(flip[te], sub.odds_w, sub.odds_l)
    rows.append(pd.DataFrame({"year": year[te], "p_raw": raw, "p": cal, "y": y[te], "o1": o1, "o2": o2,
                              "stands": [stands(s, r) for s, r in zip(sub.score, sub.retired)]}, index=sub.index))
    print(f"  {Y}: trained in {time.time() - t0:.0f}s", flush=True)
P = pd.concat(rows)
inv = 1 / P.o1 + 1 / P.o2
P["p_mkt"] = (1 / P.o1) / inv
split = "holdout" if HOLDOUT else ((("te" if "--odds-file=te_odds" in sys.argv else "lower")) if LOWER else "dev")
P.to_parquet(f"{D}/out/preds3_{TOUR}_{MODEL}_{TAG}_{split}{'_' + YEARS.replace(',', '-') if YEARS else ''}.parquet")


def ll(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return -np.mean(P.y * np.log(q) + (1 - P.y) * np.log(1 - q))


acc = lambda q: np.mean((q > .5) == P.y)
print(f"{split} {min(P.year)}-{max(P.year)}: {len(P)} matches, margin {(inv - 1).median():.2%}")
print(f"logloss market {ll(P.p_mkt):.4f} | model raw {ll(P.p_raw):.4f} calibrated {ll(P.p):.4f} | "
      f"accuracy market {acc(P.p_mkt):.3f} model {acc(P.p):.3f}")
if acc(P.p) > acc(P.p_mkt) + 0.03:
    print("!!! LEAKAGE SUSPECTED: model accuracy >3 points above the market")


def bets(thr, max_odds, pcol="p"):
    out = []
    for s in (1, 2):
        o = P.o1 if s == 1 else P.o2
        q = P[pcol] if s == 1 else 1 - P[pcol]
        win = (P.y == 1) if s == 1 else (P.y == 0)
        pick = (q * o - 1 > thr) & (o <= max_odds)
        out.append(pd.DataFrame({"pnl": np.where(P.stands, np.where(win, o - 1, -1.0), 0.0), "year": P.year})[pick])
    return pd.concat(out)


rng = np.random.default_rng(0)
log_rows = []
print(f"{'rule':24s} {'bets':>6s} {'ROI':>7s} {'90% CI':>17s}  years+   pass?")
for thr in [0.02, 0.05, 0.10]:
    for mo in [2.0, 3.0, 100.0]:
        b = bets(thr, mo)
        if len(b) < 30:
            continue
        v = b.pnl.values
        boot = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)])
        lo, hi = np.percentile(boot, [5, 95])
        yrs = b.groupby("year").pnl.mean()
        pos = int((yrs > 0).sum())
        ok = (v.mean() > 0) and lo > 0 and pos >= 0.75 * len(yrs) and len(v) >= 300
        rule = f"EV>{thr:.0%} odds<={mo:g}"
        print(f"{rule:24s} {len(v):6d} {v.mean():+7.2%} [{lo:+6.1%},{hi:+6.1%}]  {pos}/{len(yrs)}   {'PASS' if ok else ''}")
        log_rows.append({"time": pd.Timestamp.now().isoformat(timespec="seconds"), "tour": TOUR, "model": MODEL,
                         "tag": TAG, "split": split, "n_features": len(FEATS), "rule": rule, "bets": len(v),
                         "roi": round(v.mean(), 4), "ci90_lo": round(lo, 4), "years_pos": f"{pos}/{len(yrs)}",
                         "logloss_model": round(ll(P.p), 5), "logloss_market": round(ll(P.p_mkt), 5), "pass": ok})
if "--no-log" not in sys.argv:  # prediction-only runs (placeholder odds) are not experiments
    pd.DataFrame(log_rows).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
