"""Total-games model (over/under market). Walk-forward like train3: test year Y trains on 2010..Y-2 and fits the
residual distribution on Y-1. The model predicts the mean total games from the same 481 match features (no odds);
P(over line) comes from mean + the validation-year residual distribution (by best-of). Retired/walkover matches are
dropped from training and voided at settlement.

usage: train_totals.py DATA_DIR TOUR [--step=N]
Writes out/tot_{TOUR}.parquet with year, mu, the validation residuals' quantile info and the realised total.
"""
import re
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from design import build_X

D, TOUR = sys.argv[1:3]
STEP = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--step=")), 2))
TEST_YEARS = list(range(2014, 2026))
TRAIN_FROM = 2010


def total_games(score):
    if not isinstance(score, str) or re.search(r"RET|W/O|DEF|ABD|ABN|unfinished", score, re.I):
        return np.nan
    n = 0
    for tok in score.split():
        if tok.startswith("["):
            n += 1  # match tiebreak counts as one game
            continue
        m = re.match(r"(\d+)-(\d+)", tok)
        if not m:
            return np.nan
        n += int(m.group(1)) + int(m.group(2))
    return n if n > 0 else np.nan


df = pd.read_parquet(f"{D}/out/{TOUR}_f2.parquet")
df["year"] = df.date.dt.year
df = df[df.year >= TRAIN_FROM].copy()
X, _, _ = build_X(df, TOUR)
tg = df.score.map(total_games).values.astype(float)
E = pd.read_parquet(f"{D}/out/{TOUR}_te_extra.parquet").reindex(df.index)
has_line = E[["b365_tline", "ub_tline", "lb_tline"]].notna().any(axis=1).values
year, bo = df.year.values, df.best_of.values
print(f"{TOUR}: {X.shape[1]} features; matches with a totals line {has_line.sum()}", flush=True)
t0, out = time.time(), []
for Y in TEST_YEARS[::STEP]:
    tr = np.flatnonzero((year <= Y - 2) & ~np.isnan(tg))
    va = np.flatnonzero((year == Y - 1) & ~np.isnan(tg))
    te = np.flatnonzero((year >= Y) & (year < Y + STEP) & has_line)
    preds_va, preds_te = [], []
    m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6, min_child_weight=50, subsample=0.8,
                         colsample_bytree=0.4, reg_lambda=5.0, tree_method="hist", max_bin=64, n_jobs=3, early_stopping_rounds=60)
    m.fit(X.iloc[tr], tg[tr], eval_set=[(X.iloc[va], tg[va])], verbose=False)
    preds_va.append(m.predict(X.iloc[va])); preds_te.append(m.predict(X.iloc[te]))
    m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.04, num_leaves=48, min_child_samples=100, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0, max_bin=63, n_jobs=3, verbose=-1)
    m.fit(X.iloc[tr], tg[tr], eval_set=[(X.iloc[va], tg[va])], callbacks=[lgb.early_stopping(60, verbose=False)])
    preds_va.append(m.predict(X.iloc[va])); preds_te.append(m.predict(X.iloc[te]))
    mu_va, mu_te = np.mean(preds_va, axis=0), np.mean(preds_te, axis=0)
    # residual pools by best-of from the validation year: P(total > L) = mean(mu + r > L)
    res = {b: np.sort(tg[va][bo[va] == b] - mu_va[bo[va] == b]) for b in (3, 5) if (bo[va] == b).sum() > 200}
    r = pd.DataFrame({"year": year[te], "mu": mu_te, "bo": bo[te], "total": tg[te]}, index=df.index[te])
    for b, pool in res.items():
        np.save(f"{D}/out/tot_res_{TOUR}_{Y}_bo{b}.npy", pool)
    r["res_key"] = [f"{Y}_bo{b if b in res else 3}" for b in bo[te]]
    out.append(r)
    rmse = np.sqrt(np.nanmean((tg[te] - mu_te) ** 2))
    print(f"  {Y}: trained {time.time() - t0:.0f}s, test {len(te)}, rmse {rmse:.2f}, sd(total) {np.nanstd(tg[te]):.2f}", flush=True)
pd.concat(out).to_parquet(f"{D}/out/tot_{TOUR}.parquet")
