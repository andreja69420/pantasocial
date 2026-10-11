"""Train and save the production model for live use: the same ensemble as train3.py (XGB + LGBM + logistic on the
rating probabilities, Platt calibration), trained on every season up to TRAIN_TO and calibrated / early-stopped on
the rows after it. Saves everything predict-time code needs: models, calibrator, feature list, level categories.

usage: train_final.py DATA_DIR TOUR OUT_PKL [--only=REGEX] [--train-to=2025]
"""
import pickle
import re
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from design import build_X, load_f2, take

D, TOUR, OUT = sys.argv[1:4]
ONLY = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
TRAIN_TO = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--train-to=")), 2025))
t0 = time.time()
df = load_f2(f"{D}/out/{TOUR}_f2.parquet", ONLY)
levels = sorted(df.tourney_level.astype(str).unique())
X, y, _ = build_X(df, TOUR, keep=(lambda c: re.search(ONLY, c)) if ONLY else None)
year = df.date.dt.year.values
tr, va = np.flatnonzero(year <= TRAIN_TO), np.flatnonzero(year > TRAIN_TO)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
models, pv = {}, []
m = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.06, max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.4,
                      reg_lambda=5.0, tree_method="hist", max_bin=64, n_jobs=4, early_stopping_rounds=60, eval_metric="logloss")
m.fit(take(X, tr), y[tr], eval_set=[(take(X, va), y[va])], verbose=False)
models["xgb"] = m
pv.append(m.predict_proba(take(X, va))[:, 1])
m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.05, num_leaves=48, min_child_samples=100, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.4, reg_lambda=5.0, max_bin=63, n_jobs=4, verbose=-1)
m.fit(take(X, tr), y[tr], eval_set=[(take(X, va), y[va])], callbacks=[lgb.early_stopping(60, verbose=False)])
models["lgbm"] = m
pv.append(m.predict_proba(take(X, va))[:, 1])
pcols = [c for c in X.columns if c.startswith(("prob_", "mk_match"))]
Z = lambda idx: np.nan_to_num(lg(take(X, idx)[pcols].values.astype(float)))
m = LogisticRegression(C=1.0, max_iter=500).fit(Z(tr), y[tr])
models["logit"] = m
pv.append(m.predict_proba(Z(va))[:, 1])
p = np.mean(pv, axis=0)
cal = LogisticRegression(C=1e4).fit(lg(p)[:, None], y[va])
pc = cal.predict_proba(lg(p)[:, None])[:, 1]
ll = -np.mean(y[va] * np.log(pc) + (1 - y[va]) * np.log(1 - pc))
pickle.dump({"models": models, "cal": cal, "features": list(X.columns), "pcols": pcols, "levels": levels, "tour": TOUR,
             "train_to": TRAIN_TO, "val_rows": len(va), "val_logloss": ll}, open(OUT, "wb"))
print(f"{TOUR}: trained on <= {TRAIN_TO} ({len(tr)} rows), calibrated on {len(va)} later rows, log-loss {ll:.4f}, "
      f"{X.shape[1]} features, {time.time() - t0:.0f}s -> {OUT}", flush=True)
