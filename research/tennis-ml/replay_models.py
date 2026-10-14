"""Run a saved production model (train_final.py pickle) over finished matches of a build, as the live loop would have
priced them: probability that the winner wins, per match. Used to compare the live model (models_v3), a retrained
candidate and the holdout model on the same matches.

usage: replay_models.py DATA_DIR TOUR MODEL_PKL FROM_DATE OUT_PARQUET [--only=REGEX]
  DATA_DIR/out/{tour}_f2.parquet is the feature table (with the nx_ columns), rows with date >= FROM_DATE are scored.
  OUT_PARQUET: index = match id, column p_winner. The orientation is re-drawn for the selected rows (as on every live
  pass), p_winner is the probability of the side that won and does not depend on it except for model noise.
"""
import pickle
import re
import sys

import numpy as np
import pandas as pd

from design import build_X, load_f2

D, TOUR, PKL, FROM, OUT = sys.argv[1:6]
ONLY = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
M = pickle.load(open(PKL, "rb"))
df = load_f2(f"{D}/out/{TOUR}_f2.parquet", ONLY, start=FROM)
X, y, flip = build_X(df, TOUR, keep=(lambda c: re.search(ONLY, c)) if ONLY else None)
X["level"] = pd.Categorical(df.tourney_level.astype(str), categories=M["levels"]).codes
X = X[M["features"]]
p = np.mean([M["models"]["xgb"].predict_proba(X)[:, 1], M["models"]["lgbm"].predict_proba(X)[:, 1],
             M["models"]["logit"].predict_proba(np.nan_to_num(lg(X[M["pcols"]].values.astype(float))))[:, 1]], axis=0)
p = M["cal"].predict_proba(lg(p)[:, None])[:, 1]
pd.DataFrame({"p_winner": np.where(flip, 1 - p, p)}, index=df.index).to_parquet(OUT)
print(f"{TOUR}: {len(df)} matches from {FROM}, mean p_winner {np.where(flip, 1 - p, p).mean():.3f} -> {OUT}")
