"""One model only: every paper bet logged by an earlier model is priced again by the current production model at the
same opening price (the bet's entry odds, with the Shin price of that book's opening pair backed out of the logged
blend). Bets the current model would also have taken (blend EV above the bet's rule threshold) stay, with the current
model's probabilities (so the Excel stake = quarter Kelly of its estimate) and model = MODEL_TAG; the others are deleted.
Pending matches: the probability from the last loop run (q_now). Settled matches: their rows in the feature build,
whose features use earlier matches only, plus the new-feature modules (fx_live.py) on the table up to that match.
The ledger before the change is kept as LEDGER.bak-<time>.

usage: rescore_ledger.py DATA_DIR   (env LIVE_OUT, MODELS, MODEL_TAG as in paper_loop.sh)
"""
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

import fx_live
from design import build_X, load_f2

D = sys.argv[1]
LIVE_OUT, MODELS, TAG = os.environ.get("LIVE_OUT", "out_live2"), os.environ.get("MODELS", "models_v3"), os.environ.get("MODEL_TAG", "novi")
LEDGER = os.path.join(D, "paper", "ledger.csv")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
sg = lambda x: 1 / (1 + np.exp(-x))

L = pd.read_csv(LEDGER)
old = L.model.fillna("") != TAG
q = pd.Series(np.nan, index=L.index)
pending = old & (L.result.fillna("") == "")
q[pending] = L.q_now[pending]
settled = old & L.result.isin(["home", "away"])
for tour in ("wta", "atp"):
    sel = settled & (L.tour == tour)
    if not sel.any():
        continue
    base = fx_live.load_base(os.path.join(D, LIVE_OUT), tour)
    allm = pd.read_parquet(os.path.join(D, LIVE_OUT, f"{tour}_matches.parquet"), columns=["fs_mid", "winner_name", "loser_name"])
    rows = allm[allm.fs_mid.isin(L.fs_mid[sel])]
    # the modules order the rows by `order` and a row's features use earlier rows only: the whole table gives each of
    # these matches its features as they stood before it was played
    nx = fx_live.features(base.drop(rows.index), base.loc[rows.index], tour == "wta")
    F = load_f2(os.path.join(D, LIVE_OUT, f"{tour}_f2.parquet"), start=str(base.loc[rows.index, "date"].min().date()))
    Fx = F.loc[rows.index].join(nx)
    M = pickle.load(open(os.path.join(D, MODELS, f"{tour}_final.pkl"), "rb"))
    X, _, flip = build_X(Fx, tour)
    X["level"] = pd.Categorical(Fx.tourney_level.astype(str), categories=M["levels"]).codes
    X = X[M["features"]]
    p = np.mean([M["models"]["xgb"].predict_proba(X)[:, 1], M["models"]["lgbm"].predict_proba(X)[:, 1],
                 M["models"]["logit"].predict_proba(np.nan_to_num(lg(X[M["pcols"]].values.astype(float))))[:, 1]], axis=0)
    p = M["cal"].predict_proba(lg(p)[:, None])[:, 1]
    p_win = pd.Series(np.where(flip, 1 - p, p), index=rows.fs_mid.values)  # P(the match's actual winner)
    for i in L.index[sel]:
        if L.at[i, "fs_mid"] in p_win.index:
            pw = p_win[L.at[i, "fs_mid"]]
            q[i] = pw if L.at[i, "side"] == L.at[i, "result"] else 1 - pw
    print(f"{tour}: {len(rows)} settled matches priced again", flush=True)

shin_lo = (lg(L.q_blend) - 0.44 * lg(L.q_model)) / 0.64
qb = sg(0.44 * lg(q) + 0.64 * shin_lo)
ev = qb * L.open - 1
thr = L.rule.str.extract(r"EV>(\d+)%").astype(float)[0] / 100
keep = ~old | (q.notna() & (ev > thr))
assert q[old].notna().all(), L[old & q.isna()][["fs_mid", "home", "away"]]
show = L[old].assign(q_old=L.q_model, q_new=q, ev_old=L.ev, ev_new=ev, keep=keep)
print(show[["home", "away", "side", "rule", "open", "result", "q_old", "q_new", "ev_old", "ev_new", "keep"]].round(3).to_string())
shutil.copy(LEDGER, LEDGER + ".bak-" + pd.Timestamp.now("UTC").strftime("%Y%m%d%H%M"))
L.loc[old, "q_model"], L.loc[old, "q_blend"], L.loc[old, "ev"], L.loc[old, "model"] = q[old], qb[old], ev[old], TAG
L = L[keep]
L.to_csv(LEDGER, index=False)
print(f"{int(old.sum())} bets of earlier models: {int((old & keep).sum())} kept with the current model's prices, "
      f"{int((old & ~keep).sum())} deleted -> {LEDGER}")
