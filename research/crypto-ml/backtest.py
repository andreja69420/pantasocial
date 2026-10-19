import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge


def walk_forward(X, feats, label, test_start, retrain_every, purge, model="lgbm", train_window=None, seed=0):
    dates = X.index.get_level_values("date")
    udates = np.sort(dates.unique())
    test_dates = udates[udates >= np.datetime64(test_start)]
    preds = []
    for i in range(0, len(test_dates), retrain_every):
        block = test_dates[i:i + retrain_every]
        # Labels of the last `purge` training rows overlap the test block; drop them (no lookahead).
        cutoff = block[0] - purge
        tr = (dates < cutoff)
        if train_window is not None:
            tr &= dates >= cutoff - train_window
        tr &= X[label].notna().values
        te = dates.isin(block)
        Xtr, ytr = X.loc[tr, feats], X.loc[tr, label]
        if model == "lgbm":
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=500,
                                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                                  random_state=seed, verbose=-1, n_jobs=4)
            m.fit(Xtr, ytr)
            p = m.predict(X.loc[te, feats])
        elif model == "xgb":
            m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=200,
                                 subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist",
                                 device=os.environ.get("XGB_DEVICE", "cpu"), random_state=seed, n_jobs=4)
            m.fit(Xtr, ytr)
            p = m.predict(X.loc[te, feats])
        else:
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
            m = Ridge(alpha=100.0).fit(((Xtr - mu) / sd).fillna(0), ytr)
            p = m.predict(((X.loc[te, feats] - mu) / sd).fillna(0))
        preds.append(pd.Series(p, index=X.index[te]))
    return pd.concat(preds).rename("pred")


def daily_ic(pred, fwd_long):
    df = pd.concat([pred, fwd_long.rename("fwd")], axis=1).dropna()
    return df.groupby(level="date").apply(lambda g: g["pred"].corr(g["fwd"], method="spearman"))


def simulate(target, rets, rebalance=1, cost=0.0015):
    """target: DataFrame date x symbol of desired weights (only rows where we rebalance matter).
    rets: DataFrame date x symbol of close-to-close returns for bar t -> t+1 (aligned at t)."""
    dates = target.index
    cols = target.columns
    w = pd.Series(0.0, index=cols)
    out, turn = [], []
    for i, t in enumerate(dates):
        if i % rebalance == 0:
            tgt = target.loc[t].fillna(0.0)
            tr = (tgt - w).abs().sum()
            w = tgt
        else:
            tr = 0.0
        r = rets.loc[t, cols].fillna(0.0)
        gross = float((w * r).sum())
        net = gross - tr * cost
        out.append(net)
        turn.append(tr)
        # Long and short books drift with their own prices; renormalise against the NAV.
        w = w * (1 + r) / (1 + gross) if (1 + gross) != 0 else w
    return pd.DataFrame({"ret": out, "turnover": turn}, index=dates)


def top_k_weights(pred, k, long_short=False, eligible=None):
    p = pred.unstack("symbol")
    rk = p.rank(axis=1, ascending=False)
    n = p.notna().sum(axis=1)
    long = (rk <= k).astype(float).div(np.minimum(k, n), axis=0)
    if not long_short:
        return long
    short = (rk > (n - k).values[:, None]).astype(float).div(np.minimum(k, n), axis=0)
    return 0.5 * long - 0.5 * short


def metrics(r, periods=365):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / periods
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    sharpe = r.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"total_return": eq.iloc[-1] - 1, "CAGR": cagr, "Sharpe": sharpe, "MaxDD": dd,
            "vol_ann": r.std() * np.sqrt(periods)}
