import sys

import numpy as np
import pandas as pd

from backtest import daily_ic, metrics, simulate, top_k_weights, walk_forward
from features import load_panel

D = sys.argv[1]
TEST_START, COST = "2023-01-01", 0.0015
P = load_panel(D, "1d")
close = P["close"]
rets = (close.ffill().shift(-1) / close.ffill() - 1).where(close.notna())
X = pd.read_parquet(f"{D}/X_1d_h1.parquet")
feats = [c for c in X.columns if c not in ("fwd", "fwd_xr")]
pred = walk_forward(X, feats, "fwd_xr", TEST_START, retrain_every=90, purge=np.timedelta64(2, "D"), model="xgb")
ic = daily_ic(pred, X["fwd"])
print(f"IC mean {ic.mean():.4f}  t-stat {ic.mean() / ic.std() * np.sqrt(len(ic)):.1f}  days IC>0 {(ic > 0).mean():.1%}")
R = rets.reindex(pred.index.get_level_values(0).unique().sort_values())
elig = X["ret_1"].unstack("symbol").notna().reindex(R.index)
rows = {}


def rec(name, sim):
    sim = sim.iloc[:-1]
    m = metrics(sim["ret"])
    m["turnover/day"] = sim["turnover"].mean()
    rows[name] = m


rec("BTC buy&hold", pd.DataFrame({"ret": R["BTCUSDT"], "turnover": 0.0}))
ew = elig.astype(float).div(elig.sum(axis=1), axis=0)
rec("EqualWeight top100", simulate(ew.reindex(columns=R.columns, fill_value=0), R, 1, COST))
for k in [10, 20]:
    w = top_k_weights(pred, k).reindex(columns=R.columns, fill_value=0)
    for reb in [1, 7]:
        rec(f"XGB long top{k} rebal{reb}d", simulate(w, R, reb, COST))
        rec(f"XGB long top{k} rebal{reb}d NO FEES", simulate(w, R, reb, 0.0))
    wls = top_k_weights(pred, k, long_short=True).reindex(columns=R.columns, fill_value=0)
    rec(f"XGB long/short {k}x{k} rebal1d", simulate(wls, R, 1, COST))
    rec(f"XGB long/short {k}x{k} rebal1d NO FEES", simulate(wls, R, 1, 0.0))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).T.round(3).to_string())
pred.to_frame().to_parquet(f"{D}/pred_quick_xgb.parquet")
