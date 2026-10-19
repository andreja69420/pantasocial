"""Walk-forward ML on the daily cross-section, net of fees, versus simple benchmarks."""
import json
import sys

import numpy as np
import pandas as pd

from backtest import daily_ic, metrics, simulate, top_k_weights, walk_forward
from features import load_panel

D = sys.argv[1]
MODELS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["xgb", "lgbm", "ridge"]
TAG = "_".join(MODELS)
TEST_START = "2023-01-01"
COST = 0.0015  # 0.10% Binance taker fee + ~0.05% slippage, per side

P = load_panel(D, "1d")
close = P["close"]
rets = (close.ffill().shift(-1) / close.ffill() - 1).where(close.notna())  # t -> t+1, aligned at t
btc = rets["BTCUSDT"]

results, curves, ics = {}, {}, {}


def record(name, sim):
    sim = sim[sim.index >= TEST_START].iloc[:-1]
    m = metrics(sim["ret"])
    m["avg_daily_turnover"] = sim["turnover"].mean()
    results[name] = m
    curves[name] = (1 + sim["ret"]).cumprod()


bh = pd.DataFrame({"ret": btc, "turnover": 0.0})
record("BTC buy&hold", bh)

for h in [1, 7]:
    X = pd.read_parquet(f"{D}/X_1d_h{h}.parquet")
    feats = [c for c in X.columns if c not in ("fwd", "fwd_xr")]
    elig = X["ret_1"].unstack("symbol").notna()
    if h == 1:
        ew = elig.astype(float).div(elig.sum(axis=1), axis=0)
        record("EqualWeight top100 (daily rebal)", simulate(ew, rets.reindex(ew.index), 1, COST))
        mom = X["ret_30"].rename("pred")
        record("Momentum 30d top10 (no ML, weekly)", simulate(top_k_weights(mom, 10), rets.reindex(elig.index), 7, COST))
        rev = (-X["ret_1"]).rename("pred")
        record("Reversal 1d top10 (no ML, daily)", simulate(top_k_weights(rev, 10), rets.reindex(elig.index), 1, COST))

    for model in MODELS:
        pred = walk_forward(X, feats, "fwd_xr", TEST_START, retrain_every=30,
                            purge=np.timedelta64(h + 1, "D"), model=model)
        pred.to_frame().to_parquet(f"{D}/pred_{model}_h{h}.parquet")
        ic = daily_ic(pred, X["fwd"])
        ics[f"{model}_h{h}"] = {"mean_IC": ic.mean(), "IC_tstat": ic.mean() / ic.std() * np.sqrt(len(ic) / h),
                                "IC_pos_frac": (ic > 0).mean()}
        R = rets.reindex(pred.index.get_level_values(0).unique().sort_values())
        for k in [10, 20]:
            w = top_k_weights(pred, k).reindex(columns=R.columns, fill_value=0)
            record(f"ML-{model} h{h} long top{k}", simulate(w, R, h, COST))
            record(f"ML-{model} h{h} long top{k} ZERO COST", simulate(w, R, h, 0.0))
            wls = top_k_weights(pred, k, long_short=True).reindex(columns=R.columns, fill_value=0)
            record(f"ML-{model} h{h} long/short {k}x{k}", simulate(wls, R, h, COST))
        print(model, h, ics[f"{model}_h{h}"], flush=True)

pd.DataFrame(results).T.to_csv(f"{D}/results_daily_{TAG}.csv")
pd.DataFrame(curves).to_parquet(f"{D}/curves_daily_{TAG}.parquet")
json.dump(ics, open(f"{D}/ics_daily_{TAG}.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print(pd.DataFrame(results).T.round(3).to_string())
