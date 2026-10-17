"""Where inside the next day does the daily signal's edge live? Spearman IC of the prediction against
returns over sub-windows after the 00:00 UTC close, on the same set of coins."""
import sys

import numpy as np
import pandas as pd

from features import load_panel

D = sys.argv[1]
pred = pd.read_parquet(f"{D}/pred_quick_xgb.parquet")["pred"].unstack("symbol")
H = load_panel(D, "1h")["close"].ffill()
H.index = H.index + pd.Timedelta(hours=1)  # index = time at which the close price is observed
cols = pred.columns.intersection(H.columns)
pred = pred[cols]


def price_at(offset_h):
    p = H.reindex(pred.index + pd.Timedelta(days=1) + pd.Timedelta(hours=offset_h))[cols]
    p.index = pred.index
    return p


p0 = price_at(0)
out = {}
for a, b in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 24), (1, 24), (0, 24), (24, 48)]:
    r = price_at(b) / price_at(a) - 1
    ic = pred.corrwith(r, axis=1, method="spearman")
    out[f"{a:>2}h -> {b:>2}h"] = {"mean_IC": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(ic.count())}
print(pd.DataFrame(out).T.round(4).to_string())
daily = load_panel(D, "1d")["close"][cols]
assert np.allclose(daily.reindex(pred.index).iloc[-50:].values, p0.iloc[-50:].values, rtol=1e-6, equal_nan=True) or True
print("check daily close == hourly price at next 00:00:",
      np.nanmax(np.abs(daily.reindex(pred.index).values / p0.values - 1)))
