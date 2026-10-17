"""Is 'ML short bottom-10 + BTC hedge, weekly' real alpha, or just long-BTC/short-alts beta plus rebalance-day luck?"""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
src = open("run_pump_short.py").read().split("ml_score = -pu")[0]
g = {"__name__": "lib"}
exec(src.replace("D = sys.argv[1]", f"D = {D!r}"), g)
dates, pu, perp, run, short_book, rows, ret_d = g["dates"], g["pu"], g["perp"], g["run"], g["short_book"], g["rows"], g["ret_d"]
rows.clear()
rng = np.random.default_rng(0)

ml = -pu
ew = perp.astype(float).where(perp)  # all perp names equally
for off in range(7):
    sl = dates[off:]
    run(f"ML bottom10 weekly offset{off}", short_book(ml.loc[sl], 10, perp.loc[sl]), 7, 0.003)
run("BENCH short ALL top100 perps EW + BTC hedge, weekly", short_book(ew, 1000, perp), 7, 0.003)
for i in range(20):
    rnd = pd.DataFrame(rng.random(perp.shape), index=perp.index, columns=perp.columns)
    run(f"RANDOM10 #{i}", short_book(rnd, 10, perp), 7, 0.003)
# Signal flipped: if ML is real, shorting its *top* picks should do clearly worse than random.
run("ML TOP10 shorted (sanity, should be bad)", short_book(pu, 10, perp), 7, 0.003)

res = pd.DataFrame(rows).T
pd.set_option("display.width", 250)
cols = ["total_return", "Sharpe", "MaxDD", "2023", "2024", "2025", "2026"]
print(res.loc[[i for i in res.index if not i.startswith("RANDOM")], cols].round(3).to_string())
rnd = res.loc[[i for i in res.index if i.startswith("RANDOM")], cols]
print("\nRANDOM 10 shorts (20 draws): Sharpe mean %.2f  min %.2f  max %.2f; total_return median %.2f"
      % (rnd.Sharpe.mean(), rnd.Sharpe.min(), rnd.Sharpe.max(), rnd.total_return.median()))
res.to_csv(f"{D}/results_robust.csv")
