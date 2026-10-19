"""Pre-registered men's ITF 2024-2026 test at BetExplorer average odds (rules in README, written before the run).

usage: eval_bx.py DATA_DIR
"""
import sys

import numpy as np
import pandas as pd

import tactics

tactics.D = sys.argv[1]
S = tactics.load("atp", ["preds3_atp_ens_r6_hist1991_ens_lower_2024-2025-2026.parquet"], r"^(\d|S)")
n = len(S) // 2
p_mkt = (1 / S.o[:n]) / (1 / S.o[:n] + 1 / S.oo[:n])
y = np.ones(n)  # first half of S = bets on the winner
ll = lambda p: -np.mean(np.log(np.clip(p, 1e-6, 1)))
print(f"men's ITF {S.year.min()}-{S.year.max()}: {n} matches with BetExplorer odds, median margin "
      f"{np.median(1 / S.o[:n] + 1 / S.oo[:n] - 1):.1%}; log-loss market {ll(p_mkt.values):.4f} model {ll(S.q.values[:n]):.4f}")
core = (S.ev > 0.05) & (S.o <= 2.0)
T = {
    "PRIMARY core EV>5% odds<=2": core,
    "EV>10% odds<=2": (S.ev > 0.10) & (S.o <= 2.0),
    "EV>20% odds<=2": (S.ev > 0.20) & (S.o <= 2.0),
    "EV>5% odds 1.6-2.0": (S.ev > 0.05) & (S.o >= 1.6) & (S.o < 2.0),
    "core & opp retires often": core & (S.op_ret_365d > 0.05),
    "core & opp played yesterday": core & (S.op_rest_days <= 1),
    "core & me rising elo": core & (S.me_welo_mom10 > 0),
    "core & opp 3+ matches in 7d": core & (S.op_n_7d >= 3),
    "core & me lefty": core & (S.me_left == 1),
}
rng = np.random.default_rng(0)
for name, m in T.items():
    b = S[m.fillna(False).values]
    if len(b) == 0:
        continue
    boot = np.array([b.pnl.values[rng.integers(0, len(b), len(b))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [5, 95])
    pool = S.pnl[(S.o <= 2.0).values].values
    rnd = np.mean([rng.choice(pool, len(b), replace=False).mean() for _ in range(300)])
    by = " ".join(f"{k}:{v:+.1%}({c})" for k, v, c in zip(*[b.groupby("year").pnl.mean().index, b.groupby("year").pnl.mean().values,
                                                         b.groupby("year").size().values]))
    print(f"  {name:30s} bets {len(b):5d} ROI {b.pnl.mean():+6.2%} 90%CI [{lo:+.1%},{hi:+.1%}] random(odds<=2) {rnd:+.1%} | {by}")
