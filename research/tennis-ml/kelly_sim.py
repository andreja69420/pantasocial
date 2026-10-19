"""Stake sizing on the fresh BetMGM 2025-26 picks (blend fitted on Bet365 2014-2024, see blend_transfer.py):
flat stakes vs fractional Kelly f = k * (p*o - 1) / (o - 1) with p = blend probability, capped per bet.
Bets in date order. Reports final bankroll (order-independent for these rules), max drawdown, and over 200 reshuffled
bet orders the drawdown distribution and the chance of ever falling below half the starting bank.

usage: kelly_sim.py DATA_DIR
"""
import os
import sys

import numpy as np
import pandas as pd

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blend_transfer.py")).read()
exec(src.split("def rep(")[0].split('"""', 2)[2])  # builds T (BetMGM 2025-26 sides with qb, evb, pnl, o)
rng = np.random.default_rng(0)


def run(b, stake_fn, order=None):
    bank, peak, dd, low = 1.0, 1.0, 0.0, 1.0
    for i in (order if order is not None else range(len(b))):
        s = stake_fn(b, i, bank)
        bank += s * b.pnl.values[i]
        peak = max(peak, bank)
        dd, low = max(dd, 1 - bank / peak), min(low, bank)
    return bank, dd, low


for name, sel in (("blend EV>5% odds<=2", (T.evb > 0.05) & (T.o <= 2)), ("blend EV>5% odds<=5", (T.evb > 0.05) & (T.o <= 5))):
    b = T[sel].sort_values("day", kind="stable").reset_index(drop=True)
    print(f"\n{name}: {len(b)} bets, ROI {b.pnl.mean():+.2%}, mean odds {b.o.mean():.2f}")
    kel = lambda b, i: max(0.0, (b.qb.values[i] * b.o.values[i] - 1) / (b.o.values[i] - 1))
    rules = {"flat 1% of start bank": lambda b, i, bank: 0.01,
             "flat 2% of start bank": lambda b, i, bank: 0.02,
             "Kelly x0.10 (cap 2%)": lambda b, i, bank: bank * min(0.10 * kel(b, i), 0.02),
             "Kelly x0.25 (cap 3%)": lambda b, i, bank: bank * min(0.25 * kel(b, i), 0.03),
             "Kelly x0.50 (cap 5%)": lambda b, i, bank: bank * min(0.50 * kel(b, i), 0.05)}
    for rn, fn in rules.items():
        fin, dd, low = run(b, fn)
        sims = [run(b, fn, rng.permutation(len(b))) for _ in range(200)]
        dds = np.array([s[1] for s in sims])
        lows = np.array([s[2] for s in sims])
        print(f"  {rn:24s} final x{fin:7.2f}  maxDD (by date) {dd:5.1%}  | 200 reshuffled orders: maxDD median {np.median(dds):5.1%}, "
              f"95th pct {np.percentile(dds, 95):5.1%}, P(bank < 50% of start) {np.mean(lows < 0.5):.1%}")
