"""2026 from January with a 100,000 dinar bank and quarter-Kelly stakes that compound (each day's stakes are a share of
the bank at the start of that day), at about one to two bets a day, plan segments only (women's ITF, Challenger,
men's ITF), opening prices.
  Jan 1 - May 13: BetMGM.us opening prices (blend_transfer.py rows, full model retrained with lr x0.33)
  Jun 8 - Oct 11: Flashscore UK books, best opening price across books (eval_fs2026.py rows, results-only model)
Also with the void stress (bets whose fair probability later rose >12 points refunded).

usage: kelly_2026.py DATA_DIR
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1]
BANK, K = 100_000.0, 0.25

os.environ.setdefault("PREDTAG", "r6_lr02")
src = open(os.path.join(HERE, "blend_transfer.py")).read()
_argv = sys.argv
exec(src.split("def rep(")[0].split('"""', 2)[2])  # T: BetMGM 2025-26 sides with qb, evb, pnl, pv2, o, day
sys.argv = _argv
A = T[T.year == 2026].assign(src="BetMGM", day=lambda x: pd.to_datetime(x.day))
A = A.rename(columns={"evb": "ev_b", "pv2": "pnl_void"})[["src", "seg", "day", "o", "qb", "ev_b", "pnl", "pnl_void"]]

src = open(os.path.join(HERE, "eval_fs2026.py")).read()
sys.argv = [sys.argv[0], D]
exec(src.split("def rep(")[0].split('"""', 1)[1].split('"""', 1)[1])  # S: Flashscore UK sides per book
S = S[S.seg.isin(["Women ITF", "Challenger", "Men ITF"])]
Bst = S.sort_values("o", ascending=False).drop_duplicates(["mi", "won"])
Bst = Bst.assign(src="Flashscore UK best", day=pd.to_datetime(Bst.date).dt.normalize())
Bst = Bst.rename(columns={"evb": "ev_b", "pv": "pnl_void"})[["src", "seg", "day", "o", "qb", "ev_b", "pnl", "pnl_void"]]
ALL = pd.concat([A, Bst], ignore_index=True).sort_values("day", kind="stable")
days_a = (A.day.max() - A.day.min()).days + 1
days_b = (Bst.day.max() - Bst.day.min()).days + 1
print(f"BetMGM {A.day.min():%Y-%m-%d}..{A.day.max():%Y-%m-%d} ({days_a} days), "
      f"Flashscore UK {Bst.day.min():%Y-%m-%d}..{Bst.day.max():%Y-%m-%d} ({days_b} days)")


def simulate(b, col, cap=None):
    bank, peak, dd, path = BANK, BANK, 0.0, []
    for day, g in b.groupby("day", sort=True):
        f = K * np.maximum(0.0, (g.qb.values * g.o.values - 1) / (g.o.values - 1))
        stakes = bank * f if cap is None else np.minimum(bank * f, cap)
        bank += float((stakes * g[col].values).sum())
        peak = max(peak, bank)
        dd = max(dd, 1 - bank / peak)
        path.append((day, bank, stakes.sum()))
    return bank, dd, pd.DataFrame(path, columns=["day", "bank", "staked"])


# about one (or two) bets a day: each day the bets with the highest blend EV among EV>5%, odds<=5
base = ALL[(ALL.ev_b > 0.05) & (ALL.o <= 5)].sort_values(["day", "ev_b"], ascending=[True, False])
CHOICES = {"best 1 a day (EV>5%, odds<=5)": base.groupby("day").head(1), "best 2 a day (EV>5%, odds<=5)": base.groupby("day").head(2),
           "best 1 a day (EV>5%, odds<=2)": ALL[(ALL.ev_b > 0.05) & (ALL.o <= 2)].sort_values(["day", "ev_b"], ascending=[True, False])
           .groupby("day").head(1)}
for name, b in CHOICES.items():
    per_day = len(b) / (days_a + days_b)
    print(f"\n== {name}: {len(b)} bets ({per_day:.1f} per calendar day), flat ROI {b.pnl.mean():+.1%}, void-stress ROI "
          f"{b.pnl_void.mean():+.1%}, win rate {(b.pnl > 0).mean():.0%}, mean odds {b.o.mean():.2f}, "
          f"mean stake {100 * K * np.mean(np.maximum(0, (b.qb * b.o - 1) / (b.o - 1))):.1f}% of bank")
    for col, lab, cap in (("pnl", "as backtested", None), ("pnl_void", "void stress", None),
                          ("pnl", "max 10,000/bet", 10_000), ("pnl_void", "void + max 10k", 10_000)):
        fin, dd, path = simulate(b, col, cap)
        m = path.set_index("day").bank.resample("ME").last().ffill()
        print(f"  {lab:15s} 100,000 -> {fin:>12,.0f} din ({fin / BANK - 1:+6.0%}), max drawdown {dd:3.0%} | month-end: "
              + " ".join(f"{d:%b} {v / 1000:,.0f}k" for d, v in m.items()))
    for s_, g in b.groupby("src"):
        print(f"  {s_:20s} {len(g):4d} bets, flat ROI {g.pnl.mean():+.1%}, void {g.pnl_void.mean():+.1%}")
