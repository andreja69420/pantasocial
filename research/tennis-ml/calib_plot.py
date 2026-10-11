"""Calibration (reliability) chart on test data the model never trained on: matches grouped by predicted win
probability (bins of 5 points), predicted mean vs. how often that player actually won. A perfect model lies on the diagonal.
Both sides of every match are counted (p and 1-p), so the chart is symmetric around 0.5.
  1. model alone (no odds), walk-forward test 2014-2025: each year predicted by a model trained only on earlier years
  2. model alone, fresh lower levels 2025-26 (women's ITF, Challenger, men's ITF) vs the market's opening Shin probability
  3. betting probability P (blend of model and opening Shin, BetMGM 2025-26) on all sides and on the sides we would bet

usage: calib_plot.py DATA_DIR OUT_PNG
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D, OUT = sys.argv[1], sys.argv[2]
EDGES = np.linspace(0, 1, 21)


def table(p, y):
    b = np.clip(np.digitize(p, EDGES) - 1, 0, len(EDGES) - 2)
    t = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(n=("y", "size"), pred=("p", "mean"), act=("y", "mean"))
    t = t[t.n >= 30]
    z = 1.96
    den = 1 + z ** 2 / t.n
    mid = (t.act + z ** 2 / (2 * t.n)) / den
    half = z * np.sqrt(t.act * (1 - t.act) / t.n + z ** 2 / (4 * t.n ** 2)) / den
    return t.assign(lo=mid - half, hi=mid + half)


def scores(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    t = table(p, y)
    ece = float((t.n * (t.pred - t.act).abs()).sum() / t.n.sum())
    return f"Brier {np.mean((p - y) ** 2):.4f}, log-loss {-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)):.4f}, ECE {ece:.2%}"


def both(p, y):
    return np.r_[p, 1 - p], np.r_[y, 1 - y]


def line(ax, p, y, label, **kw):
    t = table(p, y)
    ax.errorbar(t.pred, t.act, yerr=[t.act - t.lo, t.hi - t.act], marker="o", ms=4, capsize=2, label=label, **kw)
    return t


def show(name, t):
    print(f"  {name}")
    print("    predicted  actual  (95% CI)        n")
    for r in t.itertuples():
        print(f"    {r.pred:8.1%} {r.act:7.1%}  ({r.lo:5.1%}-{r.hi:5.1%}) {r.n:8,d}")


fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), gridspec_kw={"height_ratios": [3, 1]})
for ax in axes[0]:
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="savršeno (predviđeno = stvarno)")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.grid(alpha=0.3)
    ax.set_xlabel("predviđena verovatnoća pobede"), ax.set_ylabel("stvarno pobedio (udeo)")

# 1. walk-forward test 2014-2025
te = "2014-2015-2016-2017-2018-2019-2020-2021-2022-2023-2024-2025"
ax, axh = axes[0, 0], axes[1, 0]
print("1. Model alone (no odds), walk-forward test 2014-2025")
for tour, c in (("wta", "tab:red"), ("atp", "tab:blue")):
    P = pd.read_parquet(f"{D}/out/preds3_{tour}_ens_r6_hist1991_ens_te_{te}.parquet")
    P = P[P.stands]
    p, y = both(P.p.values, P.y.values)
    show(f"{tour.upper()} {len(P):,} matches: {scores(p, y)}", line(ax, p, y, f"{tour.upper()} ({len(P):,} mečeva)", color=c))
    axh.hist(p, bins=EDGES, alpha=0.5, color=c, label=tour.upper())
ax.set_title("1. Model (bez kvota), test 2014–2025\nsvaka godina predviđena modelom treniranim samo na ranijim godinama")
ax.legend(loc="upper left")

# 2/3. fresh lower levels 2025-26 with BetMGM opening prices: model, market, blend
os.environ.setdefault("PREDTAG", "r6_lr02")
src = open(os.path.join(HERE, "blend_transfer.py")).read()
_argv = sys.argv
exec(src.split("def rep(")[0].split('"""', 2)[2])  # T: BetMGM 2025-26 sides with q, shin, qb, evb, pnl
sys.argv = _argv
T = T[T.pnl != 0].assign(won=lambda x: (x.pnl > 0).astype(int))  # finished matches only; both sides are already rows

print(f"\n2. Fresh lower levels 2025-26 (BetMGM opening prices), {len(T) // 2:,} matches")
ax = axes[0, 1]
show(f"model alone: {scores(T.q.values, T.won.values)}", line(ax, T.q.values, T.won.values, "model (bez kvota)", color="tab:green"))
show(f"market opening Shin: {scores(T.shin.values, T.won.values)}",
     line(ax, T.shin.values, T.won.values, "kladionica (fer početna kvota)", color="tab:gray"))
axes[1, 1].hist(T.q, bins=EDGES, alpha=0.5, color="tab:green", label="model")
axes[1, 1].hist(T.shin, bins=EDGES, alpha=0.5, color="tab:gray", label="kladionica")
ax.set_title(f"2. Niži nivoi 2025–26 (ženski ITF, Challenger, muški ITF)\nmodel naspram kladionice, {len(T) // 2:,} mečeva")
ax.legend(loc="upper left")

print("\n3. Betting probability P = blend of model and opening Shin")
ax = axes[0, 2]
show(f"all sides: {scores(T.qb.values, T.won.values)}", line(ax, T.qb.values, T.won.values, "P, sve strane", color="tab:purple"))
bets = T[(T.evb > 0.05) & (T.o <= 5)]
tb = line(ax, bets.qb.values, bets.won.values, f"P, strane koje igramo (EV>5%, kvota≤5, {len(bets):,})", color="tab:orange")
show(f"bets EV>5% odds<=5 ({len(bets)}): mean P {bets.qb.mean():.1%}, won {bets.won.mean():.1%}, "
     f"bookmaker fair {bets.shin.mean():.1%}, model {bets.q.mean():.1%}", tb)
axes[1, 2].hist(T.qb, bins=EDGES, alpha=0.5, color="tab:purple", label="sve strane")
axes[1, 2].hist(bets.qb, bins=EDGES, alpha=0.8, color="tab:orange", label="igramo")
ax.set_title(f"3. P za klađenje (model + kladionica), BetMGM 2025–26\n"
             f"igrane strane: P {bets.qb.mean():.1%}, pobedile {bets.won.mean():.1%}, kladionica {bets.shin.mean():.1%}")
ax.legend(loc="upper left")

for a in axes[1]:
    a.set_xlim(0, 1), a.set_xlabel("predviđena verovatnoća"), a.set_ylabel("broj"), a.legend(), a.grid(alpha=0.3)
fig.suptitle("Kalibracija na test podacima: kad model kaže X%, koliko stvarno pobeđuje (linije = 95% interval)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print(f"\n-> {OUT}")
