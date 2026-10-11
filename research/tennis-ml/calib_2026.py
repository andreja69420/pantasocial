"""Calibration of the production model (r6_v3) on 2026, which it never trained on: matches grouped by predicted win
probability, predicted mean vs. how often that player actually won. Both sides of every match are counted.
  1. model alone (no odds), every 2026 match, per tour
  2. matches with Serbian book prices, no ITF: model, opening Shin of the best-price book, betting P (blend)
  3. the sides we would bet (blend EV>X% at the best opening price, odds<=5): mean P vs. actually won

usage: calib_2026.py DATA_DIR BETS_PARQUET OUT_PNG   (BETS_PARQUET from eval_levels2026.py --bets=..., e.g. six RS books)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D, BETS, OUT = sys.argv[1:4]
ITF = {"15", "25", "35", "50", "75", "100"}


def shin(a, b):
    lo, hi = np.zeros_like(a), np.full_like(a, 0.5)
    B = a + b
    f = lambda z, x: (np.sqrt(z ** 2 + 4 * (1 - z) * x ** 2 / B) - z) / (2 * (1 - z))
    for _ in range(50):
        z = (lo + hi) / 2
        s = f(z, a) + f(z, b)
        hi, lo = np.where(s < 1, z, hi), np.where(s < 1, lo, z)
    return f((lo + hi) / 2, a)


def table(p, y, edges, nmin=30):
    b = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    t = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(n=("y", "size"), pred=("p", "mean"), act=("y", "mean"))
    t = t[t.n >= nmin]
    z = 1.96
    den = 1 + z ** 2 / t.n
    mid = (t.act + z ** 2 / (2 * t.n)) / den
    half = z * np.sqrt(t.act * (1 - t.act) / t.n + z ** 2 / (4 * t.n ** 2)) / den
    return t.assign(lo=mid - half, hi=mid + half)


E20 = np.linspace(0, 1, 21)


def scores(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    t = table(p, y, E20)
    ece = float((t.n * (t.pred - t.act).abs()).sum() / t.n.sum())
    return f"Brier {np.mean((p - y) ** 2):.4f}, log-loss {-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)):.4f}, ECE {ece:.2%}"


def line(ax, p, y, label, edges=E20, nmin=30, **kw):
    t = table(p, y, edges, nmin)
    ax.errorbar(t.pred, t.act, yerr=[t.act - t.lo, t.hi - t.act], marker="o", ms=4, capsize=2, label=label, **kw)
    return t


def show(name, t):
    print(f"  {name}")
    print("    predicted  actual  (95% CI)        n")
    for r in t.itertuples():
        print(f"    {r.pred:8.1%} {r.act:7.1%}  ({r.lo:5.1%}-{r.hi:5.1%}) {r.n:8,d}")


fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
for ax in axes:
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="savršeno")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.grid(alpha=0.3)
    ax.set_xlabel("predviđena verovatnoća pobede"), ax.set_ylabel("stvarno pobedio (udeo)")

print("1. Model alone (no odds), every 2026 match")
ax = axes[0]
for tour, c in (("wta", "tab:red"), ("atp", "tab:blue")):
    P = pd.read_parquet(f"{D}/preds3_{tour}_ens_r6_v3_lower_2026.parquet")
    P = P[P.stands]
    p, y = np.r_[P.p.values, 1 - P.p.values], np.r_[P.y.values, 1 - P.y.values]
    show(f"{tour.upper()} {len(P):,} matches: {scores(p, y)}", line(ax, p, y, f"{tour.upper()} ({len(P):,} mečeva)", color=c))
ax.set_title("1. Model sam (bez kvota), svi mečevi 2026")
ax.legend(loc="upper left")

S = pd.read_parquet(BETS)
S = S[(S.pnl != 0) & ~S.lvl.astype(str).isin(ITF)].copy()
S["won"] = S.won.astype(int)
S["shin"] = shin(1 / S.o.values, 1 / S.oo.values)
S["qb"] = (S.evb + 1) / S.o
n = S.mi.nunique()
print(f"\n2. Serbian book prices, no ITF, {n:,} matches")
ax = axes[1]
for col, lab, c in (("q", "model (bez kvota)", "tab:green"), ("shin", "kladionica (fer početna kvota)", "tab:gray"),
                    ("qb", "P za klađenje (model + kladionica)", "tab:purple")):
    show(f"{lab}: {scores(S[col].values, S.won.values)}", line(ax, S[col].values, S.won.values, lab, color=c))
ax.set_title(f"2. Challenger, WTA 125 i turneje, srpske kladionice\n{n:,} mečeva, bez ITF")
ax.legend(loc="upper left")

print("\n3. Sides we would bet (best opening price, odds<=5)")
ax = axes[2]
E10 = np.linspace(0, 1, 11)
sub = []
for thr, c in ((0.05, "tab:orange"), (0.10, "tab:brown"), (0.15, "tab:pink")):
    b = S[(S.evb > thr) & (S.o <= 5)]
    t = line(ax, b.qb.values, b.won.values, f"EV>{thr:.0%} ({len(b):,}): P {b.qb.mean():.1%}, pobedile {b.won.mean():.1%}",
             edges=E10, nmin=20, color=c)
    show(f"EV>{thr:.0%} ({len(b)}): mean P {b.qb.mean():.1%}, won {b.won.mean():.1%}, bookmaker fair {b.shin.mean():.1%}, "
         f"model {b.q.mean():.1%}", t)
    sub.append(f"EV>{thr:.0%}: P {b.qb.mean():.1%} / pobedile {b.won.mean():.1%} / kladionica {b.shin.mean():.1%}")
ax.set_title("3. Strane na koje igramo (najbolja početna kvota, kvota≤5)\n" + sub[0])
ax.legend(loc="upper left", fontsize=9)

fig.suptitle("Kalibracija novog modela na 2026: kad kaže X%, koliko stvarno pobeđuje (linije = 95% interval)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print(f"\n-> {OUT}")
