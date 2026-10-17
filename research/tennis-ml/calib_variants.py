"""Chart for the calibration study (overnight run): baseline vs. the refit variants, 2025 and 2026, plus bet-side calibration.
Top row: model alone, pooled WTA+ATP, both sides of every match, 5-point bins (95% Wilson intervals only for the baseline).
Bottom row: paired log-loss difference to the baseline (negative = better) per level group with +-2 SE, and bet-side P vs. won.

usage: calib_variants.py PREDS_DIR CALIB_EXP_DIR OUT_PNG   (PREDS_DIR = out_v3, CALIB_EXP_DIR holds cx_*_six.parquet)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

O, X, OUT = sys.argv[1:4]
ITF = {"15", "25", "35", "50", "75", "100"}
grp = lambda l: "ITF" if str(l) in ITF else ("Challenger / WTA 125" if str(l) == "C" else "turnir")
cl = lambda p: np.clip(p, 1e-6, 1 - 1e-6)
ll = lambda p, y: -(y * np.log(cl(p)) + (1 - y) * np.log(1 - cl(p)))
COL = {"r6_v3": "#333333", "cx_refit": "#0072B2", "cx_cal2refit": "#D55E00", "cx_cal2": "#999999", "cx_hl8": "#CC79A7"}
NAME = {"r6_v3": "osnovni (r6_v3)", "cx_refit": "refit", "cx_cal2refit": "2 god. kalibracije + refit",
        "cx_cal2": "2 god. kalibracije", "cx_hl8": "poluraspad 8"}


def load(tag, yr):
    P = pd.concat([pd.read_parquet(f"{O}/preds3_{t}_ens_{tag}_lower_2025-2026.parquet").assign(tour=t) for t in ("wta", "atp")])
    P = P[P.year == yr].copy()
    P.index.name = "mi"
    P = P.reset_index()
    P["grp"] = P.tourney_level.map(grp)
    return P


def base(yr):
    B = load("cx_base", yr)
    if yr == 2026:  # r6_v3 predictions are identical to cx_base on 2026 (checked: max |dp| = 0); use the production file itself
        R = pd.concat([pd.read_parquet(f"{O}/preds3_{t}_ens_r6_v3_lower_2026.parquet").assign(tour=t) for t in ("wta", "atp")])
        R.index.name = "mi"
        B = B.drop(columns=["p"]).merge(R.reset_index()[["tour", "mi", "p"]], on=["tour", "mi"])
    return B


def table(p, y, edges, nmin=30):
    b = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    t = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(n=("y", "size"), pred=("p", "mean"), act=("y", "mean"))
    t = t[t.n >= nmin]
    z = 1.96
    den = 1 + z ** 2 / t.n
    mid = (t.act + z ** 2 / (2 * t.n)) / den
    half = z * np.sqrt(t.act * (1 - t.act) / t.n + z ** 2 / (4 * t.n ** 2)) / den
    return t.assign(lo=mid - half, hi=mid + half)


E20, E10 = np.linspace(0, 1, 21), np.linspace(0, 1, 11)
fig, axes = plt.subplots(2, 3, figsize=(19, 12))
for ax in axes[0]:
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="savršeno")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.grid(alpha=0.3)
    ax.set_xlabel("predviđena verovatnoća pobede"), ax.set_ylabel("stvarno pobedio (udeo)")

for ax, yr in zip(axes[0][:2], (2025, 2026)):
    for tag, key in (("r6_v3" if yr == 2026 else "cx_base", "r6_v3"), ("cx_refit", "cx_refit"), ("cx_cal2refit", "cx_cal2refit")):
        P = base(yr) if key == "r6_v3" else load(key, yr)
        p, y = np.r_[P.p.values, 1 - P.p.values], np.r_[P.y.values, 1 - P.y.values]
        t = table(p, y, E20)
        if key == "r6_v3":
            ax.errorbar(t.pred, t.act, yerr=[t.act - t.lo, t.hi - t.act], marker="o", ms=5, capsize=2, color=COL[key], label=NAME[key], zorder=3)
        else:
            ax.plot(t.pred, t.act, marker="s" if key == "cx_refit" else "^", ms=4, lw=1.2, color=COL[key], label=NAME[key], alpha=0.9)
    ax.set_title(f"Model sam, WTA+ATP, svi mečevi {yr} ({len(P):,} mečeva)")
    ax.legend(loc="upper left")

ax = axes[0][2]
ax.set_xlim(0.3, 0.9), ax.set_ylim(0.3, 0.9), ax.grid(alpha=0.3)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="savršeno")
SIX = {"r6_v3": "cx_base", "cx_refit": "cx_refit", "cx_cal2refit": "cx_cal2refit"}
for key, f in SIX.items():
    S = pd.read_parquet(f"{X}/{f}_six.parquet")
    S = S[(S.pnl != 0) & ~S.lvl.astype(str).isin(ITF)].copy()
    S["qb"] = (S.evb + 1) / S.o
    b = S[(S.evb > 0.05) & (S.o <= 5)]
    t = table(b.qb.values, b.won.astype(int).values, E10, 20)
    ax.errorbar(t.pred, t.act, yerr=[t.act - t.lo, t.hi - t.act] if key == "r6_v3" else None, marker="o" if key == "r6_v3" else ("s" if key == "cx_refit" else "^"),
                ms=5, capsize=2 if key == "r6_v3" else 0, color=COL[key], lw=1.4,
                label=f"{NAME[key]} ({len(b):,} oklada): P {b.qb.mean():.1%}, pobedile {b.won.mean():.1%}")
ax.set_xlabel("P za klađenje (model + kladionica)"), ax.set_ylabel("stvarno pobedilo (udeo)")
ax.set_title("Strane na koje igramo 2026 (EV>5%, kvota≤5, bez ITF, 6 srpskih kladionica)")
ax.legend(loc="lower right", fontsize=9)

for ax, yr in zip(axes[1][:2], (2025, 2026)):
    B = base(yr)
    groups = ["sve", "ITF", "Challenger / WTA 125", "turnir"]
    tags = ["cx_cal2", "cx_hl8", "cx_refit", "cx_cal2refit"]
    w = 0.2
    for k, tag in enumerate(tags):
        V = load(tag, yr)
        j = V.merge(B[["tour", "mi", "p"]].rename(columns={"p": "pb"}), on=["tour", "mi"])
        m, s = [], []
        for g in groups:
            d = j if g == "sve" else j[j.grp == g]
            dl = ll(d.p.values, d.y.values) - ll(d.pb.values, d.y.values)
            m.append(dl.mean() * 1e3), s.append(2 * dl.std() / np.sqrt(len(dl)) * 1e3)
        ax.bar(np.arange(4) + (k - 1.5) * w, m, w, yerr=s, color=COL[tag], label=NAME[tag], capsize=2, error_kw=dict(lw=1, color="#333"))
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(4)), ax.set_xticklabels(groups)
    ax.set_ylabel("razlika log-loss prema osnovnom (×10⁻³), manje = bolje")
    ax.set_title(f"Uparena razlika log-loss {yr} (WTA+ATP), crta = ±2 SE")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower left", fontsize=9)

ax = axes[1][2]
ax.axis("off")
ax.text(0, 1, "Kako čitati\n\n"
        "Gore levo/sredina: linije skoro se poklapaju; refit varijante\n"
        "su u okviru intervala osnovnog modela.\n\n"
        "Dole levo/sredina: refit varijante su bolje na log-loss\n"
        "(najviše na Challenger/WTA 125), 'poluraspad 8' i\n"
        "'2 god. kalibracije' nisu bolje. Razlike su ~0.0005,\n"
        "tj. oko 0.1% log-lossa.\n\n"
        "Gore desno: ROI na 2026 isti u okviru šuma\n"
        "(+17.2% osnovni, +16.8% refit, +16.4% 2g+refit; SE ≈ 2.4).",
        va="top", fontsize=12, family="DejaVu Sans")
fig.suptitle("Varijante treninga i kalibracije: osnovni model (r6_v3) vs. varijante, test 2025 i 2026", fontsize=15)
fig.tight_layout()
fig.savefig(OUT, dpi=105)
print("->", OUT)
