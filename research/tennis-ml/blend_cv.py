"""Out-of-sample study of the betting-stage blend P = sigmoid(a*logit(q_model) + b*logit(Shin fair price) + c) on the
2026 Serbian-book sides (eval_levels2026.py --bets=..., best opening price of the six books, no ITF). Every variant is
fitted on one time block and scored on another (never on its own data); the production blend (0.44 / 0.64 / 0, fitted
once on TennisExplorer Bet365 2014-2024 with an older model) is the reference.

  a) refit (a, b, c) on 2026 blocks, or on the older external sets (TE Bet365, BetMGM)
  b) separate coefficients per tour / per level group (Challenger + WTA 125 vs tour events)
  c) market term from the consensus of all books quoting the match (mean / median Shin, Shin of median prices,
     Shin of the best prices on both sides) instead of the Shin of the best-price book's own pair; betting stays at the best price
  d) Platt / isotonic on top of the blend, odds-dependent market weight

Splits (scheme): half = fit Jan-May / test Jun-Oct and the reverse (pooled out-of-sample); block4 = four equal time blocks,
leave one out; walk = expanding window by month (causal, first test month = --walk-start).
Reported per variant on the pooled out-of-sample sides: log-loss of all stands-sides (difference to the production blend
with standard error clustered by tournament), flat ROI of the plan rule (EV>5%, odds<=5, voids count 0) with its
standard error and lower 90% bound, ROI of the same number of top-EV bets as the production rule, total-profit
difference to the production rule (clustered SE), and the bet-side calibration (mean P vs win share, EV>5/10/15%).

usage: blend_cv.py BETS.parquet FS_DIR [--matches=DIR] [--ext=DIR] [--scheme=half|block4|walk|all] [--cut=2026-06-01]
                   [--walk-start=2026-04-01] [--only=a,b,c,d] [--ladder='NAME;NAME'] [--perfold='NAME;NAME'] [--coefs]
  BETS.parquet  eval_levels2026.py --bets output WITH the fs_mid / start / book columns (current script writes them)
  FS_DIR        folder with {tour}_fs_odds_RS.parquet (per-book prices for the consensus variants)
  --matches=DIR  folder with {tour}_matches.parquet (tourney_id for clustering; default: cluster by match)
  --ext=DIR      folder with te_bet365_sides.parquet / betmgm_sides.parquet (older-model external fitting sets), built by
                 blend_cv.py --build-ext=DIR DATA_DIR  (re-runs the sides construction of blend_transfer.py; DATA_DIR as there)
  --only=a,b,c,d restrict to those question groups; --ladder='NAME;NAME' prints the EV-threshold ladder of those variants;
  --coefs prints the fitted coefficients (scheme half)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ITF = {"15", "25", "35", "50", "75", "100"}
CUR = (0.44, 0.64)  # production blend coefficients
ARG = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
cl = lambda p: np.clip(p, 1e-4, 1 - 1e-4)
lg = lambda p: np.log(cl(p) / (1 - cl(p)))


def shin(a, b):
    lo, hi = np.zeros_like(a), np.full_like(a, 0.5)
    B = a + b
    f = lambda z, x: (np.sqrt(z ** 2 + 4 * (1 - z) * x ** 2 / B) - z) / (2 * (1 - z))
    for _ in range(50):
        z = (lo + hi) / 2
        s = f(z, a) + f(z, b)
        hi, lo = np.where(s < 1, z, hi), np.where(s < 1, lo, z)
    return f((lo + hi) / 2, a)


def fit_lr(X, y, offset=None):
    """Unpenalised logistic regression (L-BFGS); returns coefficients and their covariance."""
    X, off = np.asarray(X, float), (0.0 if offset is None else np.asarray(offset, float))

    def f(b):
        z = X @ b + off
        return np.sum(np.logaddexp(0, z) - y * z), X.T @ (expit(z) - y)

    b = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    p = expit(X @ b + off)
    return b, np.linalg.pinv((X * (p * (1 - p))[:, None]).T @ X)


def cmean(v, g):
    """Mean of v with a standard error clustered by g."""
    v = np.asarray(v, float)
    s = pd.Series(v).groupby(np.asarray(g)).agg(["sum", "size"])
    m = s["sum"].sum() / s["size"].sum()
    return m, np.sqrt(((s["sum"] - m * s["size"]) ** 2).sum()) / s["size"].sum()


# ------------------------------------------------------------------ data
def build_ext(out, data):
    """Re-run blend_transfer.py and keep its sides: TE Bet365 2014-2025 (S) and BetMGM 2025-26 (T)."""
    sys.argv = ["blend_transfer.py", data]
    src = open(os.path.join(HERE, "blend_transfer.py")).read().replace("os.path.dirname(os.path.abspath(__file__))", repr(HERE))
    ns = {"__name__": "blend_transfer"}
    exec(compile(src, "blend_transfer.py", "exec"), ns)
    os.makedirs(out, exist_ok=True)
    ns["S"].to_parquet(f"{out}/te_bet365_sides.parquet")
    ns["T"].to_parquet(f"{out}/betmgm_sides.parquet")


def load(bets, fs_dir, matches):
    S = pd.read_parquet(bets)
    S = S[~S.lvl.astype(str).isin(ITF)].reset_index(drop=True)
    S["dt"] = pd.to_datetime(S.start, unit="s")
    S["grp"] = np.where(S.lvl.astype(str) == "C", "chal", "tour")
    S["tg"] = S.tour + "_" + S.grp
    S["stands"] = S.pnl != 0
    S["y"] = S.won.astype(float)
    S["m_best"] = shin(1 / S.o.values, 1 / S.oo.values)  # production market term: the best-price book's own pair
    A = []
    for tour in sorted(S.tour.unique()):
        O = pd.read_parquet(f"{fs_dir}/{tour}_fs_odds_RS.parquet")
        O = O[(O.open_h > 1) & (O.open_a > 1)].copy()
        O["ph"] = shin(1 / O.open_h.values, 1 / O.open_a.values)
        O["lph"] = lg(O.ph.values)
        g = O.groupby("fs_mid")
        a = pd.DataFrame({"nb": g.size(), "mean": g.ph.mean(), "lmean": g.lph.mean(), "med": g.ph.median(),
                          "hmed": g.open_h.median(), "amed": g.open_a.median(), "hbest": g.open_h.max(),
                          "abest": g.open_a.max(), "home_won": g.home_won.first()})
        a["pmed"] = shin(1 / a.hmed.values, 1 / a.amed.values)
        ib, ab = 1 / a.hbest.values, 1 / a.abest.values
        a["bb"] = np.where(ib + ab < 1, ib / (ib + ab), shin(ib, ab))  # Shin needs a positive margin
        a["lmean"] = expit(a.lmean)
        A.append(a.reset_index().assign(tour=tour))
    A = pd.concat(A)
    S = S.merge(A, on=["tour", "fs_mid"], how="left")
    home = (S.won == S.home_won).values
    for c, n in (("mean", "m_mean"), ("lmean", "m_lmean"), ("med", "m_med"), ("pmed", "m_pmed"), ("bb", "m_bb")):
        S[n] = np.where(home, S[c], 1 - S[c])
    assert S.m_mean.notna().all(), "per-book prices missing for some sides"
    if matches:
        t = pd.concat([pd.read_parquet(f"{matches}/{tour}_matches.parquet", columns=["tourney_id"]).tourney_id.astype(str)
                       .rename_axis("mi").reset_index().assign(tour=tour) for tour in sorted(S.tour.unique())])
        S = S.merge(t.rename(columns={"tourney_id": "cluster"}), on=["tour", "mi"], how="left")
        S["cluster"] = S.tour + "|" + S.cluster
    else:
        S["cluster"] = S.tour + "|" + S.mi.astype(str)
    S["lq"], S["lm"] = lg(S.q.values), lg(S.m_best.values)
    return S


# ------------------------------------------------------------------ variants: make(train) -> predict(test)
def lr_variant(feat, icpt=True, offset=None, fit_mask=None):
    def make(tr):
        t = tr[tr.stands]
        if fit_mask is not None:
            t = t[fit_mask(t)]
        X = feat(t)
        X = np.c_[X, np.ones(len(t))] if icpt else X
        b, cov = fit_lr(X, t.y.values, offset(t) if offset else None)

        def pred(te):
            Z = feat(te)
            Z = np.c_[Z, np.ones(len(te))] if icpt else Z
            return expit(Z @ b + (offset(te) if offset else 0.0))

        pred.beta, pred.se = b, np.sqrt(np.diag(cov))
        return pred
    return make


def fixed(a, b, c=0.0, mcol="lm"):
    return lambda tr: (lambda te: expit(a * te.lq.values + b * te[mcol].values + c))


def grouped(col, base):
    def make(tr):
        pooled, cells = base(tr), {}
        for k, g in tr[tr.stands].groupby(col):
            if len(g) >= 400:
                cells[k] = base(g.assign(stands=True))

        def pred(te):
            P = pooled(te).copy()
            for k, f in cells.items():
                m = (te[col] == k).values
                if m.any():
                    P[m] = f(te[m])
            return P
        pred.cells = cells
        return pred
    return make


def stage2(base, kind):
    def make(tr):
        f = base(tr)
        t = tr[tr.stands]
        p = f(t)
        if kind == "platt":
            b, _ = fit_lr(np.c_[lg(p), np.ones(len(t))], t.y.values)
            return lambda te: expit(b[0] * lg(f(te)) + b[1])
        iso = IsotonicRegression(y_min=1e-3, y_max=1 - 1e-3, out_of_bounds="clip").fit(p, t.y.values)
        return lambda te: iso.predict(f(te))
    return make


def mcol_feat(col):
    return lambda d: np.c_[d.lq.values, lg(d[col].values)]


ho = lambda d: np.log(d.o.values) - np.log(2.0)  # odds-dependent weights, centred at odds 2
FEATS_ODDS = {
    "log-odds": lambda d: np.c_[d.lq.values, d.lm.values, d.lq.values * ho(d), d.lm.values * ho(d)],
    "bands": lambda d: np.c_[d.lq.values, d.lm.values, d.lq.values * (d.o.values > 2), d.lm.values * (d.o.values > 2),
                             d.lq.values * (d.o.values > 3.5), d.lm.values * (d.o.values > 3.5)],
}
LM_ONLY = lambda d: d.lm.values[:, None]
M_TERMS = {"mean": "m_mean", "logit-mean": "m_lmean", "median": "m_med", "Shin(median prices)": "m_pmed", "Shin(best both)": "m_bb"}


def ext_fit(path, segs, scheme):
    """Coefficients (a, b, c) fitted once on an external older-model set (TE Bet365 / BetMGM), applied unchanged."""
    E = pd.read_parquet(path)
    if segs:
        E = E[E.seg.isin(segs)]
    if "stands" not in E:
        E["stands"] = E.pnl != 0
    E = E[E.stands]
    y = E.won.values.astype(float) if "won" in E else (E.pnl > 0).values.astype(float)
    b, cov = fit_lr(np.c_[lg(E.q.values), lg(E.shin.values), np.ones(len(E))], y)
    return b, np.sqrt(np.diag(cov)), len(E)


def variants(ext):
    V = {}
    V["a"] = {
        "production 0.44/0.64/0": fixed(*CUR),
        "model alone": lambda tr: (lambda te: te.q.values),
        "market alone (best-price Shin)": lambda tr: (lambda te: te.m_best.values),
        "refit a,b,c": lr_variant(lambda d: d[["lq", "lm"]].values),
        "refit a,b (c=0)": lr_variant(lambda d: d[["lq", "lm"]].values, icpt=False),
        "refit a+b=1, c": lr_variant(lambda d: (d.lq - d.lm).values[:, None], offset=lambda d: d.lm.values),
        "refit a+b=1, c=0": lr_variant(lambda d: (d.lq - d.lm).values[:, None], icpt=False, offset=lambda d: d.lm.values),
        "refit a,b,c on odds<=5 sides": lr_variant(lambda d: d[["lq", "lm"]].values, fit_mask=lambda d: d.o.values <= 5),
    }
    for name, (b, se, n) in ext.items():
        V["a"][f"external {name}"] = fixed(b[0], b[1], b[2])
    V["b"] = {
        "refit a,b,c pooled": V["a"]["refit a,b,c"],
        "per tour": grouped("tour", V["a"]["refit a,b,c"]),
        "per level group": grouped("grp", V["a"]["refit a,b,c"]),
        "per tour x level group": grouped("tg", V["a"]["refit a,b,c"]),
        "per tour x level group, a,b only": grouped("tg", V["a"]["refit a,b (c=0)"]),
    }
    V["c"] = {"production market term (best-price book)": fixed(*CUR)}
    for n, col in M_TERMS.items():
        V["c"][f"fixed 0.44/0.64, {n}"] = fixed(CUR[0], CUR[1], 0.0, mcol="lm_" + col)
        V["c"][f"refit a,b,c, {n}"] = lr_variant(mcol_feat(col))
    V["c"]["refit a,b,c, mean + best-price Shin"] = lr_variant(lambda d: np.c_[d.lq.values, lg(d.m_mean.values), d.lm.values])
    base = V["a"]["refit a,b,c"]
    V["d"] = {
        "production blend": fixed(*CUR),
        "production + Platt": stage2(fixed(*CUR), "platt"),
        "production + isotonic": stage2(fixed(*CUR), "iso"),
        "refit a,b,c + Platt": stage2(base, "platt"),
        "refit a,b,c + isotonic": stage2(base, "iso"),
        "odds-dependent weights (log odds)": lr_variant(FEATS_ODDS["log-odds"]),
        "odds-dependent weights (bands 2, 3.5)": lr_variant(FEATS_ODDS["bands"]),
        "consensus mean + odds-dependent (log odds)": lr_variant(lambda d: np.c_[
            d.lq.values, lg(d.m_mean.values), d.lq.values * ho(d), lg(d.m_mean.values) * ho(d)]),
    }
    return V


# ------------------------------------------------------------------ evaluation
def folds(S, scheme):
    t = S.dt
    if scheme == "half":
        cut = pd.Timestamp(ARG.get("--cut", "2026-06-01"))
        return [(t < cut, t >= cut), (t >= cut, t < cut)]
    if scheme == "block4":
        q = t.quantile([0.25, 0.5, 0.75]).values
        b = np.digitize(t.values.astype("int64"), q.astype("int64"))
        return [(pd.Series(b != k, index=t.index), pd.Series(b == k, index=t.index)) for k in range(4)]
    m0 = pd.Timestamp(ARG.get("--walk-start", "2026-04-01"))
    out = []
    for s in pd.period_range(m0, S.dt.max(), freq="M"):
        a, b = s.start_time, s.end_time
        out.append((t < a, (t >= a) & (t <= b)))
    return out


def run(S, make, scheme):
    P = np.full(len(S), np.nan)
    for tr, te in folds(S, scheme):
        f = make(S[tr.values].reset_index(drop=True))
        P[te.values] = f(S[te.values].reset_index(drop=True))
    return P


def summary(S, P, ref, thr=0.05, cap=5):
    """Pooled out-of-sample scores. ROI of the plan rule (EV>thr, odds<=cap, voids = 0), the log-loss difference to the
    reference P (production blend), ROI of the same number of top-EV bets as the reference rule (so a shifted P scale does
    not change the bet count) and total flat profit difference to the reference rule, all with standard errors clustered by
    tournament where paired."""
    ok = ~np.isnan(P)
    D, P, r = S[ok], P[ok], ref[ok]
    o, pnl, g, st = D.o.values, D.pnl.values, D.cluster.values, D.stands.values
    p, y = cl(P[st]), D.y.values[st]
    out = {"sides": int(st.sum()), "ll": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(), "brier": ((p - y) ** 2).mean()}
    bet = (P * o - 1 > thr) & (o <= cap)
    betr = (r * o - 1 > thr) & (o <= cap)
    pn = pnl[bet]
    out["n"] = int(bet.sum())
    out["roi"] = pn.mean() if len(pn) else np.nan
    out["se"] = pn.std(ddof=1) / np.sqrt(len(pn)) if len(pn) > 1 else np.nan
    out["lo90"] = out["roi"] - 1.645 * out["se"]
    out["roi_cl"] = cmean(pn, g[bet])[1] if len(pn) > 1 else np.nan
    k = int(betr.sum())
    score = np.where(o <= cap, P * o - 1, -9.0)
    topm = np.zeros(len(D), bool)
    topm[np.argsort(-score, kind="stable")[:k]] = True
    out["thrN"] = float(score[topm].min())  # the EV threshold that gives the reference bet count
    pt = pnl[topm]
    out["roi_topN"], out["se_topN"] = pt.mean(), pt.std(ddof=1) / np.sqrt(k)
    lr_ = -(D.y.values * np.log(cl(r)) + (1 - D.y.values) * np.log(1 - cl(r)))
    lm_ = -(D.y.values * np.log(cl(P)) + (1 - D.y.values) * np.log(1 - cl(P)))
    out["dll"], out["dll_se"] = cmean((lm_ - lr_)[st], g[st])
    m, se = cmean((bet.astype(float) - betr) * pnl, g)
    out["dprofit"], out["dprofit_se"] = m * len(D), se * len(D)
    m, se = cmean((topm.astype(float) - betr) * pnl, g)
    out["dtop"], out["dtop_se"] = m * len(D) / k, se * len(D) / k
    return out


def calib(S, P):
    ok = ~np.isnan(P)
    D, P = S[ok], P[ok]
    rows = {}
    for thr in (0.05, 0.10, 0.15):
        m = ((P * D.o.values - 1) > thr) & (D.o.values <= 5) & D.stands.values
        if m.sum() < 5:
            rows[thr] = (int(m.sum()), np.nan, np.nan, np.nan)
            continue
        e = D.y.values[m] - P[m]
        rows[thr] = (int(m.sum()), P[m].mean(), D.y.values[m].mean(), e.std(ddof=1) / np.sqrt(m.sum()))
    return rows


def fmt_table(rows, cols):
    head = "| variant | " + " | ".join(c[0] for c in cols) + " |\n|---|" + "---|" * len(cols)
    body = ["| " + n + " | " + " | ".join(c[1](r) for c in cols) + " |" for n, r in rows.items()]
    return head + "\n" + "\n".join(body)


COLS = [
    ("sides", lambda r: f"{r['sides']:,}"),
    ("log-loss", lambda r: f"{r['ll']:.4f}"),
    ("dLL vs prod (se)", lambda r: f"{r['dll'] * 1e3:+.2f} ({r['dll_se'] * 1e3:.2f})e-3"),
    ("bets", lambda r: f"{r['n']:,}"),
    ("ROI (se, se by tourney)", lambda r: f"{r['roi']:+.1%} ({r['se']:.1%}, {r['roi_cl']:.1%})"),
    ("lo90", lambda r: f"{r['lo90']:+.1%}"),
    ("ROI same N (se) / EV thr", lambda r: f"{r['roi_topN']:+.1%} ({r['se_topN']:.1%}) / {r['thrN']:.1%}"),
    ("dROI same N vs prod (se)", lambda r: f"{r['dtop']:+.1%} ({r['dtop_se']:.1%})"),
    ("dProfit units (se)", lambda r: f"{r['dprofit']:+.0f} ({r['dprofit_se']:.0f})"),
]


def perfold(S, Ps, ref, scheme):
    """Each fold's test block scored separately (variant names with their out-of-sample P)."""
    print("#### Per fold\n\n| variant | fold | sides | dLL (se) | bets | ROI (se) | ROI same N | dROI same N (se) |\n|---|---|---|---|---|---|---|---|")
    for name, P in Ps.items():
        for i, (_, te) in enumerate(folds(S, scheme)):
            m = te.values
            r = summary(S[m].reset_index(drop=True), P[m], ref[m])
            lab = f"{S.dt[m].min():%b %d}-{S.dt[m].max():%b %d}"
            print(f"| {name} | {lab} | {r['sides']:,} | {r['dll'] * 1e3:+.2f} ({r['dll_se'] * 1e3:.2f})e-3 | {r['n']:,} | "
                  f"{r['roi']:+.1%} ({r['se']:.1%}) | {r['roi_topN']:+.1%} | {r['dtop']:+.1%} ({r['dtop_se']:.1%}) |")
    print()


def ladder(S, Ps):
    """EV-threshold ladder (odds<=5) of pooled out-of-sample P for the named variants."""
    for name, p in Ps.items():
        print(f"#### EV threshold ladder, {name}\n\n| EV> | bets | ROI (se) | lo90 | mean P | won | gap (se) | mean odds |\n|---|---|---|---|---|---|---|---|")
        for thr in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20):
            m = (p * S.o.values - 1 > thr) & (S.o.values <= 5)
            if m.sum() < 30:
                continue
            pn, s_ = S.pnl.values[m], S.stands.values[m]
            e = S.y.values[m][s_] - p[m][s_]
            print(f"| {thr:.1%} | {m.sum():,} | {pn.mean():+.1%} ({pn.std(ddof=1) / np.sqrt(m.sum()):.1%}) | "
                  f"{pn.mean() - 1.645 * pn.std(ddof=1) / np.sqrt(m.sum()):+.1%} | {p[m][s_].mean():.1%} | {S.y.values[m][s_].mean():.1%} | "
                  f"{e.mean():+.1%} ({e.std(ddof=1) / np.sqrt(s_.sum()):.1%}) | {S.o.values[m].mean():.2f} |")
        print()


def main():
    if "--build-ext" in ARG or any(a.startswith("--build-ext=") for a in sys.argv):
        build_ext(ARG["--build-ext"], POS[0])
        return
    S = load(POS[0], POS[1], ARG.get("--matches"))
    for n, c in (("mean", "m_mean"), ("lmean", "m_lmean"), ("med", "m_med"), ("pmed", "m_pmed"), ("bb", "m_bb")):
        S["lm_" + c] = lg(S[c].values)
    ext = {}
    if "--ext" in ARG:
        e = ARG["--ext"]
        ext = {"TE Bet365 2014-25 (all segments)": ext_fit(f"{e}/te_bet365_sides.parquet", None, None),
               "TE Bet365 2014-25 (no ITF)": ext_fit(f"{e}/te_bet365_sides.parquet", ["WTA125", "WTA Tour", "Challenger"], None),
               "BetMGM 2025-26 (all segments)": ext_fit(f"{e}/betmgm_sides.parquet", None, None),
               "BetMGM 2025-26 (Challenger)": ext_fit(f"{e}/betmgm_sides.parquet", ["Challenger"], None)}
        print("## External fits (older model predictions, a / b / c with se)\n")
        for k, (b, se, n) in ext.items():
            print(f"- {k}: n={n:,}  a={b[0]:.3f} ({se[0]:.3f})  b={b[1]:.3f} ({se[1]:.3f})  c={b[2]:+.3f} ({se[2]:.3f})")
        print()
    V = variants(ext)
    groups = ARG.get("--only", "a,b,c,d").split(",")
    schemes = ["half", "block4", "walk"] if ARG.get("--scheme", "half") == "all" else [ARG.get("--scheme", "half")]
    ladder_names = {n: None for n in ARG.get("--ladder", "").split(";") if n}
    perfold_names = {n: None for n in ARG.get("--perfold", "").split(";") if n}
    n_all = len(S)
    print(f"## Data\n\n{n_all:,} no-ITF sides, {S.groupby(['tour', 'mi']).ngroups:,} matches, {S.stands.sum():,} with a result, "
          f"{S.dt.min():%Y-%m-%d} .. {S.dt.max():%Y-%m-%d}, {S.cluster.nunique():,} clusters\n")
    ref_make = fixed(*CUR)
    for scheme in schemes:
        ref = run(S, ref_make, scheme)
        ok = ~np.isnan(ref)
        print(f"## Scheme {scheme}: {ok.sum():,} out-of-sample sides in {len(folds(S, scheme))} folds\n")
        for g in groups:
            res, cal, lad, pf = {}, {}, {}, {}
            for name, make in V[g].items():
                P = run(S, make, scheme)
                if name in ladder_names:
                    lad[name] = P
                if name in perfold_names:
                    pf[name] = P
                res[name] = summary(S, np.where(ok, P, np.nan), ref)
                cal[name] = calib(S, np.where(ok, P, np.nan))
            print(f"### Question {g}\n")
            print(fmt_table(res, COLS))
            print("\nBet-side calibration (pooled out-of-sample; n, mean P, won share, gap se):\n")
            print("| variant | EV>5% | EV>10% | EV>15% |\n|---|---|---|---|")
            for name, c in cal.items():
                cells = [f"{c[t][0]}: P {c[t][1]:.1%} / won {c[t][2]:.1%} (gap {c[t][2] - c[t][1]:+.1%} ± {c[t][3]:.1%})"
                         if c[t][0] >= 5 else f"{c[t][0]}" for t in (0.05, 0.10, 0.15)]
                print(f"| {name} | " + " | ".join(cells) + " |")
            print()
            if pf:
                perfold(S, pf, ref, scheme)
            if lad:
                ladder(S[ok].reset_index(drop=True), {k: v[ok] for k, v in lad.items()})
        if scheme == "half" and "--coefs" in sys.argv:
            print("### Coefficients fitted on each half and on the whole year (in-sample, with se)\n")
            for name in ("refit a,b,c", "refit a,b (c=0)", "refit a+b=1, c"):
                for lab, m in (("Jan-May", S.dt < pd.Timestamp(ARG.get("--cut", "2026-06-01"))),
                               ("Jun-Oct", S.dt >= pd.Timestamp(ARG.get("--cut", "2026-06-01"))), ("whole", S.dt == S.dt)):
                    f = V["a"][name](S[m.values].reset_index(drop=True))
                    print(f"- {name} {lab}: " + ", ".join(f"{b:.3f} ({s:.3f})" for b, s in zip(f.beta, f.se)))
            for lab, m in (("ATP Challenger", S.tg == "atp_chal"), ("ATP tour", S.tg == "atp_tour"), ("WTA 125", S.tg == "wta_chal"),
                           ("WTA tour", S.tg == "wta_tour"), ("ATP", S.tour == "atp"), ("WTA", S.tour == "wta"),
                           ("Challenger+125", S.grp == "chal"), ("tour events", S.grp == "tour")):
                f = V["a"]["refit a,b,c"](S[m.values].reset_index(drop=True))
                print(f"- refit a,b,c {lab} (n={int(m.sum()):,}): " + ", ".join(f"{b:.3f} ({s:.3f})" for b, s in zip(f.beta, f.se)))
            print()


if __name__ == "__main__":
    main()
