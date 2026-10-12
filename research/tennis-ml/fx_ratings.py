"""Rating-system, opponent-quality and common-opponent features from results only (no odds, no stats).

Every value for a match is recorded before that match touches any state and only uses rows with a smaller
'order' (vectorised parts use exclusive per-group cumulative sums over (group, order); loops record first, then update).
Columns are w_<name> / l_<name> = value for the match winner / loser (float32), same index as the input.
Needs columns: winner_id loser_id date surface tourney_level score retired main order (rows sorted by order).
Time windows use the running maximum of 'date', so a row dated before an earlier row (live tail) counts as that later day.

usage: fx_ratings.py MATCHES_PARQUET OUT_PARQUET
"""
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os
import pickle
import re
import sys
from collections import deque
from math import erfc, exp, log, log1p, pi, sqrt

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import ndtr

LN2 = log(2.0)
BIG = 10 ** 6
SC = 173.7178
SURF = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
LG1 = {"10", "15", "25", "S"}
LG2 = {"35", "35+H", "40", "50", "50+H", "60", "75", "80", "100"}
_G = {}


# ---------------------------------------------------------------- inputs
_score_cache = {}


def parse_score(s):
    """(games won, games lost, sets with a tiebreak, completed sets) over completed non-match-tiebreak sets."""
    if s in _score_cache:
        return _score_cache[s]
    gw = gl = tb = cs = 0
    for a, b in re.findall(r"(\d+)-(\d+)", s.split("RET")[0]):
        a, b = int(a), int(b)
        hi, lo = max(a, b), min(a, b)
        if hi >= 10 or hi < 6 or not (hi - lo >= 2 or hi == 7):
            continue
        gw += a; gl += b; cs += 1
        tb += hi == 7 and lo == 6
    _score_cache[s] = (gw, gl, tb, cs)
    return _score_cache[s]


def _prep(M):
    n = len(M)
    codes, ids = pd.factorize(np.r_[M.winner_id.values, M.loser_id.values])
    day = np.maximum.accumulate(M["date"].values.astype("datetime64[D]").astype(np.int64))  # live tail rows can be out of date order
    lvl = M["tourney_level"].astype(str)
    sc_codes, uniq = pd.factorize(M["score"].fillna("").astype(str))
    ps = np.array([parse_score(s) for s in uniq], dtype=float).reshape(-1, 4)[sc_codes]
    gtot = ps[:, 0] + ps[:, 1]
    md = pd.DatetimeIndex(day.astype("datetime64[D]"))
    G = dict(
        n=n, nP=int(codes.max()) + 1, w=codes[:n], l=codes[n:], ids=np.asarray(ids),
        day=day,
        surf=M["surface"].astype(str).str.capitalize().map(SURF).fillna(0).astype(int).values,
        lg=np.where(lvl.isin(LG1), 0, np.where(lvl.isin(LG2), 1, np.where(lvl == "C", 2, 3))),
        ret=M["retired"].values.astype(bool), main=M["main"].values.astype(int),
        offs=(md.month * 100 + md.day >= 1015),
        mg=np.where(gtot > 0, (ps[:, 0] - ps[:, 1]) / np.maximum(gtot, 1), np.nan), tbs=ps[:, 2], cmp=ps[:, 3])
    return G


def _std_elo(G):
    """Standard Elo exactly as features2.ratings (K=250/(n+5)^0.4, retirements x0.5), recorded pre/post match."""
    n, nP = G["n"], G["nP"]
    w, l, ret = G["w"].tolist(), G["l"].tolist(), G["ret"].tolist()
    E, N = [1500.0] * nP, [0] * nP
    ew, el, pw, ewp, elp, nw, nl = ([0.0] * n for _ in range(7))
    for i in range(n):
        a, b = w[i], l[i]
        ra, rb, na, nb = E[a], E[b], N[a], N[b]
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        m = 0.5 if ret[i] else 1.0
        ra2 = ra + m * 250.0 / (na + 5) ** 0.4 * (1 - e)
        rb2 = rb - m * 250.0 / (nb + 5) ** 0.4 * (1 - e)
        E[a], E[b], N[a], N[b] = ra2, rb2, na + 1, nb + 1
        ew[i], el[i], pw[i], ewp[i], elp[i], nw[i], nl[i] = ra, rb, e, ra2, rb2, na, nb
    G.update(ew=np.array(ew), el=np.array(el), pw=np.array(pw), ewp=np.array(ewp), elp=np.array(elp),
             nw=np.array(nw), nl=np.array(nl))


def _split(prefix_pairs):
    out = {}
    for name, (a, b) in prefix_pairs.items():
        out["w_" + name] = np.asarray(a, dtype=np.float32); out["l_" + name] = np.asarray(b, dtype=np.float32)
    return out


# ---------------------------------------------------------------- vectorised per-group windows
class Long:
    """Rows sorted by (group key, order) with exclusive per-group cumulative sums and day-window lookups."""

    def __init__(self, key, order, day):
        self.srt = np.lexsort((order, key))
        self.inv = np.empty_like(self.srt); self.inv[self.srt] = np.arange(len(key))
        k = key[self.srt]
        brk = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
        lens = np.diff(np.r_[brk, len(k)])
        self.start = np.repeat(brk, lens)
        self.gid = np.repeat(np.arange(len(brk)), lens)
        self.ar = np.arange(len(k))
        self.pos = self.ar - self.start
        self.day = day[self.srt]
        self.d0 = self.day[self.start]
        self.dkey = self.gid.astype(np.int64) * BIG + self.day
        self._lo = {}

    def g(self, v):
        return np.asarray(v)[self.srt]

    def back(self, v):
        return v[self.inv]

    def excl(self, v):
        v = np.nan_to_num(np.asarray(v, dtype=float))
        c = pd.Series(v).groupby(self.gid).cumsum().to_numpy()
        ex = np.r_[0.0, c[:-1]]
        ex[self.pos == 0] = 0.0
        return ex

    def lo(self, window):
        if window is None:
            return self.start
        if window not in self._lo:
            self._lo[window] = np.searchsorted(self.dkey, self.dkey - window, side="left")
        return self._lo[window]

    def rows(self, v, k):
        ex = self.excl(v)
        return ex - ex[np.maximum(self.ar - k, self.start)]

    def dsum(self, v, window):
        ex = self.excl(v)
        return ex - ex[self.lo(window)]

    def wdsum(self, v, lam, window=None):
        sc = np.exp(lam * (self.day - self.d0))
        ex = self.excl(np.nan_to_num(v) * sc)
        return (ex - ex[self.lo(window)]) / sc

    def rows_decay(self, v, k, halflife):
        sc = np.exp(LN2 / halflife * self.pos)
        ex = self.excl(np.nan_to_num(v) * sc)
        return (ex - ex[np.maximum(self.ar - k, self.start)]) / sc

    def shift(self, v, k):
        out = np.full(len(v), np.nan)
        out[k:] = v[:-k]
        out[self.pos < k] = np.nan
        return out


# ---------------------------------------------------------------- Glicko-2
def _f(x, a, d2, phi2, v):
    ex = exp(x)
    return ex * (d2 - phi2 - v - ex) / (2.0 * (phi2 + v + ex) ** 2) - (x - a) * 4.0


def _g2_update(mu, phi, sig, muo, phio, s):
    g = 1.0 / sqrt(1.0 + 3.0 * phio * phio / (pi * pi))
    E = 1.0 / (1.0 + exp(-g * (mu - muo)))
    v = 1.0 / (g * g * E * (1.0 - E))
    delta = v * g * (s - E)
    phi2, d2 = phi * phi, delta * delta
    a = log(sig * sig)
    A = a
    if d2 > phi2 + v:
        B = log(d2 - phi2 - v)
    else:
        k = 1
        while _f(a - k * 0.5, a, d2, phi2, v) < 0 and k < 100:
            k += 1
        B = a - k * 0.5
    fA, fB = _f(A, a, d2, phi2, v), _f(B, a, d2, phi2, v)
    for _ in range(80):
        if abs(B - A) <= 1e-5:
            break
        C = A + (A - B) * fA / (fB - fA)
        fC = _f(C, a, d2, phi2, v)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
    sig2 = exp(A / 2.0)
    phin = 1.0 / sqrt(1.0 / (phi2 + sig2 * sig2) + 1.0 / v)
    return mu + phin * phin * g * (s - E), phin, sig2


# ---------------------------------------------------------------- resumable sequential tasks
# A sequential task walks the rows once with a per-player state. For live fixtures (rows i0..n at the end of the table)
# the state at row i0 is saved per player id (codes change when rows are appended) with a digest of the rows it has
# seen; the next run continues from it over the rows appended since and computes only the fixture rows. A saved state
# whose rows are no longer the first rows of the table is ignored and the walk starts from the first row.
def _ckpt_load(name):
    G = _G
    path = G.get("state")
    if not path or not G["i0"]:
        return None
    try:
        with open(f"{path}_{name}.pkl", "rb") as fh:
            c = pickle.load(fh)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    return c if c["k"] <= G["i0"] and c["digest"] == _digest(c["k"]) else None


def _ckpt_save(name, k, state, **extra):
    path = _G.get("state")
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"{path}_{name}.pkl.tmp", "wb") as fh:
        pickle.dump({"k": k, "digest": _digest(k), "state": state, **extra}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(f"{path}_{name}.pkl.tmp", f"{path}_{name}.pkl")


def _digest(k):
    return hashlib.sha1(_G["rowhash"][:k].tobytes()).hexdigest()


def _code():
    if "code" not in _G:
        _G["code"] = {p: c for c, p in enumerate(_G["ids"].tolist())}
    return _G["code"]


def _seq(name, new, loop, save, load, width):
    """Run a sequential task over rows i0..n (all rows when i0 = 0); returns its per-row record for those rows."""
    G = _G
    n, i0 = G["n"], G["i0"]
    c = _ckpt_load(name)
    st, k = (load(c["state"]), c["k"]) if c else (new(), 0)
    if k < i0:
        loop(k, i0, st, None)  # rows before the first wanted row only feed the state
        if G.get("state"):
            _ckpt_save(name, i0, save(st))
    R = np.zeros((n - i0, width))
    loop(i0, n, st, R)
    return R


def _ids_of(codes):
    ids = _G["ids"]
    return [ids[c].item() for c in codes]


# ---------------------------------------------------------------- Glicko-2 (state: mu, phi, sig, last day per player, per player x surface)
def _g2_new():
    nP = _G["nP"]
    return {"mu": [0.0] * nP, "phi": [350.0 / SC] * nP, "sig": [0.06] * nP, "last": [-1] * nP, "S": {}}


def _g2_loop(lo, hi, st, R):
    G = _G
    w, l, day, sf, ret = (G[x][lo:hi].tolist() for x in ("w", "l", "day", "surf", "ret"))
    phi0 = 350.0 / SC
    mu, phi, sig, last, S = st["mu"], st["phi"], st["sig"], st["last"], st["S"]
    for j in range(hi - lo):
        a, b, d, s = w[j], l[j], day[j], sf[j]
        pa = phi[a] if last[a] < 0 else min(sqrt(phi[a] ** 2 + 0.005 * (d - last[a])), phi0)
        pb = phi[b] if last[b] < 0 else min(sqrt(phi[b] ** 2 + 0.005 * (d - last[b])), phi0)
        ma, mb = mu[a], mu[b]
        ka, kb = a * 4 + s, b * 4 + s
        sa = S.get(ka)
        sa = [ma, max(pa, 1.0), 0.06, d] if sa is None else sa
        sb = S.get(kb)
        sb = [mb, max(pb, 1.0), 0.06, d] if sb is None else sb
        psa = min(sqrt(sa[1] ** 2 + 0.005 * (d - sa[3])), phi0)
        psb = min(sqrt(sb[1] ** 2 + 0.005 * (d - sb[3])), phi0)
        if R is not None:
            R[j] = (ma, mb, pa, pb, sig[a], sig[b], (sa[0] * SC + 1500) - 2 * psa * SC, (sb[0] * SC + 1500) - 2 * psb * SC)
        sw, sl = (0.75, 0.25) if ret[j] else (1.0, 0.0)
        mu[a], phi[a], sig[a] = _g2_update(ma, pa, sig[a], mb, pb, sw)
        mu[b], phi[b], sig[b] = _g2_update(mb, pb, sig[b], ma, pa, sl)
        x = _g2_update(sa[0], psa, sa[2], sb[0], psb, sw)
        y = _g2_update(sb[0], psb, sb[2], sa[0], psa, sl)
        S[ka] = [x[0], x[1], x[2], d]
        S[kb] = [y[0], y[1], y[2], d]
        last[a] = last[b] = d


def _g2_save(st):
    seen = [c for c, t in enumerate(st["last"]) if t >= 0]
    return {"p": dict(zip(_ids_of(seen), [(st["mu"][c], st["phi"][c], st["sig"][c], st["last"][c]) for c in seen])),
            "S": {(p, k % 4): v for k, v, p in zip(st["S"], st["S"].values(), _ids_of(k // 4 for k in st["S"]))}}


def _g2_load(s):
    st, code = _g2_new(), _code()
    for p, (m, ph, sg, t) in s["p"].items():
        c = code[p]
        st["mu"][c], st["phi"][c], st["sig"][c], st["last"][c] = m, ph, sg, t
    st["S"] = {code[p] * 4 + q: v for (p, q), v in s["S"].items()}
    return st


def _task_g2():
    R = _seq("g2", _g2_new, _g2_loop, _g2_save, _g2_load, 8)
    mw, ml, pw_, pl_, vw, vl, sw_, sl_ = R.T
    rw, rl = mw * SC + 1500, ml * SC + 1500
    dw, dl = pw_ * SC, pl_ * SC
    z = (rw - rl) / np.sqrt(dw ** 2 + dl ** 2)
    gg = 1.0 / np.sqrt(1.0 + 3.0 * (pw_ ** 2 + pl_ ** 2) / pi ** 2)
    p = 1.0 / (1.0 + np.exp(-gg * (mw - ml)))
    return _split({"g2_rating": (rw, rl), "g2_rd": (dw, dl), "g2_lb": (rw - 2 * dw, rl - 2 * dl),
                   "g2_vol": (vw, vl), "g2_surf_lb": (sw_, sl_), "g2_z": (z, -z), "g2_prob": (p, 1 - p)})


# ---------------------------------------------------------------- TrueSkill and level-aware Elo variants
def _ts_new():
    nP = _G["nP"]
    v0 = (25.0 / 3) ** 2
    return {"tmu": [25.0] * nP, "tvar": [v0] * nP, "tlast": [-1] * nP, "Eo": [1500.0] * nP, "El": [None] * nP,
            "Sg": [1500.0] * 4, "LGE": {}, "LGN": {}}


def _ts_loop(lo, hi, st, R):
    G = _G
    w, l, day, ret, lgc = (G[x][lo:hi].tolist() for x in ("w", "l", "day", "ret", "lg"))
    nw, nl, ew = (G[x][lo:hi].tolist() for x in ("nw", "nl", "ew"))
    el_ = G["el"]
    s0 = 25.0 / 3
    v0, b2, t2 = s0 * s0, (s0 / 2) ** 2, (s0 / 100) ** 2
    tmu, tvar, tlast, Eo, El, Sg, LGE, LGN = (st[x] for x in ("tmu", "tvar", "tlast", "Eo", "El", "Sg", "LGE", "LGN"))
    for j in range(hi - lo):
        a, b, d, g = w[j], l[j], day[j], lgc[j]
        na, nb = nw[j], nl[j]
        m = 0.5 if ret[j] else 1.0
        ka, kb = 250.0 / (na + 5) ** 0.4, 250.0 / (nb + 5) ** 0.4
        # TrueSkill
        va = tvar[a] if tlast[a] < 0 else min(tvar[a] + t2 + 0.00048 * (d - tlast[a]), v0)
        vb = tvar[b] if tlast[b] < 0 else min(tvar[b] + t2 + 0.00048 * (d - tlast[b]), v0)
        ma, mb = tmu[a], tmu[b]
        if R is not None:
            R[j, 0:4] = (ma, mb, sqrt(va), sqrt(vb))
        c2 = 2 * b2 + va + vb
        c = sqrt(c2)
        t = (ma - mb) / c
        vv = -t if t < -30 else exp(-0.5 * t * t) / sqrt(2 * pi) / (0.5 * erfc(-t / sqrt(2.0)))
        ww = vv * (vv + t)
        tmu[a], tmu[b] = ma + va / c * vv, mb - vb / c * vv
        tvar[a], tvar[b] = va * (1 - va / c2 * ww), vb * (1 - vb / c2 * ww)
        tlast[a] = tlast[b] = d
        # opponent-experience weighted Elo
        ra, rb = Eo[a], Eo[b]
        if R is not None:
            R[j, 4:6] = (ra, rb)
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        Eo[a] = ra + m * ka * min(1.0, (nb + 2) / 15.0) * (1 - e)
        Eo[b] = rb - m * kb * min(1.0, (na + 2) / 15.0) * (1 - e)
        # Elo whose debutants start at the level-group mean - 75
        sg = Sg[g]
        ra = El[a] if El[a] is not None else sg - 75.0
        rb = El[b] if El[b] is not None else sg - 75.0
        if R is not None:
            R[j, 6:9] = (ra, rb, sg)
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        El[a] = ra + m * ka * (1 - e)
        El[b] = rb - m * kb * (1 - e)
        if na >= 10:
            sg += 0.0005 * (ra - sg)
        if nb >= 10:
            sg += 0.0005 * (rb - sg)
        Sg[g] = sg
        # Elo per level group, initialised from the overall Elo
        qa, qb = a * 4 + g, b * 4 + g
        ra = LGE.get(qa)
        ra = ew[j] if ra is None else ra
        rb = LGE.get(qb)
        rb = el_[lo + j] if rb is None else rb
        if R is not None:
            R[j, 9:11] = (ra, rb)
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        ca, cb = LGN.get(qa, 0), LGN.get(qb, 0)
        LGE[qa] = ra + m * 250.0 / (ca + 5) ** 0.4 * (1 - e)
        LGE[qb] = rb - m * 250.0 / (cb + 5) ** 0.4 * (1 - e)
        LGN[qa], LGN[qb] = ca + 1, cb + 1


def _ts_save(st):
    seen = [c for c, t in enumerate(st["tlast"]) if t >= 0]
    return {"p": dict(zip(_ids_of(seen), [tuple(st[x][c] for x in ("tmu", "tvar", "tlast", "Eo", "El")) for c in seen])),
            "Sg": list(st["Sg"]),
            "LGE": {(p, k % 4): v for k, v, p in zip(st["LGE"], st["LGE"].values(), _ids_of(k // 4 for k in st["LGE"]))},
            "LGN": {(p, k % 4): v for k, v, p in zip(st["LGN"], st["LGN"].values(), _ids_of(k // 4 for k in st["LGN"]))}}


def _ts_load(s):
    st, code = _ts_new(), _code()
    for p, vals in s["p"].items():
        c = code[p]
        for x, v in zip(("tmu", "tvar", "tlast", "Eo", "El"), vals):
            st[x][c] = v
    st["Sg"] = list(s["Sg"])
    st["LGE"] = {code[p] * 4 + q: v for (p, q), v in s["LGE"].items()}
    st["LGN"] = {code[p] * 4 + q: v for (p, q), v in s["LGN"].items()}
    return st


def _task_ts_lvl():
    G = _G
    i0 = G["i0"]
    R = _seq("ts_lvl", _ts_new, _ts_loop, _ts_save, _ts_load, 11)
    T, O, V, Q = R[:, 0:4], R[:, 4:6], R[:, 6:9], R[:, 9:11]
    b2 = (25.0 / 3 / 2) ** 2
    ew_, el_ = G["ew"][i0:], G["el"][i0:]
    dm = T[:, 0] - T[:, 1]
    z = dm / np.sqrt(T[:, 2] ** 2 + T[:, 3] ** 2)
    p = ndtr(dm / np.sqrt(2 * b2 + T[:, 2] ** 2 + T[:, 3] ** 2))
    return _split({
        "ts_mu": (T[:, 0], T[:, 1]), "ts_sigma": (T[:, 2], T[:, 3]), "ts_z": (z, -z), "ts_prob": (p, 1 - p),
        "ow_elo": (O[:, 0], O[:, 1]), "lvl_elo": (V[:, 0], V[:, 1]),
        "elo_vs_level_mean": (ew_ - V[:, 2], el_ - V[:, 2]),
        "lg_elo": (Q[:, 0], Q[:, 1]), "lg_elo_minus_elo": (Q[:, 0] - ew_, Q[:, 1] - el_)})


RHO = [[1, .75, .65, .85], [.75, 1, .5, .6], [.65, .5, 1, .75], [.85, .6, .75, 1]]


# ---------------------------------------------------------------- cross-surface Elo, layoff-aware Elo, latent matchup factors
def _cs_new():
    nP = _G["nP"]
    init = np.array([np.random.default_rng((7, int(p))).normal(0, 0.1, 8) for p in _G["ids"]])  # keyed by player id, not data span
    return {"CS": [None] * nP, "LO": [None] * nP, "llast": [-1] * nP, "loff": [False] * nP, "since": [99] * nP,
            "U": np.ascontiguousarray(init[:, :4]), "V": np.ascontiguousarray(init[:, 4:])}


def _cs_loop(lo, hi, st, R):
    G = _G
    w, l, day, ret, sf = (G[x][lo:hi].tolist() for x in ("w", "l", "day", "ret", "surf"))
    main, offs = G["main"][lo:hi].tolist(), G["offs"][lo:hi].tolist()
    nw, nl = G["nw"][lo:hi].tolist(), G["nl"][lo:hi].tolist()
    ewl, ell = G["ew"][lo:hi].tolist(), G["el"][lo:hi].tolist()
    CS, LO, llast, loff, since, U, Vv = (st[x] for x in ("CS", "LO", "llast", "loff", "since", "U", "V"))
    cm = log(10.0) / 400.0
    for j in range(hi - lo):
        a, b, d, s = w[j], l[j], day[j], sf[j]
        na, nb = nw[j], nl[j]
        m = 0.5 if ret[j] else 1.0
        # cross-surface Elo
        xa = CS[a] = CS[a] or [1500.0] * 4
        xb = CS[b] = CS[b] or [1500.0] * 4
        if R is not None:
            R[j, 0:2] = (xa[s], xb[s])
        e = 1.0 / (1.0 + 10.0 ** ((xb[s] - xa[s]) / 400.0))
        fa, fb = m * 250.0 / (na + 5) ** 0.4 * (1 - e), m * 250.0 / (nb + 5) ** 0.4 * (1 - e)
        rho = RHO[s]
        for q in range(4):
            xa[q] += rho[q] * fa
            xb[q] -= rho[q] * fb
        # layoff-aware Elo
        for p in (a, b):
            if LO[p] is None:
                LO[p] = 1500.0 if main[j] else 1250.0
            elif d - llast[p] >= 56 and not loff[p]:
                LO[p] -= min(100.0 + 1.5 * (d - llast[p] - 56) / 7.0, 200.0)
                since[p] = 0
        ra, rb = LO[a], LO[b]
        if R is not None:
            R[j, 2:4] = (ra, rb)
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        LO[a] = ra + m * 250.0 / (na + 5) ** 0.4 * (1 + 0.55 * max(0.0, 1 - since[a] / 20.0)) * (1 - e)
        LO[b] = rb - m * 250.0 / (nb + 5) ** 0.4 * (1 + 0.55 * max(0.0, 1 - since[b] / 20.0)) * (1 - e)
        since[a] += 1; since[b] += 1
        llast[a] = llast[b] = d
        loff[a] = loff[b] = offs[j]
        # latent matchup factors (logit = Elo term + u_i.v_j - u_j.v_i), one SGD step after the match
        ua, va, ub, vb = U[a], Vv[a], U[b], Vv[b]
        x = ua @ vb - ub @ va
        if R is not None:
            R[j, 4] = x if (na >= 5 and nb >= 5) else 0.0
        gr = 1.0 - 1.0 / (1.0 + exp(-(cm * (ewl[j] - ell[j]) + x)))
        da, db_ = gr * vb - 0.05 * ua, -gr * va - 0.05 * ub
        dva, dvb = -gr * ub - 0.05 * va, gr * ua - 0.05 * vb
        U[a] = ua + 0.05 * da; U[b] = ub + 0.05 * db_
        Vv[a] = va + 0.05 * dva; Vv[b] = vb + 0.05 * dvb


def _cs_save(st):
    seen = [c for c, t in enumerate(st["llast"]) if t >= 0]
    return dict(zip(_ids_of(seen), [(list(st["CS"][c]), st["LO"][c], st["llast"][c], st["loff"][c], st["since"][c],
                                     st["U"][c].copy(), st["V"][c].copy()) for c in seen]))


def _cs_load(s):
    st, code = _cs_new(), _code()
    for p, (cs, lo_, ll, lf, sn, u, v) in s.items():
        c = code[p]
        st["CS"][c], st["LO"][c], st["llast"][c], st["loff"][c], st["since"][c] = list(cs), lo_, ll, lf, sn
        st["U"][c], st["V"][c] = u, v
    return st


def _task_cs_layoff_mf():
    R = _seq("cs_layoff_mf", _cs_new, _cs_loop, _cs_save, _cs_load, 5)
    C, Lo, F = R[:, 0:2], R[:, 2:4], R[:, 4]
    return _split({"cs_elo": (C[:, 0], C[:, 1]), "layoff_elo": (Lo[:, 0], Lo[:, 1]), "mf_latent_matchup": (F, -F)})


# ---------------------------------------------------------------- opponents' current Elo vs Elo when they played
def _hind_new():
    nP = _G["nP"]
    return {"Ec": np.full(nP, 1500.0), "ho": [[] for _ in range(nP)], "ht": [[] for _ in range(nP)]}


def _hind_loop(lo, hi, st, R):
    G = _G
    w, l = G["w"][lo:hi].tolist(), G["l"][lo:hi].tolist()
    ew, el, ewp, elp = (G[x][lo:hi].tolist() for x in ("ew", "el", "ewp", "elp"))
    Ec, ho, ht = st["Ec"], st["ho"], st["ht"]
    for j in range(hi - lo):
        a, b = w[j], l[j]
        if R is not None:
            for q, p in ((0, a), (2, b)):
                h = ho[p]
                if h:
                    dif = Ec[h[-30:]] - np.array(ht[p][-30:])
                    R[j, q] = dif.mean()
                    R[j, q + 1] = dif[-10:].mean()
                else:
                    R[j, q] = R[j, q + 1] = np.nan
        ho[a].append(b); ht[a].append(el[j])
        ho[b].append(a); ht[b].append(ew[j])
        Ec[a], Ec[b] = ewp[j], elp[j]


def _hind_save(st):
    seen = [c for c, h in enumerate(st["ho"]) if h]  # only the last 30 opponents are ever read
    return dict(zip(_ids_of(seen), [(st["Ec"][c], _ids_of(st["ho"][c][-30:]), st["ht"][c][-30:]) for c in seen]))


def _hind_load(s):
    st, code = _hind_new(), _code()
    for p, (e, ops, hts) in s.items():
        c = code[p]
        st["Ec"][c], st["ho"][c], st["ht"][c] = e, [code[o] for o in ops], list(hts)
    return st


def _task_hind():
    R = _seq("hind", _hind_new, _hind_loop, _hind_save, _hind_load, 4)
    return _split({"hind_30": (R[:, 0], R[:, 2]), "hind_10": (R[:, 1], R[:, 3])})


# ---------------------------------------------------------------- common opponents and two-hop chains
def _co_new():
    nP = _G["nP"]
    return {"dq": [None] * nP, "by": [None] * nP}


def _co_window(st, p, lim):
    q = st["dq"][p]
    if q is None:
        return None
    b = st["by"][p]
    while q and q[0][0] < lim:
        o = q.popleft()[1]
        r = b[o]
        r.popleft()
        if not r:
            del b[o]
    return b


def _co_loop(lo, hi, st, R):
    G = _G
    w, l, day, mgl, pwl = (G[x][lo:hi].tolist() for x in ("w", "l", "day", "mg", "pw"))
    dq, by = st["dq"], st["by"]
    k180, k270, k365 = LN2 / 180.0, LN2 / 270.0, 1.0 / 365.0
    for j in range(hi - lo):
        a, b, d = w[j], l[j], day[j]
        if R is not None:
            lim = d - 730
            A, B = _co_window(st, a, lim), _co_window(st, b, lim)
            if A and B:
                small, big = (A, B) if len(A) < len(B) else (B, A)
                common = [c for c in small if c in big]
                nc = len(common)
                if nc:
                    R[j, 0] = log1p(nc)
                    wd = md = ed = ws = es = 0.0
                    c1 = c2 = 0.0
                    for c in common:
                        la, lb = A[c], B[c]
                        stt = []
                        for lst in (la, lb):
                            sw = sn = ms = mw = rs = rw = 0.0
                            mn = 0
                            wins, losses = [], []
                            for dd, won, mg, r in lst:
                                age = d - dd
                                x = exp(-age * k180)
                                sw += x; sn += x * won
                                if mg == mg:
                                    ms += x * mg; mw += x; mn += 1
                                y = exp(-age * k270)
                                rs += y * r; rw += y
                                (wins if won else losses).append(age)
                            stt.append(((sn + 0.5) / (sw + 1.0), ms / mw if mw else None, mn, rs / rw, len(lst), wins, losses))
                        sa, sb = stt
                        wd += sa[0] - sb[0]
                        if sa[1] is not None and sb[1] is not None:
                            x = min(sa[2], sb[2], 3)
                            md += x * (sa[1] - sb[1]); ws += x
                        x = min(sa[4], sb[4], 3)
                        ed += x * (sa[3] - sb[3]); es += x
                        for u in sa[5]:
                            for v in sb[6]:
                                c1 += exp(-max(u, v) * k365)
                        for u in sb[5]:
                            for v in sa[6]:
                                c2 += exp(-max(u, v) * k365)
                    if nc >= 2:
                        R[j, 1] = wd / nc
                        R[j, 2] = md / ws if ws else 0.0
                        R[j, 3] = ed / es if es else 0.0
                    R[j, 4] = (c1 - c2) / (c1 + c2 + 2.0) if (c1 + c2) > 0 else 0.0
        mg, r = mgl[j], 1.0 - pwl[j]
        for p, o, won, sg in ((a, b, 1, 1.0), (b, a, 0, -1.0)):
            if dq[p] is None:
                dq[p], by[p] = deque(), {}
            dq[p].append((d, o))
            by[p].setdefault(o, deque()).append((d, won, sg * mg, sg * r))


def _co_save(st):
    # entries older than 730 days before the last row are dropped now: every later row has a later or equal day, so
    # its window would drop them anyway (the order of what is left does not change)
    lim = int(_G["day"][_G["i0"] - 1]) - 730
    seen = [c for c, q in enumerate(st["dq"]) if q is not None]
    out = {}
    for c, p in zip(seen, _ids_of(seen)):
        _co_window(st, c, lim)
        dq = st["dq"][c]
        out[p] = ([(d, o) for (d, _), o in zip(dq, _ids_of(o for _, o in dq))],
                  {o: list(v) for o, v in zip(_ids_of(st["by"][c]), st["by"][c].values())})
    return out


def _co_load(s):
    st, code = _co_new(), _code()
    for p, (dq, by) in s.items():
        c = code[p]
        st["dq"][c] = deque((d, code[o]) for d, o in dq)
        st["by"][c] = {code[o]: deque(v) for o, v in by.items()}
    return st


def _task_co():
    R = _seq("co", _co_new, _co_loop, _co_save, _co_load, 5)
    return _split({"co_count": (R[:, 0], R[:, 0]), "co_win_diff": (R[:, 1], -R[:, 1]),
                   "co_margin_diff": (R[:, 2], -R[:, 2]), "co_elo_adj_diff": (R[:, 3], -R[:, 3]),
                   "trans_2hop": (R[:, 4], -R[:, 4])})


# ---------------------------------------------------------------- PageRank snapshots
def _task_pr():
    """One PageRank per match day over the previous 365 days (warm start from the last values). With a saved state the
    days before the first wanted row's day are not recomputed."""
    G = _G
    n, nP, i0 = G["n"], G["nP"], G["i0"]
    w, l, day = G["w"], G["l"], G["day"]
    PRg = np.zeros(nP)
    loc = np.full(nP, -1)
    ln15 = log(0.15)
    prw, prl = np.full(n, ln15), np.full(n, ln15)
    pcw, pcl = np.full(n, 0.15), np.full(n, 0.15)
    ud, us = np.unique(day, return_index=True)
    ends = np.r_[us[1:], n]
    d0 = int(day[i0]) if i0 < n else None  # days before it write only rows that are not wanted
    c = _ckpt_load("pr")
    start = 0
    if c and c["day"] <= d0:
        code = _code()
        for p, v in c["state"].items():
            PRg[code[p]] = v
        start = int(np.searchsorted(ud, c["day"]))

    def run(a, b):
        for d, s0, s1 in zip(ud[a:b], us[a:b], ends[a:b]):
            lo = np.searchsorted(day, d - 365, side="left")
            if s0 <= lo:
                continue
            ww, ll = w[lo:s0], l[lo:s0]
            m = len(ww)
            wt = np.exp(-(d - day[lo:s0]) / 120.0)
            nodes, inv = np.unique(np.r_[ww, ll], return_inverse=True)
            N = len(nodes)
            iw, il = inv[:m], inv[m:]
            outw = np.bincount(il, weights=wt, minlength=N)
            P = sparse.csr_matrix((wt / outw[il], (iw, il)), shape=(N, N))
            dang = outw == 0
            x = PRg[nodes]
            x = np.where(x > 0, x, 1.0 / N)
            x = x / x.sum()
            for _ in range(30):
                x = 0.85 * (P @ x + x[dang].sum() / N) + 0.15 / N
            PRg[nodes] = x
            lo90 = np.searchsorted(day, d - 90, side="left") - lo
            act = np.zeros(N, bool)
            act[iw[lo90:]] = True; act[il[lo90:]] = True
            ref = np.sort(x[act])
            lv = np.log(x * N)
            loc[nodes] = np.arange(N)
            for ids, po, pc in ((w[s0:s1], prw, pcw), (l[s0:s1], prl, pcl)):
                j = loc[ids]
                f = j >= 0
                xs = x[j[f]]
                sl = slice(s0, s1)
                po[sl][f] = lv[j[f]]
                if len(ref):
                    pc[sl][f] = (np.searchsorted(ref, xs, "left") + np.searchsorted(ref, xs, "right")) / (2.0 * len(ref))
            loc[nodes] = -1

    if i0 and d0 is not None:
        stop = int(np.searchsorted(ud, d0))
        run(start, stop)
        if G.get("state") and stop > start:
            seen = np.flatnonzero(PRg != 0)
            _ckpt_save("pr", i0, dict(zip(_ids_of(seen), PRg[seen].tolist())), day=d0)
        run(stop, len(ud))
    else:
        run(0, len(ud))
    return _split({"pagerank_365": (prw, prl), "pagerank_pct90": (pcw, pcl)})


# ---------------------------------------------------------------- long-format vectorised features
def _level_z(G):
    """Per match: mean and sd of the level group's pooled Elo over the previous 365 days (all matches of the table)."""
    n, day, ew, el = G["n"], G["day"], G["ew"], G["el"]
    mean_m, sd_m, ok_m = np.zeros(n), np.ones(n), np.zeros(n, bool)
    for g in range(4):
        idx = np.flatnonzero(G["lg"] == g)
        if not len(idx):
            continue
        dg = day[idx]
        a, b = ew[idx] - 1500.0, el[idx] - 1500.0
        c1, c2 = np.r_[0.0, np.cumsum(a + b)], np.r_[0.0, np.cumsum(a * a + b * b)]
        lo_, hi_ = np.searchsorted(dg, dg - 365, side="left"), np.searchsorted(dg, dg, side="left")
        cn = 2.0 * (hi_ - lo_)
        mu_ = (c1[hi_] - c1[lo_]) / np.maximum(cn, 1)
        mean_m[idx] = mu_ + 1500.0
        sd_m[idx] = np.sqrt(np.maximum((c2[hi_] - c2[lo_]) / np.maximum(cn, 1) - mu_ ** 2, 1.0))
        ok_m[idx] = cn >= 40
    return mean_m, sd_m, ok_m


def _long_rows(G):
    """Matches whose rows the wanted rows' long features read: every match of the wanted rows' players and of their
    opponents in the 730 days before the first wanted day or in their last 10 matches (the opponent's Elo form and
    tiebreak bucket at those matches come from the opponent's own history)."""
    w, l, day, i0 = G["w"], G["l"], G["day"], G["i0"]
    P = np.unique(np.r_[w[i0:], l[i0:]])
    a, o = np.r_[w[:i0], l[:i0]], np.r_[l[:i0], w[:i0]]
    ix, dd = np.r_[np.arange(i0), np.arange(i0)], np.r_[day[:i0], day[:i0]]
    m = np.isin(a, P)
    srt = np.lexsort((ix[m], a[m]))
    g = a[m][srt]
    end = np.flatnonzero(np.r_[g[1:] != g[:-1], True])
    last10 = np.zeros(int(m.sum()), bool)
    last10[srt] = np.repeat(end, np.diff(np.r_[-1, end])) - np.arange(len(g)) < 10
    Q = np.unique(o[m][(dd[m] >= day[i0:].min() - 730) | last10])
    PQ = np.union1d(P, Q)
    keep = np.isin(w, PQ) | np.isin(l, PQ)
    keep[i0:] = True
    return keep


def _task_long():
    G = _G
    n, i0 = G["n"], G["i0"]
    zst = _level_z(G)
    if not i0:
        return _long(G, zst)
    # only the rows the wanted rows read (same values: every window of a kept player holds all its rows)
    keep = _long_rows(G)
    H = {k: G[k][keep] for k in ("w", "l", "day", "pw", "ew", "el", "nl", "nw", "tbs", "cmp", "surf")}
    H.update(n=int(keep.sum()), nP=G["nP"])
    out = _long(H, tuple(z[keep] for z in zst))
    pos = np.searchsorted(np.flatnonzero(keep), np.arange(i0, n))
    return {k: v[pos] for k, v in out.items()}


def _long(G, zst):
    n = G["n"]
    w, l, day, pw = G["w"], G["l"], G["day"], G["pw"]
    ew, el = G["ew"], G["el"]
    mi = np.r_[np.arange(n), np.arange(n)]
    pl, dayl = np.r_[w, l], np.r_[day, day]
    own_u, opp_u = np.r_[ew, el], np.r_[el, ew]
    won_u = np.r_[np.ones(n), np.zeros(n)]
    res_u = np.r_[1 - pw, -(1 - pw)]
    oppn_u = np.r_[G["nl"], G["nw"]]
    L = Long(pl, mi, dayl)
    own, opp, won, res = L.g(own_u), L.g(opp_u), L.g(won_u), L.g(res_u)
    pos, ar, st = L.pos, L.ar, L.start
    ones = np.ones(len(pos))
    first = pos == 0
    out = {}

    def put(name, v, lng=L):
        u = lng.back(v)
        out["w_" + name], out["l_" + name] = u[:n].astype(np.float32), u[n:].astype(np.float32)

    cnt20 = np.minimum(pos, 20)
    prov = L.rows((L.g(oppn_u) < 5).astype(float), 20) / np.maximum(cnt20, 1)
    put("prov_opp_20", np.where(first, np.nan, prov))

    oe10 = np.where(first, np.nan, L.rows(opp, 10) / np.maximum(np.minimum(pos, 10), 1))
    oth = L.inv[(L.srt + n) % (2 * n)]
    oe_o = oe10[oth]
    cv = L.rows((~np.isnan(oe_o)).astype(float), 10)
    put("sos2_10", np.where(cv > 0, L.rows(oe_o, 10) / np.maximum(cv, 1), np.nan))

    bw, wl = np.full(len(pos), np.nan), np.full(len(pos), np.nan)
    aw, al = np.where(won == 1, opp, np.nan), np.where(won == 0, opp, np.nan)
    for k in range(1, 21):
        bw, wl = np.fmax(bw, L.shift(aw, k)), np.fmin(wl, L.shift(al, k))
    put("bestwin_20", bw); put("worstloss_20", wl)

    for name, win in (("elo_chg_90d", 90), ("elo_chg_365d", 365)):
        j = np.searchsorted(L.dkey, L.dkey - win, side="right") - 1
        ref = np.where(j >= st, own[np.maximum(j, 0)], own[st])
        put(name, np.where(first, np.nan, own - ref))

    x = (L.day - L.d0) / 30.0
    y = own - 1500.0
    lo = L.lo(180)
    k = (ar - lo) + 1
    sx, sy = L.dsum(x, 180) + x, L.dsum(y, 180) + y
    sxy, sxx = L.dsum(x * y, 180) + x * y, L.dsum(x * x, 180) + x * x
    den = k * sxx - sx ** 2
    ok = (k >= 4) & (L.day - L.day[lo] >= 21) & (den > 1e-9)
    put("elo_slope_180d", np.where(ok, (k * sxy - sx * sy) / np.where(ok, den, 1.0), np.nan))

    dd = np.where(first, 0.0, own - np.r_[0.0, own[:-1]])
    lo20 = np.maximum(ar - 19, st + 1)
    c = np.minimum(pos, 20)
    ex = L.excl(dd)
    ex2 = L.excl(dd * dd)
    s1, s2 = ex + dd - ex[np.minimum(lo20, ar)], ex2 + dd * dd - ex2[np.minimum(lo20, ar)]
    cc = np.maximum(c, 1)
    var = np.maximum(s2 / cc - (s1 / cc) ** 2, 0.0)
    put("elo_vol_20", np.where(c >= 3, np.sqrt(var), np.nan))
    put("elo_frompeak", own - pd.Series(own).groupby(L.gid).cummax().to_numpy())

    lam = LN2 / 120.0
    put("ewm_resid_120", L.wdsum(res, lam, 730) / (L.wdsum(ones, lam, 730) + 3.0))

    for name, cond in (("vs_higher_resid", opp >= own + 50), ("vs_lower_resid", opp <= own - 50)):
        cf = cond.astype(float)
        ct = L.rows(cf, 40)
        num, den = L.rows_decay(cf * res, 40, 20), L.rows_decay(cf, 40, 20)
        put(name, np.where(ct >= 3, num / (den + 4.0), 0.0))

    cnt90 = L.dsum(ones, 90)
    lam = 1.0 / 45.0
    gap = L.wdsum(opp, lam, 90) / np.maximum(L.wdsum(ones, lam, 90), 1e-12) - own
    put("sos_gap_90d", np.where(cnt90 >= 2, gap, 0.0))

    lam = 1.0 / 180.0
    good = won * (opp >= own) - (1 - won) * (opp <= own - 100)
    put("quality_win_rate_365", L.wdsum(good, lam, 365) / (L.dsum(ones, 365) + 3.0))

    # z-score of opponent Elo against the level group's pooled Elo over the previous 365 days
    mean_m, sd_m, ok_m = zst
    zl = np.where(np.r_[ok_m, ok_m], np.clip((opp_u - np.r_[mean_m, mean_m]) / np.r_[sd_m, sd_m], -3, 3), 0.0)
    wm = (won - 0.5) * L.g(zl)
    num = den = np.zeros(len(pos))
    for k in range(1, 11):
        v = (pos >= k) * 0.9 ** (k - 1)
        sh = L.shift(wm, k)
        num = num + v * np.nan_to_num(sh); den = den + v
    put("oqw_form_10", np.where(pos >= 3, num / np.maximum(den, 1e-12), 0.0))

    # same-band history: player x band of (own - opp Elo)
    band = np.searchsorted([-150.0, -50.0, 50.0, 150.0], own_u - opp_u, side="right")
    L2 = Long(pl * 5 + band, mi, dayl)
    r2, o2 = L2.g(res_u), np.ones(len(pl))
    lam = LN2 / 365.0
    c2_ = L2.dsum(o2, 1095)
    put("band_resid", np.where(c2_ >= 4, L2.wdsum(r2, lam, 1095) / (L2.wdsum(o2, lam, 1095) + 6.0), 0.0), L2)

    # residual history against opponents of the same tiebreak propensity bucket
    tb_u, cm_u = np.r_[G["tbs"], G["tbs"]], np.r_[G["cmp"], G["cmp"]]
    prop = (L.dsum(L.g(tb_u), 730) + 1.2) / (L.dsum(L.g(cm_u), 730) + 10.0)
    prop_u = L.back(prop)
    bucket = (prop_u[(np.arange(2 * n) + n) % (2 * n)] > 0.20).astype(int)
    L3 = Long(pl * 2 + bucket, mi, dayl)
    r3, o3 = L3.g(res_u), np.ones(len(pl))
    c3 = L3.dsum(o3, 730)
    put("res_vs_high_tb_opp", np.where(c3 >= 5, L3.dsum(r3, 730) / (c3 + 5.0), 0.0), L3)

    # head to head
    lo_id, hi_id = np.minimum(w, l), np.maximum(w, l)
    pk = lo_id.astype(np.int64) * G["nP"] + hi_id
    wl_ = w == lo_id
    won_lo = wl_.astype(float)
    r_lo = won_lo - np.where(wl_, pw, 1 - pw)
    Lp = Long(pk, np.arange(n), day)
    Ls = Long(pk * 4 + G["surf"], np.arange(n), day)
    h2h = Lp.back(Lp.excl(Lp.g(r_lo)) / (Lp.pos + 2.0))
    out["w_h2h_resid"], out["l_h2h_resid"] = np.where(wl_, h2h, -h2h).astype(np.float32), np.where(wl_, -h2h, h2h).astype(np.float32)
    lam = log(1.25) / 365.25
    o_ = np.ones(n)
    sa = Ls.back(Ls.wdsum(Ls.g(won_lo), lam)); ss = Ls.back(Ls.wdsum(Ls.g(o_), lam))
    aa = Lp.back(Lp.wdsum(Lp.g(won_lo), lam)); sw_ = Lp.back(Lp.wdsum(Lp.g(o_), lam))
    p_lo = (0.6 * aa + 0.4 * sa + 0.5) / (0.6 * sw_ + 0.4 * ss + 1.0)
    pwin = np.where(wl_, p_lo, 1 - p_lo)
    out["w_h2h_decay_shrunk"], out["l_h2h_decay_shrunk"] = pwin.astype(np.float32), (1 - pwin).astype(np.float32)
    return out


def _task_sim():
    G = _G
    n = G["n"]
    w, l, day, pw, ew, el = G["w"], G["l"], G["day"], G["pw"], G["ew"], G["el"]
    mi = np.r_[np.arange(n), np.arange(n)]
    L = Long(np.r_[w, l], mi, np.r_[day, day])
    opp = L.g(np.r_[el, ew]); res = L.g(np.r_[1 - pw, -(1 - pw)])
    lo = L.lo(730)
    val = np.zeros(len(opp))
    for q in np.flatnonzero((L.ar - lo >= 5) & (L.g(mi) >= G["i0"])):
        s = slice(lo[q], q)
        m = np.abs(opp[s] - opp[q]) <= 100.0
        c = m.sum()
        if c >= 5:
            val[q] = res[s][m].sum() / (c + 5.0)
    u = L.back(val)
    return {"w_sim_opp_resid": u[:n].astype(np.float32), "l_sim_opp_resid": u[n:].astype(np.float32)}


TASKS = [_task_g2, _task_co, _task_pr, _task_ts_lvl, _task_cs_layoff_mf, _task_long, _task_hind, _task_sim]
# every input of the saved states (a saved state is used only while these values of its rows are unchanged)
STATE_COLS = ["winner_id", "loser_id", "date", "order", "surface", "tourney_level", "score", "retired", "main"]
# the pool gets the longest tasks first (ATP times with live 'need': cs 16s, pr 10, g2 9, long 8, co 7, ts 4, hind 1, sim 0)
LONGEST_FIRST = [4, 2, 0, 5, 1, 3, 6, 7]


def _run(i):
    n, i0 = _G["n"], _G["i0"]
    # rows before i0 are not wanted: keep them out of the pipe (the sequential tasks return only rows i0..n)
    return {k: (v[i0:] if len(v) == n else v) for k, v in TASKS[i]().items()}


def _excluded(M):
    """Rows (in 'order') that must not feed any state: self-play rows (winner_id == loser_id) and exact twins (same date,
    winner, loser and score as an earlier row; the source files list some matches twice, so a twin would see its own
    result as history). Returns (self-play mask, twin mask, position of the first copy for every row)."""
    sp = M["winner_id"].values == M["loser_id"].values
    key = pd.DataFrame({"d": M["date"].values, "w": M["winner_id"].values, "l": M["loser_id"].values,
                        "s": M["score"].fillna("").astype(str).values})
    code = key.groupby(["d", "w", "l", "s"], sort=False).ngroup().values
    first = np.unique(code, return_index=True)[1][code]
    return sp, (first != np.arange(len(M))) & ~sp, first


def _with_exclusions(M, run, need=None):
    """run(M_clean, need_clean) -> frame; self-play rows get NaN, twins get the features of their first copy."""
    sp, twin, first = _excluded(M)
    if need is not None:
        need = need.copy()
        need[first[need & twin]] = True  # a needed twin takes its first copy's features
    if not (sp | twin).any():
        return run(M, need)
    F = run(M[~(sp | twin)], None if need is None else need[~(sp | twin)]).reindex(M.index)
    di = np.flatnonzero(twin)
    F.iloc[di] = F.iloc[first[di]].values
    return F


def compute(M, workers=None, need=None, state=None):
    """Features for every row of M; returns a float32 frame with M's index (rows may come in any order, 'order' decides).
    Self-play rows get NaN and exact twin rows take the features of their first copy, see _excluded.
    need: index labels of the rows whose features are wanted (e.g. live fixtures at the end of the table); earlier rows
    then only feed the state, and the slow per-row parts of some tasks are skipped for them (their values are not valid).
    state: file prefix for the saved state of the sequential tasks at the first wanted row (with need only); the next
    call continues from it when the rows it has seen are still the first rows of the table."""
    o = M["order"].values
    if len(o) > 1 and not (np.diff(o) > 0).all():
        return compute(M.iloc[np.argsort(o, kind="stable")], workers, need, state).loc[M.index]
    mask = None if need is None else M.index.isin(need)
    return _with_exclusions(M, lambda X, m: _compute(X, workers, m, state), mask)


def _compute(M, workers=None, need=None, state=None):
    perm = np.argsort(M["order"].values, kind="stable")
    Ms = M.iloc[perm] if (perm != np.arange(len(M))).any() else M
    _G.clear()
    _G.update(_prep(Ms))
    # first row whose features are wanted: rows before it only update the state in the tasks that check i0
    _G["i0"] = 0 if need is None else (int(np.argmax(need[perm])) if need.any() else len(Ms))
    if state and need is not None:
        _G["state"] = state
        _G["rowhash"] = pd.util.hash_pandas_object(Ms[[c for c in STATE_COLS if c in Ms]], index=False).values
    _std_elo(_G)
    workers = workers or int(os.environ.get("FX_WORKERS", min(3, os.cpu_count() or 1)))
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
            done = dict(zip(LONGEST_FIRST, pool.map(_run, LONGEST_FIRST)))
        parts = [done[i] for i in range(len(TASKS))]
    else:
        parts = [_run(i) for i in range(len(TASKS))]
    cols = {}
    for p in parts:
        cols.update(p)
    i0 = _G["i0"]
    out = pd.DataFrame(cols, index=Ms.index[i0:]).reindex(Ms.index) if i0 else pd.DataFrame(cols, index=Ms.index)
    return out.loc[M.index] if Ms is not M else out


if __name__ == "__main__":
    cols = ["winner_id", "loser_id", "date", "surface", "tourney_level", "score", "retired", "main", "order"]  # duplicates are found on date, ids, score
    F = compute(pd.read_parquet(sys.argv[1], columns=cols))
    F.to_parquet(sys.argv[2])
    print(F.shape, "->", sys.argv[2])
