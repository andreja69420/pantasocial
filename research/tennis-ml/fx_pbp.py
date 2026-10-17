"""Serve / return features from Flashscore point-by-point counts (w_pbp_* / l_pbp_* columns added by fs_join.py,
mostly ITF 2023 on, where there is no box score). Strictly pre-match: each row uses the player's last N earlier
matches (smaller 'order') that have point-by-point data. Ratios are shrunk toward the pre-2025 population mean,
so a player with few such matches sits near the mean; pbp_n_30 says how many there were.

usage: fx_pbp.py MATCHES_PARQUET OUT_PARQUET
"""
import sys

import numpy as np
import pandas as pd

PRIOR_END = np.datetime64("2025-01-01")


def last_k(gcode, order, has, v, k):
    """Sum of v over each row's last k earlier rows of the same group with has=1 (rows in any order)."""
    idx = np.lexsort((order, gcode))
    g, h, x = gcode[idx], has[idx].astype(np.int64), np.where(has, v, 0.0)[idx]
    start = np.r_[True, g[1:] != g[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)[gid]
    ch = np.cumsum(h) - h  # earlier has-rows, global
    cv = np.cumsum(x) - x
    ch_g = ch - ch[first]  # earlier has-rows within the group
    key = gid.astype(np.int64) * (len(g) + 1) + ch_g
    j = np.searchsorted(key, gid.astype(np.int64) * (len(g) + 1) + np.maximum(ch_g - k, 0), side="left")
    out = np.empty(len(g))
    out[idx] = cv - cv[j]
    return out


def compute(M, need=None):
    """need: index labels of the rows whose features are wanted; the windows then run over the histories of their players
    only (the population means stay over all rows), other rows get NaN."""
    n = len(M)
    num = lambda c: pd.to_numeric(M[c], errors="coerce").values.astype(float) if c in M else np.full(n, np.nan)
    pc, _ = pd.factorize(np.concatenate([M.winner_id.astype(str).values, M.loser_id.astype(str).values]))
    order = np.r_[M.order.values, M.order.values].astype(np.int64)
    early = np.r_[M.date.values, M.date.values].astype("datetime64[D]") < PRIOR_END

    def side(me, op):
        g = lambda s, c: num(f"{s}_pbp_{c}")
        d = {"svpt": g(me, "svpt") + g(me, "tbpt"), "svwon": g(me, "svwon") + g(me, "tbwon"),
             "rtpt": g(op, "svpt") + g(op, "tbpt"), "svg": g(me, "SvGms"), "held": g(me, "holds"),
             "bpf": g(me, "bpFaced"), "bps": g(me, "bpSaved"), "rtg": g(op, "SvGms"), "bpc": g(op, "bpFaced"),
             "tbpt": g(me, "tbpt") + g(op, "tbpt")}
        d["rtwon"] = d["rtpt"] - (g(op, "svwon") + g(op, "tbwon"))
        d["brk"] = d["rtg"] - g(op, "holds")
        d["bpw"] = d["bpc"] - g(op, "bpSaved")
        d["tbwon"] = g(me, "tbwon") + g(op, "tbpt") - g(op, "tbwon")
        return d

    a, b = side("w", "l"), side("l", "w")
    L = {k: np.r_[a[k], b[k]] for k in a}
    hasG = (np.nan_to_num(L["svg"]) > 0) & (np.nan_to_num(L["rtg"]) > 0)
    hasP = hasG & (np.nan_to_num(L["svpt"]) > 0) & (np.nan_to_num(L["rtpt"]) > 0)
    hasT = hasP & (np.nan_to_num(L["tbpt"]) > 0)
    flag = {"svpt": hasP, "svwon": hasP, "rtpt": hasP, "rtwon": hasP, "tbpt": hasT, "tbwon": hasT}

    keep = slice(None)
    if need is not None:
        m = M.index.isin(need)
        keep = np.isin(pc, pc[np.r_[m, m]])  # every row of the players of the wanted rows

    def last_k_(has, v, k):
        out = np.full(2 * n, np.nan)
        out[keep] = last_k(pc[keep], order[keep], has[keep], v[keep], k)
        return out

    def sums(c, k):
        return last_k_(flag.get(c, hasG), np.nan_to_num(L[c]), k)

    def sr(num_c, den_c, k, m):
        h = flag.get(num_c, hasG) & early
        p0 = np.nansum(L[num_c][h]) / np.nansum(L[den_c][h])
        x, d = sums(num_c, k), sums(den_c, k)
        return (x + m * p0) / (d + m), p0

    F = {}
    for k in (10, 30):
        F[f"pbp_sv_won_{k}"], _ = sr("svwon", "svpt", k, 60)
        F[f"pbp_rt_won_{k}"], _ = sr("rtwon", "rtpt", k, 60)
        F[f"pbp_hold_{k}"], _ = sr("held", "svg", k, 10)
        F[f"pbp_break_{k}"], _ = sr("brk", "rtg", k, 10)
        F[f"pbp_bp_saved_{k}"], _ = sr("bps", "bpf", k, 10)
        F[f"pbp_bp_conv_{k}"], _ = sr("bpw", "bpc", k, 10)
        F[f"pbp_dom_{k}"] = F[f"pbp_rt_won_{k}"] / np.maximum(1 - F[f"pbp_sv_won_{k}"], 1e-3)
        F[f"pbp_spw_rpw_{k}"] = F[f"pbp_sv_won_{k}"] + F[f"pbp_rt_won_{k}"]
    F["pbp_tb_pt_won_30"], _ = sr("tbwon", "tbpt", 30, 20)
    F["pbp_bp_faced_per_game_30"], _ = sr("bpf", "svg", 30, 10)
    F["pbp_n_30"] = last_k_(hasG, np.ones(2 * n), 30)
    F["pbp_pts_n_30"] = last_k_(hasP, np.ones(2 * n), 30)
    out = pd.DataFrame({f"{p}_{k}": v[s] for k, v in F.items() for p, s in (("w", slice(0, n)), ("l", slice(n, 2 * n)))},
                       index=M.index)
    return out.astype(np.float32)


if __name__ == "__main__":
    compute(pd.read_parquet(sys.argv[1])).to_parquet(sys.argv[2])
