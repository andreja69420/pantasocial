"""New-feature modules (fx_ratings / fx_score / fx_context / fx_pbp) for rows that are not in the build yet, e.g.
upcoming fixtures with placeholder results: each batch (features_incr.py DRY_MATCHES, no player twice in a batch) is
appended to the build's whole match table and the modules run on it; only the new rows are kept. Every module feature
of a row uses earlier rows only, so a row's placeholder result never reaches its own features, and batches keep one
fixture's placeholder away from another fixture. Columns come out as w_nx_<name> / l_nx_<name>, like fx_merge.py.

usage: fx_live.py OUT_DIR TOUR NEW_MATCHES_PARQUET OUT_PARQUET
"""
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import fx_context
import fx_pbp
import fx_ratings
import fx_score

MODS = {"ratings": fx_ratings.compute, "score": fx_score.compute, "context": fx_context.compute, "pbp": fx_pbp.compute}
ID = ["winner_id", "loser_id", "date", "order"]
# the match columns each module reads: the table is loaded with their union only, and each module gets its own subset
COLS = {"ratings": ID + ["surface", "tourney_level", "score", "retired", "main"],
        "score": ID + ["tourney_id", "tourney_level", "score", "retired", "main", "best_of"],
        "context": ID + ["tourney_id", "tourney_name", "surface", "tourney_level", "round", "rord", "score", "retired",
                         "winner_ioc", "loser_ioc", "winner_age", "loser_age"],
        "pbp": ID + [f"{s}_pbp_{c}" for s in ("w", "l") for c in ("svpt", "svwon", "SvGms", "holds", "bpFaced", "bpSaved", "tbpt", "tbwon")]}
# two worker processes: ratings (the slowest, with its own pool inside) and the other three one after another, so at most
# two modules hold their working memory at a time
GROUPS = [["ratings"], ["score", "context", "pbp"]]
_M = {}


def _run(group):
    out = []
    for name in group:
        M = _M["M"][[c for c in COLS[name] if c in _M["M"]]]
        out.append(fx_score.compute(M, _M["wta"]) if name == "score" else fx_context.compute(M, _M["wta"]) if name == "context"
                   else MODS[name](M))
    return out


def features(base, new, wta):
    _M.update(M=pd.concat([base, new[[c for c in base.columns if c in new.columns]]]), wta=wta)
    # fork: the workers see the table without pickling it; fx_ratings starts its own pool inside its worker
    with ProcessPoolExecutor(len(GROUPS), mp_context=mp.get_context("fork")) as pool:
        parts = [p for g in pool.map(_run, GROUPS) for p in g]
    X = pd.concat([p.loc[new.index] for p in parts], axis=1)
    return X.rename(columns=lambda c: f"{c[:2]}nx_{c[2:]}").astype("float32")


def load_base(out, tour):
    path = os.path.join(out, f"{tour}_matches.parquet")
    have = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in dict.fromkeys(c for cs in COLS.values() for c in cs) if c in have])


if __name__ == "__main__":
    OUT, TOUR, NEW, DST = sys.argv[1:5]
    t0 = time.time()
    base = load_base(OUT, TOUR)
    new = pd.read_parquet(NEW)
    assert not set(new.index) & set(base.index), "new rows reuse index values of the build"
    wta = TOUR == "wta"
    X = pd.concat([features(base, new[new.batch == b], wta) for b in np.unique(new.batch)]).loc[new.index]
    X.to_parquet(DST)
    print(f"{TOUR}: {X.shape[1] // 2} new features per player for {len(X)} rows in {time.time() - t0:.0f}s -> {DST}", flush=True)
