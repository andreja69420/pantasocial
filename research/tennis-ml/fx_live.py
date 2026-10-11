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

import fx_context
import fx_pbp
import fx_ratings
import fx_score

MODS = {"ratings": fx_ratings.compute, "score": fx_score.compute, "context": fx_context.compute, "pbp": fx_pbp.compute}
_M = {}


def _run(name):
    M = _M["M"]
    if name == "score":
        return fx_score.compute(M, _M["wta"])
    if name == "context":
        return fx_context.compute(M, _M["wta"])
    return MODS[name](M)


def features(base, new, wta):
    _M.update(M=pd.concat([base, new[[c for c in base.columns if c in new.columns]]]), wta=wta)
    # fork: the workers see the table without pickling it; fx_ratings starts its own pool inside its worker
    with ProcessPoolExecutor(len(MODS), mp_context=mp.get_context("fork")) as pool:
        parts = list(pool.map(_run, MODS))
    X = pd.concat([p.loc[new.index] for p in parts], axis=1)
    return X.rename(columns=lambda c: f"{c[:2]}nx_{c[2:]}").astype("float32")


if __name__ == "__main__":
    OUT, TOUR, NEW, DST = sys.argv[1:5]
    t0 = time.time()
    base = pd.read_parquet(os.path.join(OUT, f"{TOUR}_matches.parquet"))
    new = pd.read_parquet(NEW)
    assert not set(new.index) & set(base.index), "new rows reuse index values of the build"
    wta = TOUR == "wta"
    X = pd.concat([features(base, new[new.batch == b], wta) for b in np.unique(new.batch)]).loc[new.index]
    X.to_parquet(DST)
    print(f"{TOUR}: {X.shape[1] // 2} new features per player for {len(X)} rows in {time.time() - t0:.0f}s -> {DST}", flush=True)
