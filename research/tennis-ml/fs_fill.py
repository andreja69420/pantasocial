"""Turn the matches fs_audit.py found missing into raw rows for the next build.

For every season/tour in AUDIT_DIR/{year}/ the missing Flashscore matches (audit_{tour}_missing.csv) are
  - dropped when they are team events (Davis Cup, Billie Jean King Cup, United Cup, Laver Cup, ATP Cup, Hopman Cup),
  - dropped when the same pair of players (ids or names, same winner and loser) is in the build within 120 days: the
    audit's 30-day window can miss a match whose event date differs, adding it again would duplicate it,
  - attached to the build's event when Flashscore's event is already in the build (tourney_id, name, date, level and
    surface of the build, so ordering inside the event and the level categories stay the Sackmann ones); events the
    build lacks keep their Flashscore ids (fs-<slug>-<year>).
Writes RAW_OUT/sackmann_raw/{tour}__{tour}_matches_fsfill_{year}.csv next to symlinks to every file of RAW_IN.

usage: fs_fill.py BUILD_DIR AUDIT_DIR RAW_IN RAW_OUT [--years=2010,2011,..] [--dry]   (default 2023..2026; a fill file that
already exists is extended, rows with a Flashscore id already in it are not repeated)
"""
import glob
import os
import re
import shutil
import sys
import unicodedata

import numpy as np
import pandas as pd

BUILD, AUDIT, RAW_IN, RAW_OUT = sys.argv[1:5]
DRY = "--dry" in sys.argv
FRESH = "--fresh" in sys.argv  # write the fill files of these years anew (and reset their main-file copies) instead of extending them
TEAM = re.compile(r"davis cup|billie jean king|united cup|laver cup|atp cup|hopman", re.I)
norm = lambda s: "".join(sorted(re.findall(r"[a-z]+", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())))  # order-free: "Thompson Jordan" == "Jordan Thompson"
sc = lambda s: re.sub(r"\(\d+\)|\s+|RET.*$", "", str(s))  # the same sets whatever the tie-break detail ("7-6(5)" == "7-6") and the retirement tail
td = lambda s: pd.to_datetime(pd.to_numeric(s, errors="coerce").astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
YEARS = tuple(int(v) for v in next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--years=")), "2023,2024,2025,2026").split(","))
os.makedirs(f"{RAW_OUT}/sackmann_raw", exist_ok=True)
if not DRY:
    for f in glob.glob(f"{RAW_IN}/sackmann_raw/*"):
        dst = f"{RAW_OUT}/sackmann_raw/{os.path.basename(f)}"
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(f), dst)
    for f in glob.glob(f"{RAW_IN}/*"):
        if os.path.basename(f) != "sackmann_raw" and not os.path.lexists(f"{RAW_OUT}/{os.path.basename(f)}"):
            os.symlink(os.path.abspath(f), f"{RAW_OUT}/{os.path.basename(f)}")



def build_index(B):
    """(score, key) -> event dates of the build rows, for the winner id, loser id and both name keys."""
    idx = {}
    for col, f in (("winner_id", str), ("loser_id", str), ("winner_name", norm), ("loser_name", norm)):
        for s_, k, d in zip(B.score.map(sc), B[col].map(f), B.td.values):
            idx.setdefault((s_, col[:1], k), []).append(d)
    idx = {k: tuple(v) for k, v in idx.items()}
    return idx


def is_dup(F, idx, days=45):
    """The match is in the build already: same score within DAYS and the winner and the loser both match (id or name), or
    one of them matches and the other has a made-up id (> 900000: a player Flashscore names differently from Sackmann)."""
    out = np.zeros(len(F), bool)
    lim = np.timedelta64(days, "D")
    ok = lambda s_, side, ident, name, d: any(abs(e - d) <= lim for e in idx.get((s_, side, ident), ()) + idx.get((s_, side, name), ()))
    for i, (s_, wi, li, wn, ln, d) in enumerate(zip(F.score.map(sc), F.winner_id.astype(str), F.loser_id.astype(str),
                                                    F.winner_name.map(norm), F.loser_name.map(norm), F.td.values)):
        w_ok, l_ok = ok(s_, "w", wi, wn, d), ok(s_, "l", li, ln, d)
        w_new, l_new = int(float(wi)) > 900000, int(float(li)) > 900000
        out[i] = (w_ok and l_ok) or (w_ok and l_new) or (l_ok and w_new)
    return out


INDEX = {}
if "--clean" in sys.argv:
    # BUILD_DIR is a build made WITHOUT the fill: drop fill rows that are in it already (same score, same event window, same
    # winner or loser by id or name) from the existing fsfill files
    for tour in ("atp", "wta"):
        B = pd.read_parquet(f"{BUILD}/{tour}_matches.parquet", columns=["tourney_date", "score", "winner_id", "loser_id", "winner_name", "loser_name"])
        B["td"] = td(B.tourney_date)
        idx = build_index(B[B.td >= "2009-08-01"])
        for f in sorted(glob.glob(f"{RAW_OUT}/sackmann_raw/{tour}__{tour}_matches_fsfill_*.csv")):
            F = pd.read_csv(f, low_memory=False)
            F["td"] = td(F.tourney_date)
            d = is_dup(F, idx)
            print(f"{os.path.basename(f)}: {len(F)} rows, {int(d.sum())} already in the build -> dropped", flush=True)
            F[~d].drop(columns=["td"]).to_csv(f, index=False)
    sys.exit(0)

total = {}
for tour in ("atp", "wta"):
    B = pd.read_parquet(f"{BUILD}/{tour}_matches.parquet",
                        columns=["tourney_id", "tourney_name", "tourney_level", "tourney_date", "surface", "draw_size", "main", "score",
                                 "winner_id", "loser_id", "winner_name", "loser_name"])
    B["td"] = td(B.tourney_date)
    B = B[B.td >= f"{min(YEARS) - 1}-08-01"]
    pair = {}
    for k, d in zip(B.winner_id.astype(str) + "|" + B.loser_id.astype(str), B.td.values):
        pair.setdefault(k, []).append(d)
    nm = {}
    for k, d in zip(B.winner_name.map(norm) + "|" + B.loser_name.map(norm), B.td.values):
        nm.setdefault(k, []).append(d)
    near = lambda idx, k, d: any(abs(e - d) <= np.timedelta64(120, "D") for e in idx.get(k, ()))
    for y in YEARS:
        FS = pd.read_csv(f"{AUDIT}/{y}/{tour}__{tour}_matches_fs_{y}.csv", low_memory=False)
        M = pd.read_csv(f"{AUDIT}/{y}/audit_{tour}_missing.csv", low_memory=False)
        n0 = len(M)
        M = M[~M.tourney_name.astype(str).str.contains(TEAM)]
        n_team = n0 - len(M)
        M["td"] = td(M.tourney_date)
        dup = np.array([near(pair, f"{w}|{l}", d) or near(nm, f"{norm(a)}|{norm(b)}", d) for w, l, a, b, d in
                        zip(M.winner_id.astype(str), M.loser_id.astype(str), M.winner_name, M.loser_name, M.td.values)], bool)
        IDX = INDEX.setdefault(tour, build_index(B))
        dup = dup | is_dup(M, IDX)
        M = M[~dup]
        # Flashscore event -> the build's event, from the matches of that event the build has
        F = FS[~FS.fs_mid.isin(set(pd.read_csv(f"{AUDIT}/{y}/audit_{tour}_missing.csv", low_memory=False).fs_mid))].copy()
        F["td"] = td(F.tourney_date)
        ev = B.drop_duplicates("tourney_id").set_index("tourney_id")
        idx = {}
        for tid, k, d in zip(B.tourney_id, B.winner_id.astype(str) + "|" + B.loser_id.astype(str), B.td.values):
            idx.setdefault(k, []).append((d, tid))
        votes = {}
        for fid, k, d in zip(F.tourney_id, F.winner_id.astype(str) + "|" + F.loser_id.astype(str), F.td.values):
            for e, tid in idx.get(k, ()):
                if abs(e - d) <= np.timedelta64(30, "D"):
                    votes.setdefault(fid, []).append(tid)
                    break
        emap = {fid: pd.Series(v).mode()[0] for fid, v in votes.items()}
        att = M.tourney_id.isin(emap)
        for col in ("tourney_name", "tourney_level", "surface", "draw_size", "tourney_date"):
            M.loc[att, col] = M.loc[att, "tourney_id"].map(emap).map(ev[col])
        # rows of an event the build holds in the Sackmann main file (main=1) go into that year's main file again, so the
        # main-draw ratings treat them like their event; qualifying rows and events the build lacks stay in the fsfill file
        main = (att & ~M["round"].astype(str).str.startswith("Q")).values.copy()
        main[main] = (M.loc[main, "tourney_id"].map(emap).map(ev["main"]).fillna(0).astype(int) == 1).values
        M.loc[att, "tourney_id"] = M.loc[att, "tourney_id"].map(emap)
        out = M.drop(columns=["td"])
        total[(tour, y)] = (n0, n_team, int(dup.sum()), len(out), int(att.sum()))
        print(f"{tour} {y}: main-file rows {int(main.sum())}; missing {n0}, team events {n_team}, duplicates of build rows {int(dup.sum())} -> fill {len(out)} "
              f"({int(att.sum())} into events the build has, {len(out) - int(att.sum())} into events it lacks)", flush=True)
        if FRESH and not DRY:
            for yr in {y}:
                mdst = f"{RAW_OUT}/sackmann_raw/{tour}__{tour}_matches_{yr}.csv"
                if os.path.exists(mdst) and not os.path.islink(mdst):
                    shutil.copyfile(f"{RAW_IN}/sackmann_raw/{tour}__{tour}_matches_{yr}.csv", mdst)
            fdst0 = f"{RAW_OUT}/sackmann_raw/{tour}__{tour}_matches_fsfill_{y}.csv"
            if os.path.exists(fdst0):
                pd.DataFrame(columns=FS.columns).to_csv(fdst0, index=False)
        if not DRY and len(out):
            fdst = f"{RAW_OUT}/sackmann_raw/{tour}__{tour}_matches_fsfill_{y}.csv"
            part = out[~main]
            if os.path.exists(fdst) and not FRESH:
                part = pd.concat([pd.read_csv(fdst, low_memory=False), part], ignore_index=True).drop_duplicates("fs_mid")
            part.to_csv(fdst, index=False)
            for yr, g in out[main].groupby(out[main].tourney_date.astype(str).str[:4].astype(int)):
                dst = f"{RAW_OUT}/sackmann_raw/{tour}__{tour}_matches_{yr}.csv"
                src = f"{RAW_IN}/sackmann_raw/{tour}__{tour}_matches_{yr}.csv"
                base = pd.read_csv(src if FRESH or os.path.islink(dst) else dst, low_memory=False)
                if os.path.islink(dst):
                    os.remove(dst)
                allr = pd.concat([base, g.reindex(columns=base.columns.union(g.columns, sort=False))], ignore_index=True)
                allr[~(allr.fs_mid.notna() & allr.duplicated("fs_mid"))].to_csv(dst, index=False)
                print(f"   {int(main.sum())} rows added to {os.path.basename(dst)}")
        if len(out):
            print("   by level:", out.tourney_level.astype(str).value_counts().to_dict())
T = pd.DataFrame(total, index=["missing", "team", "dup", "fill", "into_existing"]).T
print(T.groupby(level=0).sum().to_string()); print("all:", T.sum().to_dict())
