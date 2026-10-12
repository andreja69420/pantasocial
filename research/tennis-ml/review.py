"""Manual review of the bets paper_trade.py proposes (REVIEW=1): nothing reaches the ledger before someone has looked at
both players.

  review.py DATA_DIR                    pending proposals: the bet, every book's price, model vs book, and per player
                                        the ids and names of the history, matches, last match, bio and the last 10 results
  review.py DATA_DIR ok FS_MID [NOTE]   into the ledger (logged now); ledger copied to paper/ and the Excel rebuilt
  review.py DATA_DIR no FS_MID REASON   rejected: to paper/skipped.csv with the reason
A proposal whose match has already started is rejected on its own when the list is shown.
"""
import os
import subprocess
import sys

import pandas as pd

D, CMD = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "list")
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(D, "paper")
PROP, LEDGER, SKIPPED = (os.path.join(PAPER, f) for f in ("proposals.csv", "ledger.csv", "skipped.csv"))
LIVE_OUT = os.environ.get("LIVE_OUT", "out_live2")
GEO = os.environ.get("GEO", "RS")
BOOKS = os.environ.get("BOOKS", "bet365,Mozzartbetrs,Admiralbet.rs,SoccerBet,Superbet.rs,1xBet.rs").split(",")
now = pd.Timestamp.now("UTC").tz_localize(None)
P = pd.read_csv(PROP) if os.path.exists(PROP) else pd.DataFrame(columns=["fs_mid"])


def save_props(df):
    df.to_csv(PROP, index=False)


def reject(rows, reason):
    out = rows.assign(reason=reason, seen_at=now)[["fs_mid", "home", "away", "start", "side", "book", "odds", "q_model", "ev",
                                                    "reason", "seen_at"]]
    out = pd.concat([pd.read_csv(SKIPPED), out]) if os.path.exists(SKIPPED) else out
    out.to_csv(SKIPPED, index=False)


def publish():
    import shutil
    shutil.copy(LEDGER, os.path.join(HERE, "paper", "ledger.csv"))
    subprocess.run([sys.executable, os.path.join(HERE, "ledger_xlsx.py"), LEDGER, os.path.join(HERE, "paper", "opklade.xlsx")],
                   cwd=HERE, check=True, capture_output=True)


if CMD in ("ok", "no"):
    # a pass reads the ledger and the proposals at its start and writes them at its end: never write in between
    if any("paper_trade.py" in a for a in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()):
        sys.exit("a paper_trade.py pass is running; try again when it has finished")
    mid = sys.argv[3]
    row = P[P.fs_mid == mid]
    if row.empty:
        sys.exit(f"{mid} is not a pending proposal")
    if CMD == "ok":
        L = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame()
        add = row.drop(columns=["id_home", "id_away"]).assign(logged_at=now, review=" ".join(sys.argv[4:]) or "ok")
        # every book's current price for our side right away, so the "Sve kvote" sheet is not empty until the next pass
        r0 = add.iloc[0]
        fx = pd.read_parquet(os.path.join(D, "fs_raw", f"{r0.tour}_fs_fixture_odds_{GEO}.parquet"))
        fx = fx[fx.fs_mid == mid]
        for b in fx.itertuples():
            if b.book in BOOKS:
                add[f"px_{b.book}"] = b.cur_h if r0.side == "home" else b.cur_a
        L = pd.concat([L, add], ignore_index=True)
        L.to_csv(LEDGER, index=False)
        publish()
    else:
        reject(row, "manual review: " + " ".join(sys.argv[4:]))
    save_props(P[P.fs_mid != mid])
    print(f"{mid}: {'logged' if CMD == 'ok' else 'rejected'}")
    sys.exit()

if P.empty:
    print("no pending proposals")
    sys.exit()
P["start"] = pd.to_datetime(P.start)
late = P[P.start <= now]
if len(late):
    reject(late, "match started before the review")
    P = P[P.start > now]
    save_props(P)
    print(f"{len(late)} proposals rejected: the match started before the review")
fx = {t: pd.read_parquet(os.path.join(D, "fs_raw", f"{t}_fs_fixture_odds_{GEO}.parquet")) for t in P.tour.unique()}
for t, g in P.groupby("tour"):
    base = pd.read_parquet(os.path.join(D, LIVE_OUT, f"{t}_matches.parquet"))
    for r in g.itertuples():
        bg = r.start.tz_localize("UTC").tz_convert("Europe/Belgrade")
        us = r.home if r.side == "home" else r.away
        print("=" * 110)
        print(f"{r.fs_mid}  {bg:%d.%m. %H:%M}  {r.tournament}")
        print(f"bet: {us} @ {r.odds} {r.book} | model {r.q_model:.1%}, book fair {r.fair:.1%}, blend {r.q_blend:.1%}, EV {r.ev:+.1%}")
        o = fx[t][fx[t].fs_mid == r.fs_mid]
        for b in o.itertuples():
            print(f"   {b.book:14s} open {b.open_h:5.2f} {b.open_a:5.2f}   now {b.cur_h:5.2f} {b.cur_a:5.2f}")
        for side, pid in (("home", r.id_home), ("away", r.id_away)):
            m = base[(base.winner_id == pid) | (base.loser_id == pid)].sort_values("date")
            names = sorted(set(m.loc[m.winner_id == pid, "winner_name"]) | set(m.loc[m.loser_id == pid, "loser_name"]))
            print(f"-- {side}: {getattr(r, side)}  id {pid}{' (SYNTHETIC)' if pid >= 9e8 else ''}  names in history: {names}")
            if m.empty:
                print("   no matches")
                continue
            last = m.iloc[-1]
            pre = "winner" if last.winner_id == pid else "loser"
            print(f"   {len(m)} matches {m.date.min():%Y-%m-%d} .. {m.date.max():%Y-%m-%d}, {(m.date >= now - pd.Timedelta(days=365)).sum()}"
                  f" in the last 12 months, {(m.winner_id == pid).mean():.0%} won; age {last[pre + '_age']}, {last[pre + '_ioc']},"
                  f" hand {last[pre + '_hand']}")
            for x in m.tail(10).itertuples():
                w = x.winner_id == pid
                print(f"   {x.date:%Y-%m-%d} {str(x.tourney_name)[:24]:24s} {str(x.tourney_level):3s} {str(x.surface)[:5]:5s} "
                      f"{'W' if w else 'L'} vs {(x.loser_name if w else x.winner_name)[:28]:28s} {x.score}")
