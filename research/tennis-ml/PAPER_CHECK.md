# Paper-trading check (scheduled task, six times an hour: :04 :14 :24 :34 :44 :54)

The user wants the paper-trading loop (`paper_loop.sh`) to run every 10 minutes. The container is shut down when the session is
idle, which kills the loop, so these checks make sure a pass actually runs. `B=/tmp/claude-0/-home-user-pantasocial/d746d33b-0578-5ee2-a55c-5a9f85958b0a/scratchpad/tennis`.
Work in `research/tennis-ml`.

Production setup: `models_v7` (tag v7, the `paper_loop.sh` default; earlier bets were `models_v3` tag novi and `models_v6` tag v6),
Flashscore region RS, six books (bet365, MozzartBet, AdmiralBet, SoccerBet, Superbet, 1xBet), no ITF, current prices,
**EV>3% odds<=5 since 2026-10-18** (EV>5% before), SKIP_SYNTHETIC=1, REVIEW=1, pass every 10 min with a prediction cache.

1. If `$B` is gone (container reset), tell the user the data has to be rebuilt and stop.
2. If `paper_loop.sh` is not running (`ps -eo pid,etime,args | grep "[p]aper_loop.sh" | grep -v "bash -c"`), start it:
   `cd research/tennis-ml && setsid nohup ./paper_loop.sh $B >/dev/null 2>&1 </dev/null &` (it starts a pass right away).
   Never start a second loop; kill a loop by PID only (never `pkill -f`, it matches the own shell).
   If the loop was already running and its last `paper trading run` line in `$B/paper/loop.log` is younger than 12 min, skip the wait.
3. Otherwise WAIT in this turn until the pass finishes
   (`timeout 450 bash -c "tail -n0 -f $B/paper/loop.log | grep -m1 'paper trading run'"`, then ~25 s for the Excel), because the
   container may be shut down as soon as the turn ends.
4. Check the log of that pass for errors (`Traceback`, `fx_live failed`, `features_incr failed`, `no production model`) and fix
   them. Note `not logged` lines (bets the automatic data checks stopped).
5. REVIEW, two parts (user's rules since 2026-10-13). Run `python3 review.py $B` (lists each proposal with every book's price,
   model vs book, and for both players the id, names in the history, match count, last match, bio and last 10 results).
   - **5A DATA CHECK**, this one decides whether the bet goes in: for both players the id must be the right player (the names in
     the history are that player, no SYNTHETIC id, no second player of the same name, no split or merged player under another
     name), and the history under that id must really be theirs, complete and in the right ORDER: its last results must match what
     Flashscore shows for that player (check the player's recent Flashscore results when anything looks off: a match missing, a
     duplicated match, an opponent or tournament that does not fit, a gap, an event dated before matches that were played
     earlier, as happened with the Shanghai Masters on 2026-10-15). If the data is wrong: `python3 review.py $B no FS_MID reason`
     (not logged) and then ALWAYS fix it right away, without asking (standing rule, `research/CLAUDE.md`):
     fill or correct the source (missing matches: re-scrape the period with `fs_results.py`, which also reads the paged results
     feed and writes `{tour}_match_dates.csv`; wrong ids, rounds, dates or duplicates: correct the raw rows), fix the code so it does
     not happen again, rebuild the features (stop the loop by PID, `features2.py` with `F2_CLEAN_GAMES=1` from the raw dir
     `$B/fx_raw_fix` (fx_raw_bf plus the matches `fs_fill.py` added from the Flashscore audit plus the Challenger id fix of
     `fs_fix_ids.py`: Challenger event ids are `fs-ch-<slug>-<year>`; put new rows into a new fsfill file there) into a new dir, one tour
     at a time, ~8 GB each, mkdir the output dir first; check the new build against `out_live2` with `cmp_build.py`: no Flashscore id lost
     except matches the loop added after the raw snapshot, only the intended rows changed; swap it in as `out_live2` (mv, keep the old
     one until the loop runs), run `LIVE_OUT=out_live2 F2_CLEAN_GAMES=1 python3 fs_update.py $B 14` to add the recent matches again,
     **delete `$B/paper/pred_cache_*.pkl`** (the cache is only valid for the build it was made on), restart the loop; the first pass rebuilds
     the `.fx_state`), remove that match from `$B/paper/skipped.csv` so the next pass re-evaluates it on the fixed data, write it into the
     README, commit and push, and tell the user what was wrong and what changed.
   - **5B VERDICT**, recorded only, never blocks: if the data is right the bet is ALWAYS logged. Judge it (form, break or injury,
     level and surface, is the model vs the books gap believable, did the price move against us; quick WebSearch when the gap is big
     or anything looks odd) and log it with `python3 review.py $B ok FS_MID short-note` (you would play it, claude=da) or
     `python3 review.py $B ne FS_MID reason` (you would not, claude=ne). The da/ne results are compared at the end of the month.
     Do it quickly, the opening price may move.
6. If `paper/ledger.csv`, `paper/opklade.xlsx` or `paper/opklade.json` changed, `git add` only those three files, commit and push to
   `claude/awesome-shannon-j3tap7` (commit and push as two separate Bash calls; if the auto-mode classifier denies the push, do not work
   around it, stop and tell the user). Commit trailers:
   `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_019MZnAZNJkBUtw1qDMpWMK8`.
   Code changes are committed and pushed the same way.
   - **6b) ONLINE TABLE**, every pass: the user watches the dashboard https://claude.ai/artifact/Mjxr7iBZkf5ztbdoDHMsL9 (same rows as
     the Excel, from `paper/opklade.json`, written by `ledger_xlsx.py`). Upload it: Artifact publish with `url` = that link,
     `file_path` = `/home/user/pantasocial/research/tennis-ml/paper/opklade.json`, `asset: true`; then ArtifactData `get` of
     collection `datasets` doc `bets` and `update` it (`if_version`) with `source {kind:'file', url:<the new /_blob/ url>, name:'opklade.json'}`
     and `updated {at:<now ISO UTC>, by:'Claude'}`. Do not change the page.
7. Only if there are NEW bets, proposals stopped for a data error, bets that got settled, or new `not logged` bets, send the user
   `opklade.xlsx` with SendUserFile and a short message in Serbian (kick-off in Belgrade time, player, odds, bookmaker, stake, your
   da/ne verdict with one line why; for data errors the players, what was wrong and what was fixed; for settlements the result and
   the bank). Otherwise stay quiet. (The per-minute match watcher `watch.py` was switched off at the user's request; do not start it.)

## Prediction cache (paper_trade.py)

`$B/paper/pred_cache_{tour}.pkl` holds P(home) per fixture with a signature of its inputs (model, both players' ids and history sizes,
rank, round, kick-off day, bio). A fixture goes through `features_incr.py`, `fx_live.py` and the model only when it is new, when its
signature changed (a new result of either player, a new rank list, a moved round) or when the entry is older than 6 h
(`CACHE_TTL`: ratings of third players drift a little with every result). A pass with nothing new takes ~20 s instead of ~70 s.
`candidates.csv` (same dir) lists every priced side of the last pass with its EV.

## Handoff state (written 2026-10-18 ~14:00 UTC, for a fresh chat after `/clear`)

Everything needed to continue is in this file, `README.md` (sections "Production switched to models_v7" and "Paper loop: EV>3%, every 10 minutes, ...")
and `research/CLAUDE.md`. The user writes Serbian: answer in Serbian, short, no long blocking waits.

- **Data dir** `B=/tmp/claude-0/-home-user-pantasocial/d746d33b-0578-5ee2-a55c-5a9f85958b0a/scratchpad/tennis` (not in the repo; if it is gone after a
  container reset the build, models and raw data must be rebuilt, see the README sections on `out_live2`, `models_v7`, `fx_raw_fix`).
- **Production**: `paper_loop.sh` (models_v7, tag v7, RS prices, six books, no ITF, EV>3% odds<=5, quarter Kelly on 100k, REVIEW=1, 10-min pass with prediction
  cache). Ledger `paper/ledger.csv` (44 bets: 32 "novi" = models_v3, 6 v6, 6 v7), `paper/opklade.xlsx`, `paper/opklade.json` are committed to
  `claude/awesome-shannon-j3tap7` (HEAD `f6e59ec` plus later CLV refresh commits). The loop dies when the container idles; the six scheduled checks
  (:04 :14 :24 :34 :44 :54, prompt = "read PAPER_CHECK.md and follow it") restart it. They need no action from a new chat; just follow the steps above when one fires.
- **Dashboard** https://claude.ai/artifact/Mjxr7iBZkf5ztbdoDHMsL9 : document `datasets/bets`, last known version 180 (always `get` it first for `if_version`).
- **Open bets at this point** (Belgrade time = UTC+2): Basavareddy-Dougaz 1.42 Superbet (18.10 15:00), Mmoh-Kumstat 1.90 Superbet (18.10 15:30),
  Jacquet 1.65 Superbet (18.10 12:30, claude=ne), Gomez 1.45 Mozzart (18.10 21:30), Golubic 1.43 Superbet and Uchijima 1.91 bet365 (19.10 03:00),
  Alexandrova 1.28 1xBet (19.10 04:00, claude=ne), Bellucci 2.08 Mozzart and Tsitsipas 1.29 1xBet (19.10 08:00). When one settles: send `opklade.xlsx` with the result and the bank.
- **Open points, user has not decided**: weekly retrain schedule (`retrain.sh`, `weekly_test.py`), a Model column in the Excel/dashboard, splitting the ledger into
  pre-fix/clean bets, restoring the removed Kostovic bet, models_v6 as a shadow model, whether to keep six checks an hour (each costs the user one reply).
- **Not verified yet**: cache hit rate in later passes (should be ~0 recomputes when nothing new; the first pass after the switch recomputed 69 of 119 ATP fixtures
  because `fs_update` had added results and ranks). Look at the number of "need" fixtures in `$B/paper/loop.log`.
- **Standing rules**: fix data errors immediately without asking; after any `out_live2` rebuild delete `$B/paper/pred_cache_*.pkl`; features2.py max one at a time (~8 GB);
  results into the README, commit, push (two separate Bash calls); never `pkill -f`; never start a second loop; the match watcher `watch.py` stays off.
