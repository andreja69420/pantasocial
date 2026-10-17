#!/bin/bash
# Retrain the production model on the current live build: snapshot of LIVE_OUT -> the four feature modules (ratings, score,
# context, pbp) -> fx_merge -> train_final.py per tour -> DATA_DIR/models_TAG/{wta,atp}_final.pkl. The model is trained on
# every season to TRAIN_TO and early-stopped / calibrated on the rows after it (2026 to the snapshot date). Nothing in
# production changes: switch with MODELS=models_TAG MODEL_TAG=TAG in paper_loop.sh after looking at the replay.
#
# usage: retrain.sh DATA_DIR TAG [full|base] [quick|slow]
#   full: rank and serve statistics features included (the 2026 holdout test, README "Gaps filled"), base: as models_v6
#   quick: default learning rate (about 7 min a tour), slow: --lr-mult=0.333 --trees=6000 --patience=2000 (about 48 min a tour)
D=${1:?DATA_DIR}; TAG=${2:?TAG}; SET=${3:-full}; SPEED=${4:-quick}
LIVE=${LIVE_OUT:-out_live2}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
W=$D/rt_$TAG; rm -rf "$W"; mkdir -p "$W/snap" "$W/fx" "$W/v"
if [ "$SET" = full ]; then RX='nx_|^(?!.*(mk_|seeded|entry_|min_|minutes|draw))'
else RX='nx_|^(?!.*(ace_rate|bp_conv|bp_saved|break_|df_rate|dominance|first_in|first_won|hold_|rt_won|second_won|stats_n|sv_won|sv_vs_venue|venue_ace|venue_sv|mk_|rank|log_pts|seeded|entry_|min_|minutes|draw))'; fi
if [ "$SPEED" = slow ]; then SLOW="--lr-mult=0.333 --trees=6000 --patience=2000"; else SLOW=""; fi
for t in wta atp; do
  cp "$D/$LIVE/${t}_matches.parquet" "$W/snap/"; cp -r "$D/$LIVE/${t}_f2_parts" "$W/snap/"
  ln -s "$D/$LIVE/${t}_f2.parquet" "$W/snap/${t}_f2.parquet"; ln -s "$D/$LIVE/${t}_state.pkl" "$W/snap/${t}_state.pkl"
done
for m in ratings score context pbp; do
  for t in wta atp; do
    ( python3 fx_$m.py "$W/snap/${t}_matches.parquet" "$W/fx/${t}_$m.parquet" > "$W/fx/${t}_$m.log" 2>&1 ) &
  done
  wait
done
python3 -u fx_merge.py "$W/snap" "$W/out" "$W/fx" > "$W/merge.log" 2>&1 || { echo "merge failed" > "$W/FAILED"; exit 1; }
ln -s "$W/out" "$W/v/out"
mkdir -p "$D/models_$TAG"
for t in wta atp; do
  SECONDS=0
  python3 -u train_final.py "$W/v" $t "$D/models_$TAG/${t}_final.pkl" "--only=$RX" --train-to=2025 $SLOW 2>&1 | grep -vi warn | tail -n 3 > "$W/final_$t.log"
  echo "$t final exit ${PIPESTATUS[0]} ${SECONDS}s" >> "$W/final_$t.log"
done
echo done > "$W/train.flag"
