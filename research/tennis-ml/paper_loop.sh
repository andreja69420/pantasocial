#!/bin/bash
# Paper trading every INTERVAL seconds (default 30 min): results update, odds of every listed fixture, features, bets at
# opening prices, settlement. Log in DATA_DIR/paper/loop.log; the ledger and the Excel sheet (quarter-Kelly stakes) are
# written into the repo (paper/ledger.csv, paper/opklade.xlsx).
D=${1:?DATA_DIR}; INTERVAL=${INTERVAL:-1800}
# production setup: base v2 build (out_live2, clean game counts) + the new-feature modules, models trained with them
export LIVE_OUT=${LIVE_OUT:-out_live2} MODELS=${MODELS:-models_v3} NX=${NX:-1} F2_CLEAN_GAMES=${F2_CLEAN_GAMES:-1} MODEL_TAG=${MODEL_TAG:-novi}
# books the user can bet with from Serbia (Flashscore region RS)
export GEO=${GEO:-RS} BOOKS=${BOOKS:-bet365,Mozzartbetrs,Admiralbet.rs,SoccerBet}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D/paper" "$HERE/paper"
while true; do
  python3 -u "$HERE/paper_trade.py" "$D" 7 >> "$D/paper/loop.log" 2>&1
  if [ -f "$D/paper/ledger.csv" ]; then
    cp "$D/paper/ledger.csv" "$HERE/paper/ledger.csv"
    (cd "$HERE" && python3 ledger_xlsx.py "$D/paper/ledger.csv" "$HERE/paper/opklade.xlsx" >> "$D/paper/loop.log" 2>&1)
  fi
  sleep "$INTERVAL"
done
