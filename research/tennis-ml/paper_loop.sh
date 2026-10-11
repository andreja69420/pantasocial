#!/bin/bash
# Paper trading every INTERVAL seconds (default 30 min): results update, odds of every listed fixture, features, bets at
# opening prices, settlement. Log in DATA_DIR/paper/loop.log; the ledger is copied into the repo (paper/ledger.csv).
D=${1:?DATA_DIR}; INTERVAL=${INTERVAL:-1800}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D/paper" "$HERE/paper"
while true; do
  python3 -u "$HERE/paper_trade.py" "$D" 7 >> "$D/paper/loop.log" 2>&1
  [ -f "$D/paper/ledger.csv" ] && cp "$D/paper/ledger.csv" "$HERE/paper/ledger.csv"
  sleep "$INTERVAL"
done
