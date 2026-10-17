"""Fast parallel kline fetch: one curl per (symbol, 1000-bar chunk) via xargs -P, then merge to parquet.
Python threads were CPU-bound on TLS + JSON while models trained; curl processes are not."""
import glob
import json
import os
import subprocess
import sys
import time
import urllib.parse

import pandas as pd

from download import COLS

out_dir, interval, workers, universe_file = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
step_ms = {"1h": 3600_000, "4h": 4 * 3600_000, "15m": 900_000}[interval] * 1000
raw = os.path.join(out_dir, f"raw_{interval}")
dst = os.path.join(out_dir, interval)
os.makedirs(raw, exist_ok=True)
os.makedirs(dst, exist_ok=True)
syms = [s for s, _ in json.load(open(universe_file))]
todo = [s for s in syms if not os.path.exists(os.path.join(dst, f"{s}.parquet"))]
end = int(time.time() * 1000)
start = end - int(5 * 365.25 * 86400 * 1000)

lines = []
for i, s in enumerate(todo):
    for j, t in enumerate(range(start, end, step_ms)):
        f = os.path.join(raw, f"{i}_{j}.json")
        if not os.path.exists(f):
            url = (f"https://data-api.binance.vision/api/v3/klines?symbol={urllib.parse.quote(s)}"
                   f"&interval={interval}&startTime={t}&endTime={min(t + step_ms, end) - 1}&limit=1000")
            lines.append(f"{url} {f}")
print(len(todo), "symbols,", len(lines), "requests", flush=True)
subprocess.run(f"xargs -P {workers} -n 2 sh -c 'curl -sS -m 60 --retry 5 --retry-all-errors -o \"$1\" \"$0\"'",
               shell=True, input="\n".join(lines).encode(), check=False)

for i, s in enumerate(todo):
    rows = []
    for f in sorted(glob.glob(os.path.join(raw, f"{i}_*.json")), key=lambda p: int(p.rsplit("_", 1)[1][:-5])):
        try:
            d = json.load(open(f))
            if isinstance(d, list):
                rows.extend(d)
        except Exception:
            pass
    if not rows:
        continue
    df = pd.DataFrame(rows, columns=COLS).drop(columns="ignore")
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.drop_duplicates("open_time").sort_values("open_time").to_parquet(os.path.join(dst, f"{s}.parquet"), index=False)
print("done", len(glob.glob(os.path.join(dst, "*.parquet"))), "files", flush=True)
