"""Download 5y of Binance spot klines for every */USDT pair, delisted ones included.

api.binance.com geoblocks some regions (HTTP 451); data-api.binance.vision serves the same
public market data without that restriction. Delisted (BREAK) symbols are kept on purpose:
dropping them would bake survivorship bias into every backtest.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE = "https://data-api.binance.vision/api/v3"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
STABLES = {"USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "PAX", "UST", "USTC", "SUSD", "EUR",
           "GBP", "AUD", "TRY", "BRL", "RUB", "UAH", "NGN", "ZAR", "IDRT", "BIDR", "BVND",
           "AEUR", "EURI", "USD1", "XUSD", "PYUSD", "USDE", "RLUSD", "BFUSD", "U", "USDS", "PAXG"}
LEVERAGED = ("UP", "DOWN", "BULL", "BEAR")

_lock = threading.Lock()
_weight = [0]


def get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research"})
            with urllib.request.urlopen(req, timeout=30) as r:
                w = r.headers.get("x-mbx-used-weight-1m")
                if w:
                    with _lock:
                        _weight[0] = int(w)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                time.sleep(int(e.headers.get("Retry-After", "30")))
            elif e.code == 400:
                return None
            else:
                time.sleep(2 ** i)
        except Exception:
            time.sleep(2 ** i)
    raise RuntimeError(f"failed: {url}")


def universe():
    info = get(f"{BASE}/exchangeInfo")
    out = []
    for s in info["symbols"]:
        b = s["baseAsset"]
        if s["quoteAsset"] != "USDT" or b in STABLES:
            continue
        if any(b.endswith(x) and len(b) > len(x) for x in LEVERAGED):
            continue
        out.append((s["symbol"], s["status"]))
    return out


def fetch(symbol, interval, start_ms, end_ms):
    rows, cur = [], start_ms
    while cur < end_ms:
        while _weight[0] > 5000:
            time.sleep(1)
        data = get(f"{BASE}/klines?symbol={urllib.parse.quote(symbol)}&interval={interval}&startTime={cur}&endTime={end_ms}&limit=1000")
        if not data:
            break
        rows.extend(data)
        nxt = data[-1][6] + 1
        if len(data) < 1000 or nxt <= cur:
            break
        cur = nxt
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=COLS).drop(columns="ignore")
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df.drop_duplicates("open_time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = os.path.join(a.out, a.interval)
    os.makedirs(out, exist_ok=True)
    end = int(time.time() * 1000)
    start = end - int(a.years * 365.25 * 86400 * 1000)
    syms = universe()
    with open(os.path.join(a.out, "universe.json"), "w") as f:
        json.dump(syms, f)
    print(f"{len(syms)} USDT symbols ({sum(s == 'TRADING' for _, s in syms)} trading)", flush=True)
    todo = [s for s, _ in syms if not os.path.exists(os.path.join(out, f"{s}.parquet"))]
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, s, a.interval, start, end): s for s in todo}
        for f in as_completed(futs):
            s = futs[f]
            try:
                df = f.result()
                if df is not None and len(df):
                    df.to_parquet(os.path.join(out, f"{s}.parquet"), index=False)
            except Exception as e:
                print(f"ERR {s}: {e}", file=sys.stderr, flush=True)
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(todo)} weight={_weight[0]}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
