"""Download monthly funding-rate history for every perpetual that maps to a spot USDT pair in our data."""
import glob
import io
import json
import os
import re
import subprocess
import sys
import zipfile

import pandas as pd

D = sys.argv[1]
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
perps = set(json.load(open(f"{D}/perps.json")))
spot = [os.path.basename(p)[:-8] for p in glob.glob(f"{D}/1d/*.parquet")]
mapping = {}
for s in spot:
    for pre, mult in [("", 1), ("1000", 1000), ("1000000", 1_000_000), ("1M", 1_000_000)]:
        if pre + s in perps:
            mapping[s] = (pre + s, mult)
            break
json.dump(mapping, open(f"{D}/spot_to_perp.json", "w"))
print(len(mapping), "spot pairs have a USDT-M perp", flush=True)

raw = f"{D}/raw_funding"
os.makedirs(raw, exist_ok=True)
listing = "\n".join(f"{S3}?delimiter=/&prefix=data/futures/um/monthly/fundingRate/{p}/ {raw}/list_{p}.xml"
                    for p, _ in mapping.values())
subprocess.run("xargs -P 48 -n 2 sh -c 'curl -sS -m 60 --retry 5 -o \"$1\" \"$0\"'", shell=True,
               input=listing.encode(), check=False)
keys = []
for f in glob.glob(f"{raw}/list_*.xml"):
    keys += re.findall(r"<Key>([^<]+\.zip)</Key>", open(f).read())
keys = [k for k in keys if k[-11:-4] >= "2021-09"]
todo = [k for k in keys if not os.path.exists(f"{raw}/{os.path.basename(k)}")]
print(len(keys), "monthly files,", len(todo), "to fetch", flush=True)
subprocess.run("xargs -P 64 -n 2 sh -c 'curl -sS -m 60 --retry 5 -o \"$1\" \"$0\"'", shell=True,
               input="\n".join(f"https://data.binance.vision/{k} {raw}/{os.path.basename(k)}" for k in todo).encode(),
               check=False)
frames = []
for k in keys:
    f = f"{raw}/{os.path.basename(k)}"
    try:
        with zipfile.ZipFile(f) as z:
            txt = z.read(z.namelist()[0]).decode()
    except Exception:
        continue
    df = pd.read_csv(io.StringIO(txt), header=None)
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    df = df.iloc[:, [0, 2]]
    df.columns = ["time", "rate"]
    df["perp"] = os.path.basename(k).split("-fundingRate")[0]
    frames.append(df)
fr = pd.concat(frames)
fr["time"] = pd.to_datetime(fr["time"].astype("int64"), unit="ms")
fr["rate"] = fr["rate"].astype(float)
fr.to_parquet(f"{D}/funding.parquet", index=False)
print(fr.shape, fr["time"].min(), fr["time"].max(), fr["perp"].nunique(), "perps", flush=True)
