import sys

from features import build, load_panel

D = sys.argv[1]
P = load_panel(D, "1d")
for h in [1, 7]:
    X, _, _ = build(P, horizon=h)
    X.to_parquet(f"{D}/X_1d_h{h}.parquet")
    print(h, X.shape)
