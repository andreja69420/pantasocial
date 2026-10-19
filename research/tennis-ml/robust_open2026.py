"""Robustness checks for the 2026 opening-odds result: CLV trimming, bootstrap, shuffle null (p permuted).

usage: robust_open2026.py DATA_DIR TOUR PREDS LEVEL_REGEX
"""
import sys
import pandas as pd, numpy as np
D, TOUR, PREDS, LEVEL = sys.argv[1:5]
P=pd.read_parquet(f"{D}/out/{PREDS}")
O=pd.read_parquet(f"{D}/out/{TOUR}_bxo_odds.parquet")
lvl=pd.read_parquet(f"{D}/out/{TOUR}_f2.parquet",columns=["tourney_level","tourney_name"])
P=P[P.year==2026].join(O,how="inner"); P=P[lvl.tourney_level.astype(str).reindex(P.index).str.match(LEVEL).fillna(False).values]
P=P[(P.open_w>1)&(P.open_l>1)&(P.close_w>1)&(P.close_l>1)]
st=P.stands.values&~P.retired.values
def side(pp):
    rows=[]
    for won in (True,False):
        q=np.where((P.y==1)==won,pp,1-pp); oo=(P.open_w if won else P.open_l).values; oc=(P.close_w if won else P.close_l).values
        rows.append(pd.DataFrame(dict(mi=P.index,q=q,oo=oo,oc=oc,won=won,pnl=np.where(st,np.where(won,oo-1,-1.0),0.0),stands=st)))
    S=pd.concat(rows,ignore_index=True); S["ev"]=S.q*S.oo-1; S["clv"]=S.oo/S.oc-1; return S
S=side(P.p.values); b=S[(S.ev>0.05)&(S.oo<=2)]
print("picks",len(b),"won share %.3f mean odds %.3f implied %.3f model q %.3f"%(b.won.mean(),b.oo.mean(),(1/b.oo).mean(),b.q.mean()))
print("CLV quantiles",b.clv.quantile([.05,.25,.5,.75,.95]).round(3).to_dict())
for cap in (0.15,0.30,10):
    x=b[b.clv.abs()<=cap]; print(f"|CLV|<={cap}: n={len(x)} ROI {x.pnl.mean():+.2%}")
print("picks with no move: %d ROI %+.2f%%"%((b.clv==0).sum(), 100*b[b.clv==0].pnl.mean()))
print("void picks", (~b.stands).sum(), "| tournaments", lvl.tourney_name.reindex(b.mi).nunique())
r=np.random.default_rng(0); bs=[b.pnl.values[r.integers(0,len(b),len(b))].mean() for _ in range(5000)]
print("bootstrap 5%/95%", np.percentile(bs,[5,95]).round(4))
nulls=[]
for k in range(300):
    N=side(r.permutation(P.p.values)); nulls.append(N[(N.ev>0.05)&(N.oo<=2)].pnl.mean())
print("shuffle null ROI mean %+.2f%%, share >= observed %.3f"%(100*np.mean(nulls),np.mean(np.array(nulls)>=b.pnl.mean())))
