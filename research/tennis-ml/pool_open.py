"""Pooled 2025+2026 opening-odds result over women's ITF, Challenger and men's ITF (EV>5% & odds<=2, BetMGM.us open),
with monthly hit count and flat-stake drawdown.

usage: pool_open.py DATA_DIR
"""
import sys, numpy as np, pandas as pd
D=sys.argv[1]
def side(tour, preds, bxo, rx):
    P=pd.concat([pd.read_parquet(f"{D}/out/{f}") for f in preds]); P=P[~P.index.duplicated()]
    O=pd.concat([pd.read_parquet(f"{D}/out/{tour}_{n}.parquet") for n in bxo]); O=O[~O.index.duplicated()]
    lvl=pd.read_parquet(f"{D}/out/{tour}_f2.parquet",columns=["tourney_level","date"])
    P=P[P.year.isin([2025,2026])].join(O,how="inner"); P=P[lvl.tourney_level.astype(str).reindex(P.index).str.match(rx).fillna(False).values]
    P=P[(P.open_w>1)&(P.open_l>1)]
    out=[]
    for won in (True,False):
        q=np.where((P.y==1)==won,P.p,1-P.p); oo=(P.open_w if won else P.open_l).values
        st=P.stands.values&~P.retired.values
        pnl=np.where(st,np.where(won,oo-1,-1.0),0.0); pick=(q*oo-1>0.05)&(oo<=2)
        out.append(pd.DataFrame(dict(pnl=pnl[pick],date=lvl.date.reindex(P.index).values[pick],o=oo[pick])))
    return pd.concat(out)
S=pd.concat([side("wta",["preds3_wta_ens_r6_hist1991_ens_lower_2025.parquet","preds3_wta_ens_r6_hist1991_ens_lower_2026.parquet"],["bxo25_witf","bxo_odds"],r"^\d"),
             side("atp",["preds3_atp_ens_r6_hist1991_ens_lower_2025.parquet","preds3_atp_ens_r6_hist1991_ens_lower_2026.parquet"],["bxo25_chal","bxo_odds_chal"],r"^C$"),
             side("atp",["preds3_atp_ens_r6_hist1991_ens_lower_2025.parquet","preds3_atp_ens_r6_hist1991_ens_lower_2026.parquet"],["bxo25_mitf","bxo_odds_mitf"],r"^(\d|S)")])
m=S.pnl.mean(); se=S.pnl.std()/np.sqrt(len(S))
print(f"pooled 2025+2026: {len(S)} bets, ROI {m:+.2%}, 90% lo {m-1.645*se:+.2%}, mean odds {S.o.mean():.2f}")
mo=S.groupby(pd.to_datetime(S.date).dt.to_period("M")).pnl.agg(["size","sum"])
print("months positive:", (mo["sum"]>0).sum(), "/", len(mo))
# flat 1-unit stakes: drawdown
c=S.sort_values("date").pnl.cumsum().values; dd=(np.maximum.accumulate(c)-c).max()
print(f"flat 1u stakes: profit {c[-1]:.1f}u, max drawdown {dd:.1f}u, bets/month {len(S)/len(mo):.0f}")
