"""Do the models play the same bets as the live loop? Holdout predictions (eval_levels2026.py --bets) and a replayed
production model (replay_models.py) are put next to the paper ledger, match by match (Flashscore id).

usage: compare_live.py BETS_PARQUET REPLAY_PREFIX NAME [FROM_DATE=2026-10-12]
  REPLAY_PREFIX: replay_models.py outputs are REPLAY_PREFIX{NAME}_{atp|wta}.parquet (column p_winner, index = match id)
  reads paper/ledger.csv and paper/opklade.json from the working directory (research/tennis-ml)
"""
import json
import sys

import numpy as np
import pandas as pd

BETS, PREFIX, NAME = sys.argv[1:4]
FROM = pd.Timestamp(sys.argv[4] if len(sys.argv) > 4 else "2026-10-12")
ITF={"15","25","35","50","75","100"}
lg=lambda p: np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
def shin(a,b):
    lo,hi=np.zeros_like(a),np.full_like(a,0.5); Bq=a+b
    f=lambda z,x:(np.sqrt(z**2+4*(1-z)*x**2/Bq)-z)/(2*(1-z))
    for _ in range(50):
        z=(lo+hi)/2; s=f(z,a)+f(z,b); hi,lo=np.where(s<1,z,hi),np.where(s<1,lo,z)
    return f((lo+hi)/2,a)
H=pd.read_parquet(BETS); H['day']=pd.to_datetime(H.start,unit='s').dt.normalize()
R=pd.concat([pd.read_parquet(f'{PREFIX}{NAME}_{t}.parquet').assign(tour=t) for t in ('atp','wta')]).reset_index()
H=H[H.day>=FROM].merge(R,on=['mi','tour'],how='left')
print('priced matches 12-14 Oct',H.fs_mid.nunique(),'with replay q:',H[H.p_winner.notna()].fs_mid.nunique())
H['q_m']=np.where(H.won,H.p_winner,1-H.p_winner)
H['evb_m']=1/(1+np.exp(-(0.44*lg(H.q_m.values)+0.64*lg(shin(1/H.o.values,1/H.oo.values)))))*H.o-1
L=pd.read_csv('paper/ledger.csv'); H['in_ledger']=H.fs_mid.isin(L.fs_mid)
rule=lambda e:(e>0.05)&(H.o<=5)&(~H.lvl.astype(str).isin(ITF))
H['ho']=rule(H.evb); H['pr']=rule(H.evb_m)
for name,m in (('holdout (never saw 2026)',H.ho),(f'{NAME} model at the opening price',H.pr)):
    p=H[m]; print(f'{name}: {len(p)} bets, wins {int(p.won.sum())}, flat ROI {p.pnl.mean():+.1%}, also in live ledger {int(p.in_ledger.sum())}')
print('both models:',int((H.ho&H.pr).sum()))
x=H[H.in_ledger].merge(L[['fs_mid','side','odds','q_model','ev','result','home','away']].rename(columns={'ev':'ev_live'}),on='fs_mid')
x['lw']=(x.result==x.side); x=x[x.won==x.lw]
print('\nlive-ledger bets (side we bet), q live vs replay vs holdout; EV at best OPEN price:')
print(x[['home','away','odds','o','q_model','q_m','q','ev_live','evb_m','evb','won']].round(3).to_string())
print('mean q: live',x.q_model.mean().round(3),NAME,x.q_m.mean().round(3),'holdout',x.q.mean().round(3))
y=H[(H.ho|H.pr)&~H.in_ledger]
print('\nplan bets (either model, open price) NOT in the live ledger:'); print(y[['fs_mid','tour','lvl','o','q_m','q','evb_m','evb','won','pnl']].round(3).to_string())

# --- was the gap between live q and replay q caused by data that was fixed later? split by when the bet was logged
x=x.merge(L[['fs_mid','logged_at','odds','pnl']].rename(columns={'odds':'odds_l','pnl':'pnl_l'}),on='fs_mid',suffixes=('','_x'))
x['logged_at']=pd.to_datetime(x.logged_at)
cut=pd.Timestamp('2026-10-13 22:30')   # features rebuilt from the corrected raw data and swapped in
x['d_live_replay']=x.q_model-x.q_m
for name,g in (('logged before the rebuild (10-13 22:30 UTC)',x[x.logged_at<cut]),('logged after',x[x.logged_at>=cut])):
    print(f'\n{name}: {len(g)} bets, mean live q {g.q_model.mean():.3f}, replay q {g.q_m.mean():.3f}, diff {g.d_live_replay.mean():+.3f}, mean abs {g.d_live_replay.abs().mean():.3f}')
# same live prices, replay probability: would the corrected model have taken them? blend with the book's fair price at the taken price is not stored, use the replay EV at open as proxy
x['ev_replay_at_taken']=1/(1+np.exp(-(0.44*lg(x.q_m.values)+0.64*lg(shin(1/x.o.values,1/x.oo.values)))))*x.odds_l-1
g=x[x.ev_replay_at_taken>0.05]
print(f'\nlive bets the corrected model (replay, blend EV at the price we actually took) would still play: {len(g)} of {len(x)}; wins {int(g.won.sum())}, flat ROI {g.pnl_l.mean():+.1%}')
g=x[x.ev_replay_at_taken<=0.05]
print(f'live bets it would NOT play: {len(g)}; wins {int(g.won.sum())}, flat ROI {g.pnl_l.mean():+.1%}')
print('all live bets flat ROI',f'{x.pnl_l.mean():+.1%}')

J={r['id']:r for r in json.load(open('paper/opklade.json'))['rows']}
x['stake']=x.fs_mid.map(lambda m:J[m]['stake']); x['profit']=x.fs_mid.map(lambda m:J[m]['profit'])
for name,g in (('would still play',x[x.ev_replay_at_taken>0.05]),('would not play',x[x.ev_replay_at_taken<=0.05])):
    print(f"\n{name}: {len(g)} bets, staked {g.stake.sum():,.0f}, profit {g.profit.sum():+,.0f}")
    print(g[['home','away','odds_l','q_model','q_m','ev_replay_at_taken','stake','profit']].round(3).to_string())
print('max logged_at',x.logged_at.max(),' min',x.logged_at.min())
