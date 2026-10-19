# Basketball (NBA, EuroLeague) - opening-odds test

Same idea as the tennis study: an odds-free model, bets placed at the bookmaker's OPENING price.

## Data
BetExplorer (`scrape_betexplorer_bball.py`): results of every season since 2008-09 (regular season, play-in,
playoffs, Final Four; pre-season and All-Star skipped), and from 2020-21 on each bookmaker's opening and closing
home/away (incl. OT) odds. From this server BetExplorer lists bet365.us (all seasons from 2020-21), BetMGM.us and
Stake.com (from 2024-25). Before 2020-21 no odds are listed, so those seasons only train the model.

## Model (`bball_model.py`)
Results only, no odds: margin-of-victory Elo, rolling form over 3/5/10/20 games (wins, margin, points for/against,
opponent-adjusted margin), season record, home/away form, rest, back-to-back, games in 4/7 days, road-trip length,
head-to-head, home-minus-away differences. Walk-forward: test season S trains on seasons < S-1 and is calibrated on
S-1 (XGB + LGBM + logistic on Elo, Platt).

## Pre-registered (written before the odds were parsed)
Primary: NBA 2020-21..2025-26, EV>5% & odds<=2 at the bet365.us opening price. Success = ROI > 0 and 90% lower bound > 0.
Also reported: EuroLeague (same rule, small sample), BetMGM and Stake.com (2024-25+), closing-price ROI, CLV of
picks vs random sides, void stress, and a model+market Shin blend (leave-one-season-out, exploratory).

## Results -> FAIL (NBA and EuroLeague)
Odds coverage: NBA 7,617 games 2020-21..2025-26 (bet365.us all, BetMGM 2,581 and Stake 1,335 from 2024-25);
EuroLeague 1,989 games (BetMGM 687, Stake 406). Opening margins are low: 4.3-5.9% (tennis ITF books: 7.5-9%).
| League @ book | Model log-loss | Opening fair | Closing fair | PRIMARY EV>5% odds<=2 | Random sides | CLV of picks |
|---|---|---|---|---|---|---|
| NBA @ bet365.us | 0.6261 | 0.6079 | 0.5996 | 981 bets, **-4.6%** (lo -9.1%) | -4.4% | +1.9% |
| NBA @ BetMGM | 0.6041 | 0.5897 | 0.5799 | 356, -6.4% | -5.4% | +2.1% |
| NBA @ Stake.com | 0.5999 | 0.5875 | 0.5760 | 181, -7.5% | -5.1% | +1.1% |
| EuroLeague @ bet365.us | 0.6320 | 0.6153 | 0.6159 | 387, **-7.2%** (lo -13.9%) | -5.6% | +1.5% |
| EuroLeague @ BetMGM | 0.6343 | 0.6171 | 0.6196 | 172, -10.6% | -7.3% | +1.9% |
| EuroLeague @ Stake.com | 0.6338 | 0.6219 | 0.6262 | 76, -6.7% | -7.6% | +1.9% |
No rule (EV 0-10%, odds caps 2/3/any) is positive with a lower bound above 0 anywhere. The model is clearly worse
than even the OPENING line (log-loss gap 0.012-0.02), and its picks' prices shorten only 1-2.5% (tennis: 5-10%).
In tennis the opening lines of lower-tier events are soft; NBA/EuroLeague opening lines are already sharp, so a
results-only model has nothing to add. The model+market blend puts almost all weight on the market (a handful of
bets); the Stake EuroLeague blend row is a degenerate fit on two seasons and is ignored.
Verdict: no profitable plan in NBA or EuroLeague moneyline with this approach.
