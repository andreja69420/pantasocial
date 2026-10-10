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
