# Tennis betting ML (ATP + WTA, all levels)

Question: can a model built on the complete match history (futures/ITF, challengers, qualifying, main tour)
beat bookmaker prices? Short answer: **not at Bet365 margins.** The model carries a small amount of
information the price does not, but it is far smaller than the ~6% margin.

## Project requirements (from the owner)
- The model must **not** use bookmaker odds as features. Odds are used only to decide value and settle bets.
- The model must have **at least 150 features**.
- Iterate on a short recent window first (last ~5 years); extend the history only when asked.
- Development period 2021-2024 is used for iterating on features; 2025-2026 is a locked holdout evaluated once.

## Data
- **Matches:** Jeff Sackmann's `tennis_atp` / `tennis_wta` (the GitHub originals are no longer reachable;
  used the archival mirror `Aneeshers/tennis-sackmann-archive` on Hugging Face, CC BY-NC-SA 4.0).
  1991 - June 2026: 855k ATP and 675k WTA matches across all levels.
- **Odds:** tennis-data.co.uk Bet365 match odds via the Kaggle "ATP/WTA tennis daily" dataset mirrored on
  Hugging Face (`groundhog2107/atp_tennis`, `Nevoreuven/tennis-betting-odds-model`). tennis-data.co.uk itself
  is behind a Cloudflare block from this environment. Bookmaker identified from the margin (5.3-7%).
  Odds are matched to Sackmann matches by name + date window: 97.7% (ATP) / 94.6% (WTA) of rows, the
  winner agrees on 99.9%.

## Method
- `build.py` - one chronological pass over every match: overall and surface Elo (538-style K), form (last 10,
  52 weeks, surface), rest days, matches in the last 14 days, minutes in the last 7 days, serve/return points
  won over the last 30 matches with stats, head-to-head. All features are strictly pre-match.
- `train_bet.py` - walk-forward XGBoost: train on all levels for the previous 12 years, predict the next
  year's tour-level matches that have odds. Retirements void the bet unless the first set was completed.
  Margin is removed proportionally for the "fair" market probability; bets are settled at the real odds.
  A stacking logistic model (market logit + model logit, fit on prior years only) tests whether the model
  adds information beyond the price.
- `segments.py` - ROI by segment and sensitivity to the bookmaker margin.

## Results
| | ATP 2006-2026 | WTA 2009-2025 |
|---|---|---|
| Matches with odds | 47,873 | 36,191 |
| Bet365 median margin | 6.4% | 6.2% |
| Log-loss market / XGBoost / Elo | **0.568** / 0.577 / 0.586 | **0.581** / 0.589 / 0.597 |
| Accuracy market / XGBoost | **70.0%** / 69.1% | **68.9%** / 68.0% |
| XGBoost value bets EV>5% | 20,525 bets, **ROI -8.7%**, 1/21 years positive | 15,240 bets, **ROI -7.1%**, 1/17 |
| Bet every favourite | -4.8% | -5.1% |
| Stacked market+model EV>5% | 277 bets, +3.1% (CI -10%..+17%) | 252 bets, +13.5% (CI -3.5%..+33%) |

- The stacking coefficient on the model is a stable ~0.2 every year: the model knows *something* the price
  doesn't, but bets on its disagreements with the market lose more than the margin.
- No segment (low-experience players, outside top-100, early rounds, Slams) shows a significant edge.
- Margin sensitivity (odds rebuilt from Bet365's fair price): the stacked strategy is ~+1.5-2% ROI at
  2-3% margin (Pinnacle-like or best price across books), roughly at the edge of significance. This is an
  optimistic estimate: Pinnacle's own prices are sharper than Bet365's fair price.

## Next step if pursuing this
Get Pinnacle and max-odds columns (tennis-data.co.uk xlsx files contain `PSW/PSL`, `MaxW/MaxL`; download them
from a normal browser) and re-run the test against Pinnacle closing odds. Beating Pinnacle closing is the
real bar; soft books like Bet365 also limit winning accounts quickly.

## Pre-registered out-of-sample test (written before looking at 2020+ Pinnacle/Max odds)
Finding on 2014-2019 (dev): the model (r6_hist1991_ens, no odds as features) is profitable when settled at
best-available (Max) odds: ATP value>5% & odds<=3 -> +4.1% ROI on 6,211 bets (5/6 years positive), clearly above
random bets at the same prices (+0.85%). At Bet365 it loses; at Pinnacle it is about +1%.

Rules fixed now, evaluated once on ATP 2020-2024 (dev predictions, odds never inspected) and on the locked
2025-2026 holdout (model trained 2010-2023, early stopping 2024):
- **Rule A (primary):** ATP, bet a player when p_model * MaxOdds > 1.05 and MaxOdds <= 3.0, flat 1 unit.
- **Rule B:** same model, Pinnacle odds, p_model * PSOdds > 1.02 and PSOdds <= 3.0.
Success = ROI > 0 with a 90% bootstrap interval above zero, separately on 2020-2024 and on 2025-2026.

## Results of the pre-registered test and the Challenger/ITF test
- **Rule A (Max odds)**: 2020-2024 **PASS** (+3.8% ROI, 90% CI [+1.3%, +6.4%], 4,474 bets); 2025-2026 holdout **FAIL**
  (-7.8%). The tennis-data "Max" column has a 1.2-2.7% margin from 2025 (about 0% before), so that edge depended on a
  best-price source that changed. **Rule B (Pinnacle)**: FAIL on both.
- **Challenger/ITF** (Kaggle "A large Tennis dataset for ATP and ITF betting", odds 2009-2018, mostly OddsPortal):
  the same model (no odds as features) on ATP 2014-2018:
  - ITF/Futures (14,055 matches, 8.2% margin): the model's log-loss **beats the market** (0.5435 vs 0.5455).
    Value >5% & odds <= 2: **+5.9% ROI**, 90% CI [+3.6%, +8.0%], 2,602 bets, 5/5 years positive. All 9 rule variants
    are positive. Betting every favourite at the same prices: -4.6%; 2,000 random draws of the same size: mean -5.1%,
    99th percentile -2.4%. The model beats all of them.
  - Challenger: about break-even. Qualifying: negative.
- Caveats: the ITF segment was identified on 2014-2018 and no later ITF odds are available yet, so a forward
  (paper-trading) test is the remaining proof. ITF markets have low limits and integrity risk (match-fixing).

## Pre-registered: fresh 2012-2013 test (written before running those years)
- **ITF rule** (found on 2014-2018): ATP ITF/Futures, bet when p_model * odds > 1.05 and odds <= 2.0, flat stake.
  Success = ROI > 0 on 2012-2013 combined.
- **Challenger**: any segment rule must be chosen on 2012-2015 and confirmed on 2016-2018.

### Result of the 2012-2013 test
- **ITF rule: PASS.** 2012-2013 (never seen before): **+7.7% ROI** on 2,091 bets (2012 +10.4%, 2013 +4.0%).
  ITF 2012-2018 overall: +6.7% ROI on 4,693 bets (90% interval about +/-1.7%). All lower levels together in 2012-2013:
  model log-loss 0.5583 vs market 0.5634.
- **Challenger: no reliable edge.** Overall ~0% (explore 2012-2015 +1.3%, confirm 2016-2018 -1.3% at odds <= 2).
  No segment (experience, ranking, round, surface) is positive in both halves with a meaningful margin; the best
  ("weaker player ranked >300", any odds) is +1.9% / +3.1% but within noise.

## Pre-registered: new ITF odds sources (written before running)
Sources: Kaggle "Tennis Results and Betting Odds (2014-2025)" (TennisExplorer; Bet365/Betfair/Ladbrokes/Unibet/best;
women's ITF + men's Challenger, almost no men's ITF) and "Tennis Closing Odds: Pinnacle vs 1xBet" (Mar-Sep 2026,
men's and women's ITF + Challenger).
- **Women's ITF 2014-2025** (new domain): same ITF rule, p_model * odds > 1.05 and odds <= 2.0, at Bet365 and at
  best odds. Success = ROI > 0 with 90% interval above zero.
- **Men's Challenger 2019-2025** (years not tested before): same rule; expectation from 2012-2018 is ~0%.
- **2026 ITF (men + women) at Pinnacle and 1xBet**: same rule, holdout models (trained up to 2024).

### Result: women's ITF 2014-2025 (walk-forward, 131k matches)
| Odds | Margin | Bets | ROI | 90% CI | Years + | Random bets, same odds |
|---|---|---|---|---|---|---|
| Bet365 | 7.4% | 21,127 | -0.3% | [-1.1, +0.5] | 3/12 | -4.7% |
| Best price | 4.8% | 26,558 | **+1.7%** | [+1.0, +2.4] | 11/12 | -2.0% |

Bet365: FAIL. Best price: PASS. The model beats random selection by 4-5 points at every source, but a 7% margin
absorbs the edge. Best price is an upper bound (needs many accounts; the maximum is often stale or stake-limited).
WTA125 and WTA Tour: no significant edge at any source.

### Men's ITF 2024-2026 at BetExplorer average odds (pre-registered same rule, written before running)
Scraped 1,044 BetExplorer men's ITF result pages (2024-2026); 17,813 Sackmann matches matched (41%), median margin
9.0% (average odds across bookmakers, so harsher than any single sharp book). Rule: EV>5%, odds<=2.

### Result: men's Challenger 2014-2025 (TE odds)
Bet365 EV>5% odds<=2: -3.0% (6,152 bets, 2/12 years +). Best price: +1.2%, CI [-0.1, +2.3] -> FAIL, as expected.

### Odds-range systems (user request: bet only at odds >= 2, 2.5, ... 5) - `odds_systems.py`
Post-hoc exploration, reported with an explore half / confirm half of the years.
- Odds floors (only odds >= X) are worse than no floor at Bet365 for women's ITF (EV>5%: -2.5% all odds,
  -4.0% at >=2, -5.6% at >=3, -8.9% at >=5): the favourite-longshot bias works against high odds.
- "Model flips the favourite" (market underdog the model rates >60%) showed +36% ROI, 12/12 years on women's ITF.
  **This is a data artefact, not an edge.** The TennisExplorer odds have swapped prices on some ITF matches
  (e.g. a #400 player at 11.00 vs a #848 player at 1.02, won 6-1 6-0; best price disagrees with Bet365 on the
  favourite in 11% of these bets). Even without a rank-inversion signature, dog odds >= 3.5 showed +201% ROI,
  impossible on real prices. The model spots the mispriced row and the backtest pays out a price no bookmaker
  offered. The ehallmar men's ITF data shows the same pattern at smaller scale.
- Consequence: **backtests on these free datasets are not trustworthy at odds above ~2.5.** The odds<=2 cap in the
  pre-registered rule protects against this (a swapped row puts the true favourite at the long price, which the
  cap excludes), which is why it stays the only rule used for decisions.

### Tactic screen (`tactics.py`): ~50 tactics per segment, explore/confirm halves, no-skill control
Survivor = >=300 bets, positive in both halves, 90% lower bound > 0. Control: model probabilities shuffled within
year (5 shuffles) -> 0-2 survivors per segment, so the screen itself rarely produces false positives. It does not
control for bad odds data, though: every "model p>X & EV>0" survivor on women's ITF and Challenger comes from a
handful of bets at odds > 2 with +130% to +1000% ROI (swapped-price rows). Capped at odds <= 2 they shrink to ~+1%.
- ATP Tour (Bet365 and best): 0 survivors.
- Challenger: Bet365 nothing clean; best price only small (+2 to +3%).
- Women's ITF: Bet365 nothing clean at odds <= 2 above ~+2%; best price +1 to +4%.
- Men's ITF 2012-2018: 41 of 48 survive, all subsets of the core rule (EV>5%, odds<=2, +6.7%). Strongest:
  lefty bettee +13.8% (627 bets), EV>20% +13.7% (851), opponent 3+ matches in 7 days +11.6% (712), opponent with
  retirements in the last year +10.5% (1,510), odds 1.6-2.0 +10.2% (1,510), rising Elo +8.7% (3,009).

## Pre-registered: men's ITF 2024-2026 at BetExplorer average odds (written before the predictions finished)
Primary: core rule EV>5% & odds<=2 (success = ROI > 0 with 90% lower bound > 0).
Secondary (picked from the 2012-2018 screen, reported regardless of outcome): EV>10% odds<=2; EV>20% odds<=2;
EV>5% odds 1.6-2.0; core & opponent retires often; core & opponent played yesterday; core & rising Elo;
core & opponent 3+ matches in 7 days; core & lefty. Average odds carry ~9% margin, so a best-price bettor would do
better than this test shows.

### Result: men's ITF 2024-2026 at BetExplorer average odds -> FAIL
17,812 matches, margin 9.0%. Log-loss market 0.5330 vs model 0.5424: the market now beats the model (in 2012-2018
the model beat it, 0.5435 vs 0.5455). The market got sharper; the model did not get worse.
| Rule | Bets | ROI | 90% CI | Random at odds<=2 |
|---|---|---|---|---|
| PRIMARY EV>5% odds<=2 | 2,439 | -2.9% | [-5.4, -0.5] | -3.9% |
| EV>10% odds<=2 | 1,225 | -4.4% | [-8.0, -0.7] | -4.1% |
| EV>20% odds<=2 | 339 | -8.6% | [-16.3, -1.2] | -4.0% |
| core & opp retires often | 489 | +1.1% | [-4.5, +7.0] | -3.8% |
| core & opp played yesterday | 1,039 | -1.3% | [-5.0, +2.4] | -4.0% |
| core & lefty | 240 | -3.9% | [-11.8, +3.7] | -4.5% |
By year (primary): 2024 +4.8%, 2025 -3.6%, 2026 -7.1%. The 2012-2018 men's ITF edge does not hold on current
prices; none of the subset tactics picked from the 2012-2018 screen holds either.

## Opening odds (moneyline) - `eval_open.py`
Same walk-forward predictions, settled at Bet365 OPENING prices from the TennisExplorer data instead of closing.
EV>5% & odds<=2, 2020-2025 ("modern market"):
| Segment | Bet365 close | Bet365 OPEN | best OPEN |
|---|---|---|---|
| ATP Tour | -5.2% | -0.8% | +1.5% |
| Challenger | -2.8% | +2.8% (4,253 bets, 90% lo +0.8%) | +4.7% |
| Women's ITF | -1.1% | **+4.9%** (11,825 bets, 90% lo +3.8%) | +7.0% |
| WTA125 | -2.5% | +5.4% (868) | +7.7% |
| WTA Tour | -2.3% | +4.8% (1,593) | +7.1% |
The picks get closing-line value: Bet365 shortens the model's side by +4.8% on average between open and close
(women's ITF; +5.4% Challenger; random sides +0.2%). The model anticipates the market's own moves, which does not
depend on match outcomes. Women's ITF at Bet365 open is positive in 11 of 12 years (2014-2025).
Caveats: all 2014-2025 years have now been looked at; opening prices carry low stake limits and accounts that take
them get limited; TE opening odds may include a few erroneous prices (moves >= 25% are 2% of bets).

## Pre-registered: 2026 opening-odds test on fresh data (written before scraping)
Data: BetExplorer odds history for 2026 matches (Jan-Jun, Sackmann coverage). From this server only BetMGM.us is
listed, so the test is at BetMGM.us opening prices (first price in its history), not Bet365.
Model: walk-forward ens trained to 2024, calibrated on 2025 (no 2026 data).
Primary: women's ITF 2026, EV>5% & odds<=2 at the opening price. Success = ROI > 0 and 90% lower bound > 0.
Also reported: the same at BetMGM closing price, and mean closing-line value of the picks (open/close - 1);
CLV > random sides is the secondary success signal, since a few hundred bets give a wide ROI interval.

## Total games (over/under) - `train_totals.py`, `eval_totals.py`
Regression on total games from the same 481 features (no odds), walk-forward; P(over L) from the predicted mean +
the validation-year residuals. Settled at Bet365 / Unibet / Ladbrokes lines from TennisExplorer; retired = void.
Men (75,996 matches with a line): RMSE ~6.0-6.2 games vs sd 7.0-7.4. Over log-loss model 0.6906 vs market 0.6927
(slightly better than Bet365's line), but Bet365 totals margin is 8.2%.
| Men, Bet365 | Rule | 2014-19 | 2020-25 |
|---|---|---|---|
| Challenger close | EV>5% both sides | +2.7% (5,107) | -1.6% (3,706) |
| Challenger close | EV>10% over | +6.3% (1,582) | +0.6% (532) |
| Challenger OPEN | EV>3% both | +2.3% (6,940) | +0.9% (6,178), 10/12 yrs+ |
| Challenger OPEN | EV>10% both | +5.9% (2,328) | +2.8% (1,181) |
| Tour close | EV>5% both | +7.4% (4,487) | -5.3% (3,015) |
| Tour OPEN | EV>10% both | +11.7% (2,579) | -1.0% (1,480) |
Men's totals at closing prices: FAIL on the modern market (explore half positive, confirm half negative).
At opening prices Challenger stays slightly positive in 2020-25 but within noise (+2.8% on 1,181 bets, se ~2.8%).
"Best book" totals rows (+30-80%) are a data artefact: in 2014-19 even random overs return +6-12% there, i.e. the
Unibet/Ladbrokes over prices in the free data are often for a different line; ignore them.
