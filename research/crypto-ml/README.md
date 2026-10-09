# Crypto ML research (Binance, 5 years)

Can a machine-learning model trade Binance profitably? Short version: **not on spot**. On USDT-M perpetuals
there is one result that survives realistic costs (a weekly, BTC-hedged short book of the coins XGBoost ranks
worst). It is still a backtest found after trying many variants, so the next step is paper trading, not money.

## Data
- `download.py` - daily klines for every */USDT spot pair from `data-api.binance.vision` (api.binance.com
  returns 451 from some regions), **delisted pairs included** (650 symbols with data, Oct 2021 - Oct 2026).
- `fetch_fast.py` - hourly klines via parallel curl for the 102 coins that were in the top-50 by volume for 180+ days.
- `list_perps.py`, `fetch_funding.py` - every USDT-M perpetual ever listed and its full funding-rate history
  from `data.binance.vision` (475 spot pairs map to a perp).

## Method
- `features.py` - 27 point-in-time features per coin per day (returns 1-90d, volatility, volume trend,
  taker-buy ratio, distance from highs/lows, beta/correlation to BTC, age) plus cross-sectional ranks and
  market-state features. Universe each day = top-100 coins by trailing 30d volume with 90+ days of history.
- `backtest.py` - walk-forward training (expanding window, purged gap, retrain every 30-90 days), test only on
  unseen data from 2023-01-01. Models: XGBoost (`XGB_DEVICE=cuda` for GPU), LightGBM, Ridge.
  Label = next-day cross-sectional return rank.

## Results (out-of-sample 2023-01 -> 2026-09)
Signal quality: mean daily Spearman IC 0.14 (t = 27), positive on 77% of days, stable every year.

| Strategy | Total | Sharpe | Max DD |
|---|---|---|---|
| BTC buy & hold | +392% | 1.14 | -53% |
| Spot, XGB long top-10, daily, 0.15% cost | -15% | 0.24 | -76% |
| Spot, XGB long top-10, weekly, 0.15% cost | +58% | 0.50 | -74% |
| Perps long/short 10x10 daily, **no funding, no perp filter** | +2800% | 2.45 | -32% |
| Perps long/short 10x10 daily, real funding + perp filter, 0.2% cost | +15% | 0.29 | -54% |
| **Perps: short XGB bottom-10 + BTC long hedge, weekly, 0.3% cost + funding** (7 rebalance-day offsets) | +134% .. +438% | 0.73 .. 1.36 | -26% .. -49% |
| Same structure, short *all* alts equally (no ML) | +95% | 0.88 | -23% |
| Same structure, 10 *random* shorts (20 draws) | median +19% | mean 0.31, max 0.79 | |

What the attribution showed (`run_futures.py`, inline analysis):
- The headline long/short number was mostly an illusion: funding costs 25-40%/yr (crowded shorts pay), many
  coins had no perp, and daily rebalancing at 0.2-0.3% per side eats the rest.
- Almost all remaining edge is in **shorting coins that pumped into the liquid set and then bleed**. On
  consistently liquid majors the model has no edge after costs.
- Edge persists after the close (IC 0.095 for 01:00 -> 24:00), so it is not just bid-ask bounce at 00:00.

## Caveats (why this is not yet a money printer)
- ~50 variants were tried on the same 2023-2026 window: data snooping is real. No untouched holdout remains.
- Daily close-to-close simulation ignores intraday squeezes on shorts; a +200% wick on one of 10 shorts at
  0.5x gross is about -10% NAV. Slippage on pumped small caps can exceed 0.3%.
- 2025 (alt crash) contributes a large share of the profit; 2023 was flat.

## Reproduce
```
D=/path/to/data
python3 download.py --out $D --interval 1d
python3 build_features.py $D
python3 quick.py $D            # XGBoost walk-forward + spot/LS baselines
python3 list_perps.py $D/perps.json && python3 fetch_funding.py $D
python3 run_futures.py $D && python3 run_pump_short.py $D && python3 run_robust.py $D
```
