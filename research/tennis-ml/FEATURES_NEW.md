# FEATURES_NEW: 138 new per-player features (results-only)

Selected from 214 candidates by five researchers. Three implementation modules: ratings (48), score (44), context (46).

## Rules

- Bookmaker odds are never features.
- Results-only: only date/tourney_date, round, tourney_id/name, surface, tourney_level, best_of, score string, retired/walkover, ids, hand, height, ioc, age. No serve/return stats, ranking, minutes, entry, seeds, draw size.
- Strictly pre-match: use only rows that come before the match in 'order' (date, tourney_id, round order, match_num). Never the match's own result or later rounds of the same event. Ratings recorded BEFORE the update.
- Every feature is per player: computed for winner and loser, used as p1, p2 and diff = p1 - p2. "Perspective" features are pair-level ideas rewritten as value of the player vs this opponent (p2 gets the mirrored value, negated for antisymmetric ones, 1-p for probabilities). "Match-level" features are the same for both players (diff = 0, use p1 only).
- Do not modify existing files; new code goes in new modules.

## Shared conventions

- P(N): the player's last N earlier matches in 'order' (any level, walkovers excluded, retirements included unless stated). Ps(N): same restricted to the surface of the current match. Pd(D): earlier matches with date in (cur_date - D days, cur_date].
- SR(x,n,p0,m) = (x + m*p0)/(n + m) (shrinkage toward prior p0 with weight m). If n = 0 it returns p0.
- Clean score parse C1: split the score on spaces, drop RET/W/O/DEF tokens. Match tiebreak set (MTB) = a set with max >= 10 or in brackets: counts as one set won/lost, 0 games for game-share stats, 3 games in load stats. Complete set = max >= 6 and (diff >= 2 or max == 7). The unfinished last set of a RET match counts in load games but not in set stats. A 7-6 set = 13 games in load stats. games_gw/gl = games won/lost in complete non-MTB sets. Parse errors give NaN rows.
- elo_exp_k = stored pre-match win probability of the existing standard Elo for each earlier match k of the player; resid_k = won_k - elo_exp_k; oelo_k = opponent's pre-match standard Elo at match k; own_elo_k = own pre-match Elo. All produced in the same sequential pass.
- est_day = tourney_date + min(6, rord_cur - min rord of earlier-ordered matches in the same tourney_id) (0 for the first match of an event); needed because date is only the event start. Check that rord increases with round progression.
- event_country(E): (a) 3-letter code parsed from tourney_id when it contains 'ITF-XXX' or 'FC-...-XXX'; else (b) learned host country: key = normalised tourney_name (lower case, strip W15/W25/M15/H/+H, trailing integers, ', ST'), value = modal player ioc over earlier editions (tourney_date strictly earlier), needing >= 2 editions or >= 8 distinct players with modal share >= 0.35; else (c) modal ioc of players already seen in earlier-ordered matches of the same tourney_id (>= 6 distinct, share >= 0.35); else unknown. Static table ioc -> (lat, lon, UTC offset, continent) for about 200 codes.
- Previous event of a player = last tourney_id with tourney_date strictly smaller than the current one. "No prev event" = none within 21 days.
- Level groups: LG1 low ITF {10,15,25,S}, LG2 mid ITF {35,35+H,40,50,50+H,60,75,80,100}, LG3 Challenger/WTA125 {C}, LG4 tour (rest).

## Module: ratings (48)

| name | definition | source |
|---|---|---|
| g2_rating | Glicko-2 rating pre-match in Elo units (mu*173.7178+1500). One chronological pass in 'order', one match = one rating period. Init mu=0, phi=350/173.7178, sigma=0.06, tau=0.5. Standard single-game update (Illinois volatility iteration, eps 1e-5), opponent mu/phi pre-update; s=1/0, for retired matches winner s=0.75, loser s=0.25. Inactivity: phi_pre=min(sqrt(phi^2+0.15*days_since_last/30), 350/173.7178) with days from 'date'; use and update from phi_pre. Record mu before update. Debutant 1500. | Glickman 2001, proto2.py |
| g2_rd | Glicko-2 deviation at match time in Elo units: phi_pre*173.7178 (inactivity-inflated). Debutant 350. | Glicko-2 |
| g2_lb | g2_rating - 2*g2_rd (pre-match). Debutant 800. | own, on Glicko-2 |
| g2_vol | Glicko-2 volatility sigma after the player's last match; 0.06 for debutants. | Glicko-2 |
| g2_surf_lb | Separate Glicko-2 state per (player, surface, Capitalised). First appearance on a surface: mu = overall mu, phi = max(overall phi, 1.0), sigma = 0.06. Updated only by matches on that surface, same rules. Value = (mu_s*173.7178+1500) - 2*phi_s_pre*173.7178 for the current surface. | own, on Glicko-2 |
| g2_z | Perspective, antisymmetric: (g2_rating_self - g2_rating_opp)/sqrt(g2_rd_self^2 + g2_rd_opp^2). | own |
| g2_prob | Perspective: P(self wins) = 1/(1+exp(-g(sqrt(phi_s^2+phi_o^2))*(mu_s-mu_o))), g(phi)=1/sqrt(1+3*phi^2/pi^2), inflated pre-match phi in mu units. p2 = 1-p1. | Glickman 2001 |
| ts_mu | TrueSkill 1v1 no-draw mean mu pre-match. mu0=25, sigma0=25/3, beta=sigma0/2, tau=sigma0/100. Before each match every participant var += tau^2 + 0.0144*days_since_last/30, capped at sigma0^2. Update: c^2=2*beta^2+var_w+var_l; t=(mu_w-mu_l)/c; v=pdf(t)/cdf(t); w=v*(v+t); mu_w+=var_w/c*v; mu_l-=var_l/c*v; var_x*=1-var_x/c^2*w. | Herbrich 2006, proto_ts.py |
| ts_sigma | sqrt(var) pre-match after inactivity inflation (range 0.5-8.33). | TrueSkill |
| ts_z | Perspective: (mu_self-mu_opp)/sqrt(sigma_self^2+sigma_opp^2). | own |
| ts_prob | Perspective: Phi((mu_s-mu_o)/sqrt(2*beta^2+sigma_s^2+sigma_o^2)). p2 = 1-p1. | TrueSkill |
| ow_elo | Elo as in existing ratings() (K=250/(n+5)^0.4 on own n, retirement multiplier 0.5, init 1500) but each player's update is multiplied by min(1, (n_opp_before+2)/15), n_opp_before = opponent's prior matches. Record pre-match rating. | own |
| lvl_elo | Elo like ratings() but new players start at S_g(current level group) - 75 instead of 1500. S_g per level group (LG1..LG4): after each match, for each participant with n_career >= 10, S_g += 0.0005*(pre-match rating - S_g), init 1500. Record pre-match rating. | own |
| elo_vs_level_mean | existing standard elo minus S_g of the current match's level group (S_g as in lvl_elo, n_career >= 10 participants, alpha 0.0005). | own |
| lg_elo | Elo per (player, level group LG1..LG4). First appearance in a group: initialise from the player's overall standard Elo at that time. K=250/(n_group+5)^0.4, n_group = player's matches in that group; retirement multiplier 0.5. Expected score uses both players' ratings in the current match's group. Feature = player's rating in the current group. | own |
| lg_elo_minus_elo | lg_elo - standard elo (0 on the first match in the group). | own |
| cs_elo | Cross-surface Elo: 4 ratings per player (Hard, Clay, Grass, Carpet; surface Capitalised), init 1500. Expected score from both players' ratings on the current surface. After the match each of the 4 ratings of each player is updated by rho[s_match][s']*K*mult*(result-e), K=250/(n_career+5)^0.4, mult=0.5 for retirement, rho rows (Hard,Clay,Grass,Carpet) = [1,.75,.65,.85],[.75,1,.5,.6],[.65,.5,1,.75],[.85,.6,.75,1]. Feature = rating on the current surface pre-match. | Sackmann surface Elo, proto_ratings.py |
| layoff_elo | Elo with layoff handling. Start 1500 if first seen in a main=1 file, else 1250. K=250/(n+5)^0.4 times m_k, m_k = 1.55 on the first match after a layoff decaying linearly to 1.0 over the next 20 matches. Layoff = gap >= 56 days between consecutive matches of the player where the gap does not start between Oct 15 and Dec 31. At the first match back rating -= min(100 + 1.5*(gap_days-56)/7, 200), applied before the expectation. Feature = pre-match rating. | Sackmann Elo, Covid post |
| hind_30 | For the player's last 30 prior opponents (deque storing opp_id and opp Elo at match time): mean of (opponent's CURRENT pre-this-match standard Elo - opponent Elo at the time they played the player). Min 1 prior match, else NaN. | own |
| hind_10 | Same as hind_30 over the last 10 prior opponents (min 1). | own |
| prov_opp_20 | Share of the last 20 prior opponents who had fewer than 5 career matches when they played the player. Min 1 prior match, else NaN. | own |
| sos2_10 | Mean over the last 10 prior opponents of that opponent's own oppelo_10 (mean Elo of the opponent's previous 10 opponents) as stored at the moment of their match with the player. Min 1, else NaN. | own |
| bestwin_20 | Maximum opponent pre-match Elo among the player's wins in P(20). NaN if no win. | own |
| worstloss_20 | Minimum opponent pre-match Elo among the player's losses in P(20). NaN if no loss. | own |
| elo_chg_90d | Pre-match standard Elo now minus the player's pre-match Elo at the latest earlier match with date <= cur_date-90 days (as-of lookup); if none that old use the first-ever pre-match Elo (1500). NaN if no prior match. | own |
| elo_chg_365d | Same as elo_chg_90d with 365 days. | own |
| elo_slope_180d | OLS slope (Elo points per 30 days) of the player's pre-match Elo (including the current pre-match value at cur_date) against date over the prior 180 days. Needs >= 4 points spanning >= 21 days, else NaN. | own |
| elo_vol_20 | Std of successive changes of the player's pre-match Elo over the last 20 prior matches (diffs of last 21 pre-match values incl. current). Needs >= 3 diffs, else NaN. | own |
| elo_frompeak | Current pre-match standard Elo minus the running career maximum of the player's pre-match Elo (init 1500). <= 0. 0 if no prior match. | own |
| ewm_resid_120 | sum_k w_k*resid_k/(sum_k w_k + 3), w_k = 0.5^((cur_date-date_k)/120), over earlier matches within 730 days. Fallback 0. | Sipko time discounting |
| pagerank_365 | PageRank on a directed graph per tour (WTA, ATP; all levels pooled): edge loser->winner, weight exp(-age_days/120), age from tourney_date, using matches in the 365 days before the event's first match (snapshot per distinct tourney_date, only matches with order < first match of the event). Damping 0.85, uniform teleport over nodes, 30 power iterations warm-started from the previous snapshot. Value = ln(PR*N), N=nodes. Player not in graph: ln(0.15). | Radicchi 2011; Dingle et al 2012 |
| pagerank_pct90 | Percentile (0-1) of the player's PR among nodes with a match in the last 90 days at the same snapshot. Unseen player 0.15. | own |
| co_win_diff | Perspective, antisymmetric. C = opponents (excluding each other) both players faced in the 730 days before the match. For each c: a_i(c) = (sum w*win + 0.5)/(sum w + 1) with w=0.5^(days/180) over player i's matches vs c. Feature = mean over c of (a_self(c) - a_opp(c)). Needs >= 2 common opponents else 0. | Knottenbelt 2012 |
| co_margin_diff | Same C as co_win_diff. m_i,c = recency-weighted (half-life 180d) mean of (gw-gl)/(gw+gl) over player i vs c (C1 parse, completed non-MTB games). Feature = sum_c w_c*(m_self,c - m_opp,c)/sum_c w_c, w_c = min(n_self_c, n_opp_c, 3). Needs >= 2 common opponents else 0. | Knottenbelt 2012 |
| co_elo_adj_diff | Same C. For each c: mean resid (won - elo_exp) of the player's matches vs c, decay half-life 270d. Feature = weighted mean over c (weights min(n_self_c, n_opp_c, 3)) of (resid_self,c - resid_opp,c). Needs >= 2 common opponents else 0. | own on Knottenbelt |
| co_count | Match-level: log1p(number of common opponents in the last 730 days), symmetric. Gate for co_* features. | Knottenbelt 2012 |
| trans_2hop | Perspective, antisymmetric. Within 730 days: chains self beat x and x beat opp, minus chains opp beat x and x beat self (x distinct, direct meetings excluded), each chain weighted exp(-age_of_older_match_days/365). Value = diff/(total chains + 2). 0 if no chains. | Spanias & Knottenbelt 2013 |
| mf_latent_matchup | Online logistic matrix factorisation: logit P(i beats j) = ln10/400*(elo_i - elo_j) + u_i.v_j - u_j.v_i, k=4, u,v ~ N(0,0.01) (fixed seed), SGD lr 0.05, L2 0.05 on the two players' u,v, one step per match after it is observed. Value = u_self.v_opp - u_opp.v_self before the match (antisymmetric). 0 if either player has < 5 prior matches. | Chen & Joachims 2016 |
| vs_higher_resid | Over P(40) restricted to matches where opponent pre-match Elo >= own pre-match Elo + 50: sum_k w_k*resid_k/(sum_k w_k + 4), w_k = 0.5^(age_in_matches/20). If fewer than 3 such matches: 0. | own |
| vs_lower_resid | Same as vs_higher_resid for matches where opponent Elo <= own Elo - 50. | own |
| band_resid | Opponent-gap band of the current match from the player's view: elo_diff = own - opp pre-match Elo, bands (<-150, -150..-50, -50..50, 50..150, >150). Value = sum w*resid / (sum w + 6) over the player's matches in the last 3 years that fall in the same band (own - opp Elo at that match), w=0.5^(days/365). < 4 matches in band: 0. | own |
| sos_gap_90d | Weighted mean (w=exp(-age_days/45)) of opponent pre-match Elo over the player's matches in Pd(90) minus the player's current pre-match Elo. Needs >= 2 matches else 0. | Massey SOS |
| oqw_form_10 | Over P(10): sum_k w_k*(won_k - 0.5)*z_k / sum_k w_k, w_k = 0.9^k (k=0 most recent), z_k = clip((oelo_k - mean_oelo_level_group)/std_oelo_level_group, -3, 3) with mean/std over the level group's matches in the previous 365 days. < 3 matches: 0. | own |
| quality_win_rate_365 | Over Pd(365) with w=exp(-age_days/180): (sum w over wins with oelo >= own_elo_k  -  sum w over losses with oelo <= own_elo_k - 100)/(n_matches + 3). 0 if no matches. | own |
| res_vs_high_tb_opp | Tiebreak propensity of a player: SR(sets with a tiebreak, completed sets, 0.12, 10) over his matches in the last 24 months. An opponent is high-TB if his propensity (at the time of the past match) > 0.20. For the current opponent's bucket: sum resid / (n+5) over the player's matches in the last 24 months vs opponents of that bucket. < 5 such matches: 0. | own |
| sim_opp_resid | Mean resid (shrunk: sum resid/(n+5)) of the player's matches within 730 days against opponents whose pre-match Elo then was within +/-100 of the CURRENT opponent's current Elo. < 5 matches: 0. | own |
| h2h_resid | Perspective, antisymmetric. sum over prior meetings of (won_self - elo_exp_self at that meeting)/(n_meetings + 2). 0 if none. | Sipko h2h; own |
| h2h_decay_shrunk | Perspective. (sum w*won + 0.5*1)/(sum w + 1) over prior meetings, w = 0.8^(years_since) * (1 if same surface else 0.6). 0.5 if none. p2 = 1-p1. | Sipko DIRECT feature |

## Module: score (44)

| name | definition | source |
|---|---|---|
| gshare_clean_30 | Games-won share over P(30) using C1 (MTB sets excluded, unfinished RET set excluded): sum gw/sum(gw+gl). NaN if < 12 games. | own (fixes MTB distortion) |
| comeback_resid_100 | Over P(100) restricted to matches where the player lost set 1 (set 1 complete, includes RET after set 1): x = sum(won - q1(elo_exp)), n = count; value x/(n+4). q1(e)=P(win match given lost set 1) logistic fit on training years only (about 0.16 at e=0.5). n=0: 0. | own |
| closeout_resid_100 | Same for matches where the player won set 1: x = sum(won - q2(elo_exp)), q2 about 0.84 at e=0.5, value x/(n+6). n=0: 0. | own |
| rebound_set_100 | Over P(100), complete non-MTB sets k>=2: prevlost = lost set k-1, a = sum(won set k and prevlost), n = sum(prevlost); s = SR(sets won, completed sets, 0.5, 10) over same window. Value = (a+4*s)/(n+4) - s. n=0: 0. | own |
| dec_win_resid_100 | Over P(100) restricted to completed non-retired matches that went the full distance (3 sets in Bo3, 5 in Bo5, MTB counts as set): x = sum(won - elo_exp), value x/(n+5). n=0: 0. | own |
| dec_share_50 | Over P(50) non-retired matches: SR(deciding-set matches, n, p0, 8), p0 = 0.26 WTA ITF, 0.31 Challenger/tour, 0.28 otherwise. Fallback p0. | own |
| tb_play_rate_100 | Over P(100): SR(sets decided 7-6/6-7, completed non-MTB sets, p0, 15), p0=0.10 women, 0.15 men. Fallback p0. | own (hold-rate proxy) |
| tb_win_shr_100 | Over P(100): (tiebreak sets won + 4)/(tiebreak sets played + 8). 0.5 if none. | own |
| close_set_win_100 | Over P(100): SR(wins in close sets, close sets, 0.5, 8); close set = 7-5, 5-7, 7-6, 6-7. Fallback 0.5. | own |
| straight_win_share_50 | Over P(50), wins not by retirement: SR(wins without dropping a set (MTB loss counts as dropped), those wins, p0, 6), p0=0.72 (0.66 if Challenger/tour). Fallback p0. | own |
| loss_resist_50 | Over P(50) losses (not retired, not W/O): SR(losses in which the player took >= 1 set, losses, p0, 6), p0=0.30 (0.27 women ITF, 0.34 Challenger). Fallback p0. | own |
| dom_gl_per_set_won_50 | Over P(50): (games lost in sets won + 8*3.1)/(sets won + 8) on complete non-MTB sets. Fallback 3.1. | own |
| fight_gw_per_set_lost_50 | Over P(50): (games won in sets lost + 8*3.2)/(sets lost + 8) on complete non-MTB sets. Fallback 3.2. | own |
| blowout_for_rate_100 | Over P(100): SR(sets won 6-0 or 6-1, completed non-MTB sets, p0, 15), p0=0.12 women, 0.07 men. | own |
| blowout_against_rate_100 | Over P(100): SR(sets lost 0-6 or 1-6, completed non-MTB sets, p0, 15), p0 as blowout_for_rate_100. | own |
| luck_match_40 | Over P(40): w = SR(wins, n, 0.5, 4); g = SR(games won, games, 0.5, 40) (C1); value = w - f(g), f logistic map from cumulative game share to win probability fit on training years only. n < 5: 0. | own (Pythagorean idea) |
| luck_set_40 | Over P(40): SR(sets won, completed sets, 0.5, 8) - h(g), g as above, h(g)=clip(0.5+1.7*(g-0.5),0,1). n < 5: 0. | own |
| gshare_std_20 | Std of per-match game share gw/(gw+gl) (C1, matches with >= 8 games) over P(20). Needs >= 6 valid matches else NaN. | own |
| gshare_q25_20 | 25th percentile of per-match game share over P(20), needs >= 6 valid matches else NaN. | own |
| gshare_up_50 | Over P(50), matches vs opponents with pre-match Elo > own: (sum gw_up + 30*g_all)/(sum games_up + 30), g_all = game share over all of P(50) (C1). Fallback g_all, 0.5 if empty. | own |
| gshare_down_50 | Same for opponents with Elo < own. | own |
| fs_win_resid_50 | Over P(50) with complete set 1: sum(won_set1 - p1set(elo_exp))/(n+5); p1set(e) = P(win set 1 given e) fit on training years only. 0 if none. | own |
| start_delta_50 | Over P(50): SR(games won, games, 0.5, 20) in set 1 only minus SR(...) in sets >= 2 (complete non-MTB). 0 if no data. | own |
| mtb_win_career | Career (all earlier matches) record in third-set match tiebreaks: (wins + 1.5)/(n + 3). 0.5 if none. | own |
| ret_me_rate_100 | Over P(100): SR(matches in which the player retired, n, 0.03, 25). | own |
| prev_ret_me | 1 if the player's immediately preceding match ended with the player himself retiring; else 0; 0 if no previous match. | Sipko RETIRED |
| days_since_own_ret | min(60, cur_date - date of the player's last own retirement) in days; 60 if none. | own |
| prev_win_by_ret | 1 if the player's previous match was won because the opponent retired; else 0. | own |
| event_games | Total load games (clean, incl. partial RET set, MTB = 3) played by the player in the current tourney_id in earlier-ordered matches (qualifying incl.). 0 if none. | own |
| event_deciders | Number of full-distance matches the player played in the current tourney_id before this match. | own |
| event_gshare_delta | SR(event games won, event games played, g30, 12) - g30, g30 = gshare_clean_30 (pre-match, 0.5 if NaN), games from earlier matches of this tourney_id. 0 if none. | own |
| prev_games | Total load games of the player's previous match (any tournament, excluding walkovers). NaN if none. | Tennis Abstract time-on-court proxy |
| prev_gshare_clean | Game share (C1) of the player's previous match. NaN if no games. | own |
| games_3d | Sum of load games over the player's matches with date in (cur_date-3, cur_date) i.e. earlier events and event days strictly before the event start window; matches of the current tourney_id are excluded (see event_games). 0 if none. | own |
| games_10d | Same with a 10-day window. | own |
| fatigue_games_decay3d | Sum over earlier matches k with 0 < days(cur)-days(k) <= 3 of load_games_k * 0.75^(days(cur)-days(k)); same-day (same tourney date, earlier order) matches get weight 1.0. 0 if none. | Sipko fatigue |
| fatigue_games_decay7d | Same with window 7 days and decay 0.85/day. | Sipko fatigue |
| prev_marathon | 1 if the previous match went the full distance and its deciding set had >= 13 games (7-6, 8-6, long set) or was a tiebreak decider in a best-of-3/5; else 0. | Tennis Abstract hangover effect |
| heavy_matches_7d | Number of matches in the last 7 days with total load games >= 28 (Bo3) or >= 40 (Bo5) plus 0.1*sum over those 7 days of max(0, games-22). 0 if none. | own on Sipko |
| games_per_set_30 | Over P(30), non-retired: SR(total C1 games, completed non-MTB sets, 9.5, 4) i.e. (games + 4*9.5)/(sets + 4). | Newton-Keller proxy |
| win_clean_50 | Over the last 50 completed matches (not W/O, not RET either way): (wins + 1.5)/(n + 3). 0.5 if n < 3. | Sackmann |
| noisy_share_20 | Share of P(20) plus walkovers recorded for the player that were walkovers or retirements (either side). 1 if < 3 matches. | own |
| h2h_last_gshare | Perspective. Game share (C1) the player achieved in the most recent earlier meeting vs this opponent minus 0.5. NaN if no meeting. p2 gets the negated value. | own |
| lvl_gshare_clean_30 | Over the last 30 earlier matches in the same 'main' group (main=1 vs 0) as the current match: SR(games won, games, 0.5, 24) with C1. NaN if none. | own |

## Module: context (46)

| name | definition | source |
|---|---|---|
| ctx_ev_country_change | 1 if event_country(prev event) != event_country(current event); 0 if equal; NaN if either unknown. If no prev event within 60 days: 0 and the flag ctx_no_prev_event=1 is also emitted. | own |
| ctx_travel_km_log | log1p(haversine km between centroids of event_country(prev event) and current event_country), only if the previous event's last match is <= 21 days before the current tourney_date; else training median. | own |
| ctx_tz_abs | abs(UTC offset(current event country) - UTC offset(prev event country)) in hours (standard offsets, centroid longitude/15 for multi-zone countries), only when prev event ended <= 21 days before; else 0. | own |
| ctx_tz_east | Signed (cur - prev) UTC offset in hours (positive = travelled east), same conditions; else 0. | own |
| ctx_dist_home_km_log | log1p(haversine km between the player's ioc centroid and event_country centroid); 0 at home; training median if unknown. | own |
| ctx_same_continent | 1 if continent of the player's ioc equals continent of event_country (home included); 0 otherwise or unknown. | own |
| ctx_country_streak_weeks | Number of consecutive immediately preceding events (gap between events <= 14 days) played in the same event_country as the current event (0 if the prev event was elsewhere or gap larger); cap 8. | own |
| ctx_country_share_30 | Share of P(30) played in the current event_country. 0 if n_career < 5 or country unknown. | own |
| ctx_country_resid_shrunk | sum of resid over the player's earlier matches in the current event_country / (n + 6). 0 if unknown. | own |
| ctx_consec_weeks | Number of consecutive weeks (back from tourney_date - 7, no empty week) with at least one match of the player; events with the same tourney_date as the current one not counted. Cap 12; 0 at debut. | Sipko fatigue, adapted |
| ctx_weeks_active_8 | Number of distinct weeks with >= 1 match among the 8 weeks before the current tourney_date (0-8). | own |
| ctx_prev_ev_deep | Depth reached in the previous event (qualifying incl.): (rord_first - rord_last_round_reached)/(rord_first - rord_F) clipped to [0,1] (1 = final or title). 0 and ctx_no_prev_event=1 if none within 21 days. | own |
| ctx_prev_ev_games | Load games the player played in the previous event (0 if none within 21 days). | own |
| ctx_prev_ev_title | 1 if the player's last match in the previous event was a win in round F; else 0. | own |
| ctx_load_ewm4 | Sum over earlier matches of load_games_i * exp(-(est_day_cur - est_day_i)/4), divided by 10. 0 if none. | own |
| ctx_load_ewm10 | Same with exp(-d/10). | own |
| ctx_win_streak | Consecutive wins immediately before this match (walkover wins excluded, retirement wins count), cap 15. 0 if last match was a loss. | own |
| ctx_loss_streak | Consecutive losses immediately before this match, cap 15. | own |
| ctx_days_since_loss | log1p(min(365, est_day_cur - est_day of the last loss)); log1p(365) if never. | own |
| ctx_days_since_win | Same for the last win. | own |
| ctx_post_layoff_idx | Layoff = gap >= 42 days (est_day) between consecutive matches. Number of matches played since returning from the last layoff (0 = first match back), cap 8; 8 if no layoff. | own |
| ctx_last_layoff_len_log | log1p(length in days of the player's last layoff); 0 if none. | own |
| ctx_comeback_resid_hist | sum of resid over the player's earlier matches that were among the first 2 after a layoff >= 42 days, / (n + 4). 0 if none. | own |
| ctx_surf_days_since | log1p(min(730, days between cur date and the last match on the current surface)); log1p(730) if never. Surface: Clay/Hard/Grass; Carpet mapped to Hard. | Sipko surface weighting |
| ctx_surf_switch | 1 if the surface of the previous event differs from the current surface and the previous event ended <= 21 days before; else 0. | own |
| ctx_surf_share_365d | (matches on current surface in Pd(365))/(all matches in Pd(365)). NaN if no matches. | own |
| ctx_indoor | Match-level. 1 if lower-case tourney_name matches indoor/hall/halle/arena/dome/palais, or surface == Carpet, or (surface in Hard/Clay and event_country centroid latitude >= 44 and month in 10-3); else 0. | own |
| ctx_season_phase | Match-level categorical 0-8 from ISO week of tourney_date, surface and continent: 0 Australian summer hard (wk1-8), 1 sunshine hard (wk9-14), 2 clay swing (wk15-23 Clay), 3 grass (wk24-28), 4 US summer hard (wk29-36), 5 Asian autumn (wk37-42, Asia), 6 indoor autumn (wk37-48 Europe or ctx_indoor), 7 South American clay (wk1-14 Clay S.America), 8 off-season (wk49-52). First rule that matches wins; else 8. | own |
| ctx_lvl_rank_cur | Match-level ordinal level rank: ITF 10/15=1, 25=2, 35=3, 50/50+H=4, 60/75=5, 100=6, C=7 (8 if name contains 100/125/175), P/A/250-500=9, M/I/G/F/1000=10-11, S=2. | own |
| ctx_lvl_delta_prev | lvl_rank_cur minus level rank of the player's previous event; 0 if none within 21 days (flag ctx_no_prev_event). | own |
| ctx_lvl_step_above_max | lvl_rank_cur minus max level rank of any earlier event of the player (qualifying counts); 0 if no history. | own |
| ctx_first_at_level | 1 if the player has no earlier match in this level group (LG1..LG4); else 0. | own |
| ctx_years_at_level | (est_day_cur - est_day of first match in this level group)/365; 0 if first. | own |
| ctx_age_debut | Age at the player's first match in the data (min age over earlier matches, else current age - years_active), clipped 13-35. Left-censored careers (first match before 1990): training median. | own |
| ctx_career_slope | (max level rank so far - level rank of first event)/max(0.5, years_active), levels per year. | own |
| ctx_age_peak_gap | age - 24.5 (WTA) or age - 26.5 (ATP); 0 if age missing. | own |
| ctx_age_vs_level_z | (age - mean age)/std age of players in the current level group over matches in the previous 365 days (earlier rows only, min 200 matches else all-time). 0 if age missing. | own |
| ctx_elo_vs_event | existing standard elo minus mean Elo of distinct players already seen in earlier-ordered matches of the same tourney_id (min 6 players, else the level group's mean Elo of the previous 365 days). | own |
| ctx_q_matches | Number of qualifying-round matches (Q1-Q4) the player has already played in the current tourney_id, cap 4. | own |
| ctx_q_games | Load games in those qualifying matches. | own |
| ctx_lucky_loser | 1 if the player lost his last earlier-ordered qualifying match in this tourney_id and the current match is a main-draw round. | own |
| ctx_bye_proxy | 1 if the current match is the player's first in the tourney_id, is a main-draw round, and the event already has earlier-ordered main-draw matches in a lower rord (earlier round); else 0. | own |
| ctx_def_depth | Depth in [0,1] reached by the player in the edition of the same normalised tourney_name and level group 300-430 days earlier (as in ctx_prev_ev_deep); -1 if not played. | own |
| ctx_beyond_last_year | 1 if the current round is deeper than the player's deepest round last year (same event key); 0 if not; -1 if he did not play. | own |
| ctx_n_events_90d | Number of distinct tourney_id (qualifying incl.) the player played in the 90 days before the current tourney_date, current event excluded. | own |
| ctx_titles_365d | Number of round-F wins in Pd(365). | own |
