# Plan klađenja — tenis, početne kvote

Sažetak svega što je testirano (detalji i brojevi su u `README.md`). Ovo je plan za probu, ne garancija zarade.

## 1. Šta se igra
- **Da:** ženski ITF (W15–W100), WTA125, WTA Tour, muški Challenger, muški ITF — samo pobednik meča.
- **Ne:** ATP Tour (nema prednosti), totali gemova, NBA i Evroliga (testirano, gubi).

## 2. Kada
- **Odmah kad kladionica objavi početnu kvotu** (obično dan pred meč). Što ranije, to bolje: na BetMGM 2025–26
  opklade na kvote objavljene više od 6 sati pre dana meča dale su +15%, a one objavljene na dan meča +2,5% do +8%.
- Na završnoj kvoti iste opklade gube u svim testovima — cela prednost je u tajmingu. Do meča kvota na našu
  stranu padne u proseku 5–10%, a kod nasumičnih strana 0%.

## 3. Gde
- Kladionica sa „mekanim“ početnim kvotama. Najbolja u istorijskim podacima je **Bet365** (2020–2025, +4,4% do
  +8,6% zavisno od praga, plus svake godine). Na svežim podacima 2025–26 potvrđeno na **BetMGM**.
- Stake: nije testirano (nema istorijskih kvota sa ovog servera; za tenis BetExplorer odavde pokazuje samo BetMGM).

## 4. Kako se računa verovatnoća
1. Model (481 feature, bez kvota) daje verovatnoću pobede `q`.
2. Iz obe početne kvote izračuna se fer verovatnoća tržišta Shinovom metodom (`shin`).
3. Konačna verovatnoća: `P = σ(0,44·logit(q) + 0,64·logit(shin))`. Težine su naučene samo na Bet365 2014–2024.
   Ovaj korak koristi kvotu, ali samo pri odluci o opkladi, a ne kao feature modela.

## 5. Pravilo
- Igra se kad je `P · kvota − 1 > 5%`.
- **Oprezna varijanta:** kvota ≤ 2. **Agresivnija:** kvota ≤ 5.

| Varijanta (BetMGM 2025–26, podaci koje ništa nije videlo) | Opklada | ROI | Donja granica 90% | Najgori slučaj* |
|---|---|---|---|---|
| EV>5%, kvota ≤ 2 | 1.766 (~3,5 dnevno) | +6,5% | +3,7% | +3,4% |
| EV>5%, kvota ≤ 5 | 2.626 (~5 dnevno) | +9,0% | +5,9% | +4,0% |
| EV>10%, kvota ≤ 2 (~1 dnevno) | 458 | +9,1% | +3,4% | +3,9% |

Brojevi su za model treniran sa manjim learning rate-om (malo bolji od prvobitnog na svim glavnim pravilima).

\* Kao da kladionica poništi svaku opkladu čija je fer verovatnoća do početka meča porasla za više od 12 poena.

## 6. Ulog
- **Preporuka:** Kelly × 0,10, najviše 2% banke po opkladi (ili ravno 1% početne banke).
- Simulacija na 2025–26: Kelly × 0,10 → tipičan najveći pad oko 20%, u 95% slučajeva ispod 30%.
- Kelly × 0,25 daje padove 35–50%; pola Kelly-ja (padovi 55–75%) je preagresivno za model sa nesigurnošću.

## 7. Rizici
- **Ograničenja naloga:** kladionice brzo smanje ulog nalozima koji stalno pobeđuju početne kvote. Ovo je
  najveći praktični rizik; zarada u jedinicama zavisi od toga koliko dugo nalog prima pune uloge.
- **Mali uzorak 2026:** 1.549–2.526 opklada; prava vrednost može biti niža.
- **Poništene opklade:** kod velikih grešaka u kvoti kladionica može da poništi opkladu („palpable error“).
- **Drift modela:** model se mora ponovo trenirati svake godine (walk-forward, kao u testovima).

## 8. Šta još treba napraviti pre pravog novca
1. **Svež izvor rezultata:** Sackmann podaci kasne, a model mora da zna jučerašnje mečeve (Elo, forma, umor).
   Treba dnevno skidati rezultate (npr. BetExplorer/TennisExplorer) i preračunavati feature-e.
2. **Dnevna skripta:** raspored mečeva + početne kvote → feature-i → `q` → `P` → lista opklada.
3. **Probni period 4–8 nedelja bez novca:** beležiti opklade po početnoj kvoti i proveriti da je prosečan pad kvote
   (CLV) na našim stranama pozitivan, kao u testovima (+5–10%). Tek onda mali ulozi.
