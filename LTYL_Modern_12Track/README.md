# Love The Way You Lie — Modern Rework (15 tracks)

A fully synthesized, mixed and mastered 64-bar dark trap/drill instrumental.
Started as 12 tracks; T13 (lead synth, later rebuilt as piano) was added, then
T15 (solo violin) joined in the emotional-arc pass documented below. Output
filenames keep the original `12Track` name so existing links stay valid.
90 BPM, G minor, 48 kHz / 24-bit stereo, peaks at exactly **−6.00 dBFS** so
there is headroom to record raw vocals on top.

```
cd LTYL_Modern_12Track
python3 -m venv venv && ./venv/bin/pip install pydub numpy scipy pedalboard requests librosa mido
./venv/bin/python build.py       # renders LoveTheWayYouLie_Modern12Track.wav + stems/
./venv/bin/python analyze.py     # QA pass + analysis.png
./venv/bin/python midi_export.py # writes midi/*.mid for Logic Pro re-tracking
```

---

## Phase 0 — sonic profiling

**The 2010 original** (Eminem ft. Rihanna, prod. Alex da Kid) is a mid-tempo
hip-hop ballad in **G minor around 84 BPM**, built on a melancholic
acoustic-guitar figure with piano and strings, and stadium-sized, reverb-heavy
rock drums. Its emotional weight comes from the contrast between a fragile,
looping acoustic bed and drums that hit like an arena — and from Eminem's raw,
un-tuned, angry delivery sitting right in the middle of it.

**Modern dark drill/trap** works the opposite way round: a colder, grayscale
palette, eerie bell and pluck motifs, sparse and clinical drums, aggressively
distorted 808s with portamento slides, and lo-fi filtering that keeps the top
end dull so the vocal owns the presence range.

**Translation strategy.** Keep what carries the emotion — the G minor
i–VI–III–VII loop, a guitar as the melodic anchor, the huge ambient keys —
and replace the delivery mechanism entirely:

| Original | This rework |
| --- | --- |
| ~84 BPM stadium-rock kit | 90 BPM trap grid, dry rimshot on beat 3 |
| Big reverb on everything | Reverb on the *beds only*; snare bone dry |
| Live bass / low piano | Distorted sine 808 with 60 ms glides |
| Acoustic guitar, full-range | Electric guitar, notched at 2.4 kHz and capped at 2.8 kHz |
| Dense, produced center | 1–5 kHz carved open for raw, pitch-correction-free rap |

The 1–5 kHz band is deliberately the quietest region of the mix (see below).
Everything that would normally live there — guitar top end, piano presence, pad
shimmer, vocal textures — is filtered or shelved out of the way, so an
unprocessed, aggressive vocal can sit in the middle without fighting anything.

---

## The 15 tracks

Everything is synthesized from first principles rather than sampled, so the whole
kit is tuned to G minor by construction instead of pitch-shifted into key after
the fact. `build.py` first *attempts* to fetch CC0 source material over HTTP and
falls back to synthesis for anything it can't retrieve (in practice: all of it —
the run log reports what was fetched).

| # | Track | Synthesis method | DSP chain |
| --- | --- | --- | --- |
| T1 | Electric Guitar | Karplus–Strong with solid-body sustain (0.9993 loop decay), magnetic-pickup comb at p=0.22, passive LC resonance @2.7 kHz, cabinet rolloff, amp breakup | LP 2.8 kHz, 2.4 kHz notch, chorus, wash |
| T2 | Ambient Synth | FM bell (fast-decaying index) + detuned triangle body, filtered dark | HP 150 Hz ×2, large hall (room 0.92, 40 % wet) |
| T3 | Synth Pad | 3× detuned additive saw stack, slow filter bloom | LP 3 kHz ×2, per-channel chorus + M/S widening |
| T4 | Reverse Swell | Inharmonic cymbal + guitar chord, decayed then reversed | HP 180 Hz, heavy reverb (62 % wet) |
| T5 | High Pluck | Sine core + two inharmonic bell partials | Stereo delay (1/4 L, 1/8 R) + reverb |
| T6 | Kick | Pitch-swept sine (135→46 Hz) + HP click, tanh soft-clip | Overdrive +2 dB, clipper, 62 Hz bell |
| T7 | Snare / Rim | Bandpassed noise crack + F4/1740 Hz tonal body, 115 ms | **Bone dry, dead centre.** HP + 2.6 kHz presence only |
| T8 | Hi-Hat | 6-oscillator metallic stack + noise, HP 7.2 kHz | Phaser (15 % mix), 9 kHz air shelf |
| T9 | Perc / Open Hat | Same stack, longer decay + resonant woodblock | Small room, panned 15 % right |
| T10 | Crash / Impact | 33 Hz sub boom + dark inharmonic crash | HP 24 Hz, very long tail (room 0.97) |
| T11 | 808 Sub | Pure sine driven by a per-sample frequency curve | Distortion +5 dB, +100 Hz bell, **sidechain duck** |
| T12 | Vocal Textures | Formant-synthesized vowel, resampled 2:1 (−12 semitones) | 100 % wet reverb, LP 2 kHz ×2 |
| T14 | Strings | Bowed ensemble: 3 players per part, independent detune/vibrato rate/vibrato depth/attack, bow noise | HP 180 Hz, LP 3.4 kHz, 1.9 kHz notch, big hall, widened |
| T13 | Grand Piano (2-hand) | Physically-modelled Steinway: hammer strike at 1/8 (nulls every 8th partial), 1–3 true unison strings, two-stage decay, register-dependent inharmonicity, velocity-as-timbre. Right hand as before; a left hand joins in the instrumental-only zones (see below) | HP 140 Hz, LP 6.5 kHz, short hall (no chorus) |
| T15 | Solo Violin | Single bowed voice (not the T14 ensemble): written melodic phrase, portamento between notes, per-note re-bow accent, vibrato | HP 260 Hz, LP 9 kHz, slap delay (¾ beat), medium hall |

---

## T13 — the piano stab pattern

Three-note close voicings, one hand, in the **F4–Eb5 register only** — no low
keys held down. Voice-led so common tones hold and the top moves by step:

```
Gm  G4 Bb4 D5   ->  Eb  G4 Bb4 Eb5   ->  Bb  F4 Bb4 D5   ->  F  F4 A4 C5
```

Each chord lasts **2 bars** and is struck **3 times**, spread across the whole
section — bar 1 downbeat, the "and" of 3 in bar 1, then beat 2 of bar 2 —
leaving 3 beats to breathe before the chord switches. Hits are 1.67 s apart with
a 2.0 s gap at the end. An earlier version packed all three into the first four
of eight beats, which read as busy rather than somber; the original is a
midtempo ballad and the piano is there to set tone, not to drive rhythm. The first two hits are choked to 0.95 s so
they read as stabs; the third rings 2.9 s through the empty bar. Over an 8-bar
chorus that is 4 chords × 3 hits = **12 stabs**.

**Damper release.** Notes are not truncated at a fixed length. `steinway_note`
and `electric_note` take a `release` argument that applies an exponential damper
over the tail, because a struck string stops when felt lands on it — fast, but
finite. Rendering a fixed-length note and ending it with a 4 ms fade was cutting
notes off while they were still **15–21 dB from silence**, which reads as an
unnatural click between hits rather than as a note ending. Stabs are now 1.15 s
with a 0.25 s damper (3.2 s / 0.60 s for the ringing third hit), and the electric
guitar got the same treatment (0.80/0.22 arp, 2.4/0.55 strum). All four now end
below −52 dB relative to their own peak instead of −15 to −21 dB.

Verified: **96 of 96** scheduled stabs land with a measurable attack (energy
ratio above 1.5× across the hit; most land at 20-75×). A naive envelope
detector reports far more onsets on this stem — those extra points are
unison-string decay ripple, not notes.

Register check on the mixed stem (right hand only): **−59.1 dBFS below
220 Hz** against −31.7 dBFS in 220 Hz–1 kHz. The left hand below
deliberately reopens some of that headroom, on purpose, only where nothing
else needs it.

## Two-hand piano — the left hand

The right hand above is a deliberately thin sound: three notes, high
register, nothing below F4. That was correct while the piano had to share
space with a rap vocal, but it leaves real weight on the table anywhere the
vocal isn't going to be — the bare solo intro, and the two new instrumental
peaks the extended choruses buy (see "Growing the emotional arc" below). In
those three zones only (`arrange.TWO_HAND_ZONES`), a left hand joins: one
sustained root+5th voicing per 2-bar chord, struck once and left to ring
under the right hand's stabs, the way a real pianist would actually play a
dramatic passage rather than repeating the right hand's rhythm an octave
down.

The register is chosen per bar, not fixed, because the 808 changes what's
safe to sit under:

- **Chorus 1** has no bass at all (T11 is silent there — it's the bare solo
  intro), so the left hand goes all the way down: `G2/D3`, `Eb2/Bb2`,
  `Bb1/F2`, `F2/C3`. This is the only place in the record the piano is
  allowed real bass weight.
- **The two peak zones** have the 808 playing, so the left hand sits an
  octave higher — `G3/D4`, `Eb3/Bb3`, `Bb2/F3`, `F3/C4` — clear of the root
  the 808 owns, still well below the right hand.

This is also why T13's mix chain highpass moved from a double 220 Hz stage
down to a single 140 Hz stage: 220 Hz would have filtered the left hand
right back out, defeating the point of adding it. The right-hand stabs sit
at F4 (349 Hz) and above regardless, so nothing about their sound changes.

## Superseded — the lead synth (Godzilla-informed sound design)

*Kept for reference; T13 is now the piano above.*

**Research.** *Godzilla* (Eminem ft. Juice WRLD, 2020, prod. D.A. Got That Dope)
is **E♭ minor at 166 BPM** (83 half-time), progression **E♭m–G♭**. Its synth is
characterised as a growling, monstrous line that evolves *through filtering and
register shifts*.

**What was borrowed and what was not.** The *patch* is modelled on it —
a detuned saw stack for the growl, a hard filter-swept pluck for the bite, drive
for aggression. The *riff is original*: transcribing a recognisable melodic hook
into a releasable beat is the part that carries copyright exposure, and the line
had to be rewritten for G minor at 90 BPM regardless.

**Implementation.** `synth.lead_pluck()` sums five saws detuned ±18 cents plus a
pulse layer and a sub octave, then runs them through `_sweep_lp()` — a lowpass
whose coefficients are recomputed per 256-sample block while the biquad state
carries across the boundary, so the cutoff glides from 7 kHz down to 620 Hz
without zipper noise.

The two behaviours the research called out are both sequenced in: the filter
opens (620 Hz → 1350 Hz) in the back half of each chorus, and the riff jumps an
octave for the final two bars.

**Placement.** Two voicings, not one. Chorus 1 is the synth *alone* — the
research on the original notes that its first chorus is a solo instrument, used
to set the somber tone before the beat arrives. An aggressive detuned-saw patch
would set entirely the wrong tone there, so `lead_pluck()` is run in a **soft
voicing**: filter mostly shut (3 kHz → 420 Hz), a slower sweep, four voices
instead of five, detune cut to ±11 cents, drive down to 1.25, longer notes and a
sparser 4-note-per-bar figure. The hard voicing returns for choruses 2 and 3.

The synth is kept out of the verses entirely: the riff occupies the same range
the rap needs, and the mix is built around leaving that range empty. It routes
through the harmonic bed bus, so it takes the same 1.6 kHz / 3.2 kHz pocket
carve as everything else melodic.

Its chorus-1 arrangement gain is **3.0** rather than 1.0. Its mix level was set
to sit inside an 11-track chorus, which left the solo intro 16 dB below the rest
of the record; the boost brings it to ~10 dB down on broadband while making it
the *loudest* zone in the 300 Hz–6 kHz band, so it reads as present and
deliberate rather than as a quiet fade-in.

---

## Phase 2 — the 64-bar grid

```
bars  1–8    CHORUS 1        solo piano — nothing else at all
bars  9–16   VERSE 1a        drums + 808 + guitar enter
bars 17–24   VERSE 1b        + pad, FM chord, perc, hats up
bars 25–32   CHORUS 2        full — every track
bars 33–36   CHORUS 2 PEAK   + left-hand piano, denser pluck, quiet violin foreshadow
bars 37–44   VERSE 2a        stripped back hardest (the drop)
bars 45–52   VERSE 2b        rebuild, bigger than verse 1b
bars 53–60   CHORUS 3        full again, hotter than chorus 2
bars 61–64   CHORUS 3 PEAK   full violin solo + two-hand piano — the record's peak
```

Choruses 2 and 3 were extended from 8 to 12 bars each. The extra 4 bars are not
more of the same loop — they're a new instrumental-only "peak" zone
(`chorus2_peak`, `chorus3_peak`), free of the vocal-pocket constraint every
other zone respects, used to push two-hand piano and a solo violin in without
crowding anywhere a vocal actually needs the space. See "Growing the emotional
arc" below.

Track presence is a table (`arrange.LAYERS`), not scattered conditionals: each
of the nine zones maps every track to a gain, where 0.0 means silent. Measured
result — the 300 Hz–6 kHz column is where the arrangement actually moves, since
broadband RMS just tracks the 808:

```
zone           tracks   broadband   300 Hz-6 kHz
chorus1           1      -22.1        -22.9      solo piano, no low end at all
verse1a           7      -15.6        -28.2
verse1b          11      -15.5        -26.9
chorus2          12      -14.7        -25.5
chorus2_peak     13      -14.7        -24.5      left hand + denser pluck + quiet violin
verse2a           7      -15.6        -29.9      the drop
verse2b          11      -15.5        -27.0
chorus3          12      -14.7        -25.2      hotter than chorus 2 by design
chorus3_peak     13      -14.4        -20.3      the record's loudest, densest moment
```

Chord loop, **two bars each**: **Gm → Eb → Bb → F** (i–VI–III–VII), so the full
progression spans 8 bars. Harmonic rhythm is global (`arrange.chord_index`) —
the piano brief is written around 2-bar chord sections, and if only the piano
slowed down it would sit on Gm while the guitar and 808 had moved to Eb. The
8-bar-aligned section starts (bars 1, 9, 25, 37, 53) land on Gm; the 4-bar peak
extensions (33, 61) do not, since they add half an 8-bar loop rather than a
whole one. Nothing downstream assumes zone boundaries sit on the tonic, so this
is a cosmetic consequence of the extension, not a bug.

- **T4** crests exactly on each chorus downbeat. Chorus 1 sits at bar 1, so its
  lead-in falls off the front of the grid — a double-length swell placed two bars
  early leaves precisely the final bar audible, still peaking on the downbeat.
  The one other big texture change in the record, chorus 1 into verse 1, gets
  its own transition instead (see "A modern transition" below).
- **T5** plays D5→Bb4→G4 on the "and of 3 / 4 / and of 4" of alternating bars in
  the two full choruses, threading between the kick and snare rather than over
  them. In the two peak zones it drops the alternating-bar rule and plays every
  bar — a denser shimmer that's part of what makes the peaks read as bigger.
- **T6** hits beat 1 and the "and" of 2, plus beat 4 in choruses (and the "and"
  of 4 every fourth bar). 138 triggers — chorus 1 has no drums at all, and the
  very last bar drops the kick entirely (see "The ending" below).
- **T7** lands on beat 3 of every bar that has drums — 55 of them.
- **T8** runs steady 1/8ths with 1/32 rolls filling beat 4 of every 2nd and 4th
  bar, at reduced velocity in the stripped verses. 588 hits.
- **T11** follows the roots with a 60 ms portamento between overlapping notes,
  and drops 2 dB through the verses.

### 808 octave placement

The roots are octave-placed to **G1 (49 Hz) → Eb2 (78 Hz) → Bb1 (58 Hz) → F2
(87 Hz)** rather than descending literally. A strict descent would put Eb1 at
38.9 Hz, below what a phone or laptop reproduces at all, and the octave jumps
give the portamento something audible to slide across — which is the point of a
drill bassline. Root motion is unchanged.

---

## A modern transition — chorus 1 into verse 1

Every other big texture change in the record (verse into chorus) already had a
transition: `T4`'s reverse swell, cresting on the downbeat. The one gap was the
very first one, chorus 1's bare piano dropping into the full band at bar 9 — and
a reverse swell is the wrong tool there, because it announces a *chorus*
arrival specifically, not a texture change in general.

`synth.transition_riser()` is the genre-standard fix: a white-noise sweep whose
highpass cutoff rises exponentially through the duration, layered under a synth
"lift" gliding up about two octaves (90 → 380 Hz), both crescendoing into a
bright noise snap at the moment verse 1 lands. It runs 1.5 bars into bar 9 and
is mixed into the T4 bus, so it inherits the same big hall reverb the swells
use without needing a new track.

## Growing the emotional arc

The brief was to make the arrangement's emotional curve climb continuously
rather than plateau after chorus 2 — every section a little bigger than its
counterpart before it, the final chorus the maximum. Four things do this
together, and the "ARRANGEMENT DYNAMICS" table above is the measured proof it
worked (chorus2_peak > chorus2, chorus3 > chorus2, chorus3_peak is the loudest
zone in the record by nearly 5 dB in the 300 Hz–6 kHz band):

1. **Choruses 2 and 3 grew**, chorus 3 gained a further step over chorus 2
   (`T2`, `T3`, `T13`, `T14` all tick up a little further at `chorus3` than
   `chorus2` in `arrange.LAYERS`), and the two new peak zones step up again on
   top of that — a staircase, not a single jump.
2. **Two-hand piano** (above) adds real low-register weight exactly where
   there's room for it — chorus 1's bare intro and the two peak zones — rather
   than everywhere, which would just be louder, not bigger.
3. **A solo violin** (`T15`) plants a single quiet motif in `chorus2_peak`
   (one held note, `arrange.VIOLIN_FORESHADOW`) and answers it in full in
   `chorus3_peak` (`arrange.VIOLIN_FINALE`, an 8-note phrase). `strings_chord`
   is an ensemble — many detuned bowed voices smeared into a pad — which is
   the wrong instrument for a melody that needs to read as one performer, so
   the violin is a separate synthesis (`synth.solo_violin_phrase`): a single
   voice, notes tied by portamento the way a fingered string glides between
   positions, a soft re-bow accent on each new note rather than a clean synth
   retrigger. The phrase's register climbs across its 4 bars — D5 up to D6, the
   highest note anywhere in the record — so the melodic shape and the
   arrangement's dynamics peak at the same instant.
4. **The ending** (next) removes everything except what's ringing, so the
   final chord actually gets heard instead of being buried under a full kit
   still going at full velocity.

## The ending

The last bar (`arrange.FINAL_BAR`) breaks from the loop on purpose: `T6`/`T7`/
`T8`/`T9` are all silenced — no kick, no snare, no hats, no perc — while `T11`
holds one long root note instead of retriggering, and the piano/strings/violin
notes already sounding (all of them longer than a bar) simply ring out through
the tail. This is the standard way a produced record ends a climax: pull the
rhythm section out from under the pitched instruments so the last chord is
actually audible as an ending, not just where the loop happens to stop.

## MIDI export for Logic Pro

`midi_export.py` writes one standard MIDI file per melodic/harmonic instrument
to `midi/`: `piano_RH`, `piano_LH`, `guitar`, `pad`, `strings`, `violin`,
`bass808`. It calls `arrange.schedule_events()`, a symbolic pass that mirrors
`build()`'s exact scheduling math (same `lg`/`chord_index`/`bar_time` calls)
but never touches `synth.py`, so it runs in under a second instead of the
couple of minutes a full audio render takes — it only needs to know *when* and
*what*, not *how it sounds*.

Note durations are musical (how long a key would be held), not the audio
layer's decay length: a struck piano note keeps ringing after the MIDI
note-off, same as a real instrument, so short note-offs are correct even
though the rendered audio note is much longer. Drums aren't exported — they
already sound the way they're meant to, and re-tracking a groove this specific
in a DAW would just be re-deriving `arrange.py` by ear.

Workflow: import each `.mid` into its own Logic track, assign a real
instrument or a better VST than this project's own synthesis, bounce, and send
the rendered audio back — `mixdsp.py`'s mix/master stage is happy to take real
stems in place of (or alongside) the synthesized ones.

---

## Phase 3 — sidechain

`mixdsp.sidechain_env` builds the ducking curve mathematically from the kick
trigger list: −5 dB floor, 2 ms linear attack, then a power-curve recovery
reaching unity at exactly 80 ms. Overlapping ducks combine by minimum, so a fast
kick pattern never lets the 808 fully recover before the next hit. A power curve
rather than an exponential holds the sub down *through* the kick transient
instead of smearing the recovery past it.

Verified by `analyze.py`:

```
envelope depth       -5.00 dB   (spec -5.0)
attack to floor       2.00 ms   (spec 2.0)
release to -0.5 dB   73.8 ms
measured on stem     -3.96 dB median  (envelope predicts -4.48 dB)
```

The stem probe only samples kicks that land *between* 808 note onsets — a kick
on a note onset has nothing sustaining to duck and would read as a level rise.

---

## Phase 4 — master and measured results

Master bus: HP 20 Hz ×2 → LP 19 kHz ×2 → Compressor (1.5:1, 30 ms / 100 ms) →
Limiter → tanh soft clip → 1.2 s tail fade → exact peak normalization.

```
peak      -6.000 dBFS      (target -6.0000)
RMS      -15.52 dBFS
LUFS-I   -14.98            (~-9.0 LUFS if normalized to 0 dBFS)
crest      9.52 dB         transients intact
L/R corr  +0.954           mono-safe
DC offset +2.4e-05
clipped samples: 0         NaN/Inf: none
```

**Octave tilt** — the 1–5 kHz vocal pocket is the quietest part of the spectrum
by design:

```
    20-60   Hz   -4.70
    60-120  Hz   -2.90   <- 808 lives here
   120-250  Hz  -14.96
   250-500  Hz  -13.31
   500-1000 Hz  -12.72
  1000-2000 Hz  -18.04   <- vocal pocket
  2000-4000 Hz  -23.65   <- vocal pocket
  4000-8000 Hz  -29.39
  8000-16000Hz  -33.32
```

**Section lift** — see the zone table under Phase 2. Chorus 3's peak zone is
now the loudest section in the record, by design — see "Growing the emotional
arc".

**Groove grid** — every scheduled hit is checked for a real energy jump at its
sample-exact index (edge-counting on layered drums is unreliable — see "2026
standards pass" below for why this moved from jitter-measurement to
schedule verification):

```
kick   138/138 scheduled hits confirmed
snare   55/55  scheduled hits confirmed
hat    588/588 scheduled hits confirmed
```

Expected counts are derived from `arrange.LAYERS` rather than hard-coded, so the
check stays honest when the arrangement changes — it picked up all 22 extra
kicks, 7 extra snares and 60 extra hats from the two 4-bar extensions
automatically.

**Portamento**, measured as instantaneous frequency across a chord change:

```
bar 34  G1->Eb2   before 48.4 Hz | t=0 48.1 | +30ms 45.7 | +60ms 44.1 | +120ms 42.1
bar 35  Eb2->Bb1  before 48.3 Hz | t=0 50.0 | +30ms 62.7 | +60ms 76.0 | +120ms 77.0
bar 36  Bb1->F2   before 77.8 Hz | t=0 85.1 | +30ms 78.6 | +60ms 79.1 | +120ms 77.6
```

---

## Files

```
synth.py       DSP primitives + the 15 asset generators
arrange.py     64-bar sequencer, chord/root tables, sidechain trigger capture,
               schedule_events() for MIDI export
mixdsp.py      per-track Pedalboard chains, sidechain, bus summing, master
build.py       end-to-end render -> LoveTheWayYouLie_Modern12Track.wav + stems/
analyze.py     automated QA: levels, tilt, dynamics, grid, glide -> analysis.png
midi_export.py writes midi/*.mid (piano RH/LH, guitar, pad, strings, violin, 808)
```


---

## Pitch verification

Every track was verified by three independent agents writing their own analysis
code, plus a symbolic pass over the note tables. Findings:

| Track | Verdict | Max abs deviation |
| --- | --- | --- |
| T1 Electric Guitar | PASS | 7.25 cents (Karplus–Strong delay quantization) |
| T2 Ambient Synth | PASS | 1.78 cents |
| T3 Synth Pad | PASS | 0.65 cents (detune stack exactly ±7 c by design) |
| T5 High Pluck | PASS | **0.00 cents** |
| T11 808 Sub | PASS | **0.0022 cents**, glide 59.92 ms vs 60 ms spec |
| T12 Vocal Textures | PASS | 0.12 cents; octave drop exact to +0.05 cents |
| T13 Grand Piano | PASS | 2.96 cents (unison detune) |
| T6/T7/T9/T10 drums | PASS | within spec; kick verified not to beat against the 808 |

T14 (strings) and T15 (solo violin) were added after this multi-agent pass and
haven't been run through it — both use `nf()` for every pitch like everything
else here, so there's no reason to expect a different result, but that's an
expectation, not a verified claim, until someone actually runs it.

Symbolic check: 90 sequenced notes, **zero out of key**; every voicing spells its
triad; chord progression correct in **112 of 112** bars.

**Two real defects found and fixed:**

1. **Snare tuned out of key.** Its tonal body sat at 331 Hz — E natural, the only
   pitch in the kit foreign to G minor. At 13 ms it reads as a transient, but it
   measured genuinely tonal (Q≈25). Retuned to F4, the ♭7. Now F4 +0.87 cents.
2. **Raw stems exported as PCM_24.** Raw buffers legitimately exceed 1.0 (T13
   peaks at 2.83) because per-track gains are applied downstream, so fixed-point
   export clamped them and reported clipping that does not exist in the mix.
   Now written as float. Processed stems peak at −6.08 dBFS and the master has
   **zero** clipped samples.

Two agent observations that were *not* defects, worth recording so they are not
"re-fixed" later:

- **Piano partials read sharp** if measured by dividing a high harmonic by its
  index — partial 4 sits +10.45 cents above 4×f0. That is modelled string
  inharmonicity, and it is correct. Verify pianos at the fundamental only.
- **T12 carries ±10–17 cents of vibrato** at 2.7–10 Hz. Deliberate. A per-window
  ±15 cent gate will produce false failures on that track; its amplitude-weighted
  centre is exact to 0.4 cents.


---

## 2026 standards pass

Researched against current delivery specs and production practice, then closed
the gaps that were real.

**Confirmed first:** Gm–E♭–B♭–F *is* the original's progression, and the record
sits in B♭ major / G minor relative — so the harmony was already right. Alex da
Kid's method (loops and electronic drums first, live instrumentation second)
matches this build's order.

### What was fixed

**1. Drum layering.** Single-layer synthesized drums were the biggest gap.

- **Kick** → three layers: a 42 Hz sub for weight, a faster 105→255 Hz punch
  band for body on small speakers, and a separate transient top that survives
  limiting. Each has its own pitch envelope and decay.
- **Snare** → body + clap + air. The clap is four transients staggered over
  ~25 ms, which is what separates a clap from a single noise burst.
- **808** → two parallel layers: a clean sub carrying weight, and a hard-driven
  (+13 dB) mid layer for the harmonic ladder that makes a 49 Hz root audible on
  a phone. One band cannot do both jobs.
- **Hats** → three timbres at different pitches, rotated through the pattern.

**2. Micro-timing.** The track was 100 % quantised, which is a large part of
what makes programmed drums read as programmed. Offsets are now applied per hit,
deterministically per (track, bar, position): kick ±2.5 ms, snare ±2.0 ms, hats
±4.5 ms, perc ±5.5 ms, piano ±4.0 ms. Swing of 0.56 pushes offbeat 8ths 40 ms
late. Velocity is jittered per hit as well.

**3. Hi-hat programming.** Straight swung 8ths with accent patterning, plus a
1/32 roll on beat 4 of every 2nd and 4th bar *and* a 1/8-triplet fill closing
each 8-bar zone.

**4. Strings.** The layer the original has and this lacked. Bowed ensemble with
three players per part, each with independent detune, vibrato rate, vibrato
depth and attack time. Notched at 1.9 kHz before the bed bus, since strings are
the classic thing that fights a vocal.

### Also fixed along the way

Seeds were derived from Python's `hash()`, which is salted per process — so the
render was **not reproducible between runs**. Replaced with FNV-1a.

### Measured result

```
true peak        -5.84 dBTP          spec <= -1.0        PASS
integrated       -14.98 LUFS  (-9.98 normalised to -1 dBTP)
max short-term   -12.44 LUFS  (-7.44 normalised)         in the -7..-9 window
crest              9.52 dB
L/R correlation   +0.954                                 mono-safe
clipped samples    0
```

Integrated sits under the genre norm because the bare solo intro drags the
average down — a deliberate arrangement choice. Short-term through the peak
zones now sits at the loud edge of the -7..-9 window rather than comfortably
inside it, which is the point: chorus3_peak is meant to be the loudest moment
in the record. The −6 dBFS ceiling is intentional headroom for vocal tracking,
not a finished master; mastered to −1 dBTP with a vocal, this still lands in
spec.

Groove verification is schedule-based rather than edge-counting (a two-hump
layered kick envelope reads as two onsets to a naive detector):
**138/138 kick, 55/55 snare, 588/588 hat** scheduled hits confirmed — up from
116/48/516 before the two choruses were extended, and the check picked up the
new counts automatically because they're derived from `arrange.LAYERS`, not
hard-coded.


---

## Leaving room for the rap

Verses have to be measurably more open than choruses, or a vocal has nowhere to
sit. Measured as *the percentage of 50 ms windows where the mastered mix's
1-5 kHz band is above a fixed -38 dBFS gate*, over all nine zones:

```
chorus1         90.7%
verse1a         55.8%
verse1b         71.0%
chorus2         89.3%
chorus2_peak    91.7%
verse2a         37.5%   the drop
verse2b         67.6%
chorus3         92.0%
chorus3_peak   100.0%   instrumental-only — see below

verses avg (4)  58.0%
choruses avg (3, core zones only)  90.7%
gap             32.7 points
```

The verse/chorus split that mattered before the extension is still intact:
verse2a (the drop) is the most open moment in the record at 37.5%, and every
verse sits well below every chorus. The two peak zones sit *above* even the
main choruses, and that's intentional rather than a regression — they're
zones a rap vocal was never routed through in the first place (see
"Growing the emotional arc"), so there's nothing to protect there. Two
earlier changes are what keep the verse/chorus gap this wide in the first
place: the guitar arpeggio drops from four notes per bar to two under the
verses (at 8 events/bar it was the densest melodic source), and the piano
sits 3–5 dB further back there.

**Known cosmetic detail:** all 96 piano stabs clear the 1.5× attack-detection
threshold, but two are far weaker than the rest — the downbeats of bars 9 and
37 (1.7-1.8× against a typical 20-75× everywhere else). Both are zone
boundaries (chorus1 → verse1a, chorus2_peak → verse2a) where the piano level
drops sharply into a verse while the previous zone's note is still ringing, so
the new stab is partly masked by its own predecessor's tail. The note is
sequenced and written; it is simply quieter than what is already decaying. The
full band enters at both points, so it is inaudible either way.
