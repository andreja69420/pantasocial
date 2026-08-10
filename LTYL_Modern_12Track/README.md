# Love The Way You Lie — Modern Rework (13 tracks)

A fully synthesized, mixed and mastered 56-bar dark trap/drill instrumental.
Started as 12 tracks; T13 (lead synth) was added later. Output filenames keep
the original `12Track` name so existing links stay valid.
90 BPM, G minor, 48 kHz / 24-bit stereo, peaks at exactly **−6.00 dBFS** so
there is headroom to record raw vocals on top.

```
cd LTYL_Modern_12Track
python3 -m venv venv && ./venv/bin/pip install pydub numpy scipy pedalboard requests librosa
./venv/bin/python build.py      # renders LoveTheWayYouLie_Modern12Track.wav + stems/
./venv/bin/python analyze.py    # QA pass + analysis.png
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

## The 12 tracks

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
| T13 | Grand Piano | Physically-modelled Steinway: hammer strike at 1/8 (nulls every 8th partial), 1–3 true unison strings, two-stage decay, register-dependent inharmonicity, velocity-as-timbre | HP 220 Hz ×2, LP 6.5 kHz, short hall (no chorus) |

---

## T13 — the piano stab pattern

Three-note close voicings, one hand, in the **F4–Eb5 register only** — no low
keys held down. Voice-led so common tones hold and the top moves by step:

```
Gm  G4 Bb4 D5   ->  Eb  G4 Bb4 Eb5   ->  Bb  F4 Bb4 D5   ->  F  F4 A4 C5
```

Each chord lasts **2 bars** and is struck **3 times** — on the downbeat, the
"and" of 2, and beat 4 — after which the whole second bar is left empty to
breathe before the chord switches. The first two hits are choked to 0.95 s so
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

Verified: **84 of 84** scheduled stabs land with a clear attack (energy ratio
2.15–4.15× across the hit). A naive envelope detector reports ~180 onsets on
this stem — those extra points measure at most 1.18×, i.e. unison-string decay
ripple, not notes.

Register check on the mixed stem: **−59.1 dBFS below 220 Hz** against
−31.7 dBFS in 220 Hz–1 kHz. Nothing muddies the 808.

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

## Phase 2 — the 56-bar grid

```
bars  1–8    CHORUS 1   solo lead synth — nothing else at all
bars  9–16   VERSE 1a   drums + 808 + guitar enter
bars 17–24   VERSE 1b   + pad, FM chord, perc, hats up
bars 25–32   CHORUS 2   full — every track
bars 33–40   VERSE 2a   stripped back hardest
bars 41–48   VERSE 2b   rebuild
bars 49–56   CHORUS 3   full again
```

Track presence is a table (`arrange.LAYERS`), not scattered conditionals: each
of the seven zones maps every track to a gain, where 0.0 means silent. Measured
result — the 300 Hz–6 kHz column is where the arrangement actually moves, since
broadband RMS just tracks the 808:

```
zone       tracks   broadband   300 Hz-6 kHz
chorus1       1      -25.4        -26.2      solo synth, no low end at all
verse1a       6      -16.4        -30.5
verse1b       9      -16.3        -29.2
chorus2      11      -15.3        -27.6
verse2a       6      -16.5        -32.5      the drop
verse2b       9      -16.3        -29.3
chorus3      11      -15.3        -27.6
```

Chord loop, **two bars each**: **Gm → Eb → Bb → F** (i–VI–III–VII), so the full
progression spans 8 bars. Harmonic rhythm is global (`arrange.chord_index`) —
the piano brief is written around 2-bar chord sections, and if only the piano
slowed down it would sit on Gm while the guitar and 808 had moved to Eb. Every
section boundary (bars 1, 9, 25, 33, 49) lands on Gm.

- **T4** crests exactly on each chorus downbeat. Chorus 1 sits at bar 1, so its
  lead-in falls off the front of the grid — a double-length swell placed two bars
  early leaves precisely the final bar audible, still peaking on the downbeat.
- **T5** plays D5→Bb4→G4 on the "and of 3 / 4 / and of 4" of alternating bars in
  the two full choruses, threading between the kick and snare rather than over
  them.
- **T6** hits beat 1 and the "and" of 2, plus beat 4 in choruses (and the "and"
  of 4 every fourth bar). 116 triggers — chorus 1 has no drums at all.
- **T7** lands on beat 3 of all 48 bars that have drums.
- **T8** runs steady 1/8ths with 1/32 rolls filling beat 4 of every 2nd and 4th
  bar, at reduced velocity in the stripped verses. 528 hits.
- **T11** follows the roots with a 60 ms portamento between overlapping notes,
  and drops 2 dB through the verses.

### 808 octave placement

The roots are octave-placed to **G1 (49 Hz) → Eb2 (78 Hz) → Bb1 (58 Hz) → F2
(87 Hz)** rather than descending literally. A strict descent would put Eb1 at
38.9 Hz, below what a phone or laptop reproduces at all, and the octave jumps
give the portamento something audible to slide across — which is the point of a
drill bassline. Root motion is unchanged.

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
measured on stem     -4.59 dB median  (envelope predicts -4.48 dB)
```

The stem probe only samples kicks that land *between* 808 note onsets — a kick
on a note onset has nothing sustaining to duck and would read as a level rise.

---

## Phase 4 — master and measured results

Master bus: HP 20 Hz ×2 → LP 19 kHz ×2 → Compressor (1.5:1, 30 ms / 100 ms) →
Limiter → tanh soft clip → 1.2 s tail fade → exact peak normalization.

```
peak      -6.000 dBFS      (target -6.0000)
RMS      -16.58 dBFS
LUFS-I   -15.84            (~-9.8 LUFS if normalized to 0 dBFS)
crest     10.58 dB         transients intact
L/R corr  +0.951           mono-safe
clipped samples: 0         NaN/Inf: none
```

**Octave tilt** — the 1–5 kHz vocal pocket is the quietest part of the spectrum
by design:

```
   20-60   Hz   -5.05
   60-120  Hz   -2.30   <- 808 lives here
  120-250  Hz  -15.05
  250-500  Hz  -15.26
  500-1000 Hz  -15.43
 1000-2000 Hz  -19.74   <- vocal pocket
 2000-4000 Hz  -25.50   <- vocal pocket
 4000-8000 Hz  -28.28
 8000-16000Hz  -33.96
```

**Section lift** — see the zone table under Phase 2.

**Groove grid** — every hit is placed at a sample-exact index; onset detection
confirms the counts and sub-millisecond jitter:

```
kick   116 onsets (expect 116)   jitter max 0.81 ms
snare   48 onsets (expect  48)   jitter max 0.00 ms
hat    528 onsets (expect 528)   jitter max 1.08 ms
```

Expected counts are derived from `arrange.LAYERS` rather than hard-coded, so the
check stays honest when the arrangement changes.

**Portamento**, measured as instantaneous frequency across a chord change:

```
bar 34  G1->Eb2   before 49.3 Hz | +30ms 59.3 | +60ms 75.6 | +120ms 77.8
bar 35  Eb2->Bb1  before 77.9 Hz | +30ms 69.5 | +60ms 58.0 | +120ms 56.7
```

---

## Files

```
synth.py     DSP primitives + the 12 asset generators
arrange.py   56-bar sequencer, chord/root tables, sidechain trigger capture
mixdsp.py    per-track Pedalboard chains, sidechain, bus summing, master
build.py     end-to-end render -> LoveTheWayYouLie_Modern12Track.wav + stems/
analyze.py   automated QA: levels, tilt, dynamics, grid, glide -> analysis.png
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
true peak       -5.88 dBTP           spec <= -1.0        PASS
integrated      -15.10 LUFS  (-10.22 normalised to -1 dBTP)
max short-term  -13.37 LUFS  ( -8.48 normalised)         in the -7..-9 window
crest            9.71 dB
L/R correlation +0.960                                   mono-safe
clipped samples  0
```

Integrated sits ~1 LU under the genre norm because the bare solo intro drags the
average down — a deliberate arrangement choice. Short-term through the choruses
is in range. The −6 dBFS ceiling is intentional headroom for vocal tracking, not
a finished master; mastered to −1 dBTP with a vocal, this lands in spec.

Groove verification is now schedule-based rather than edge-counting (a two-hump
layered kick envelope reads as two onsets to a naive detector):
**116/116 kick, 48/48 snare, 516/516 hat** scheduled hits confirmed.
