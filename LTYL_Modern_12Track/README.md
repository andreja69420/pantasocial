# Love The Way You Lie — Modern Rework (13 tracks)

A fully synthesized, mixed and mastered 56-bar dark trap/drill instrumental.
Started as 12 tracks; T13 (lead synth) was added later. Output filenames keep
the original `12Track` name so existing links stay valid.
104 BPM, G minor, 48 kHz / 24-bit stereo, peaks at exactly **−6.00 dBFS** so
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
i–VI–III–VII loop, the acoustic guitar as the melodic anchor, the huge
ambient piano — and replace the delivery mechanism entirely:

| Original | This rework |
| --- | --- |
| ~84 BPM stadium-rock kit | 104 BPM trap grid, dry rimshot on beat 3 |
| Big reverb on everything | Reverb on the *beds only*; snare bone dry |
| Live bass / low piano | Distorted sine 808 with 60 ms glides |
| Guitar full-range | Guitar lowpassed at 2.5 kHz with vinyl wobble |
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
| T1 | Main Guitar | Extended Karplus–Strong with filtered pick burst + body resonance | LP 2.5 kHz ×2, Chorus (vinyl wobble), light room |
| T2 | Ambient Piano | Additive, stiff-string inharmonicity, per-partial decay, hammer noise | HP 150 Hz ×2, large hall (room 0.92, 40 % wet) |
| T3 | Synth Pad | 3× detuned additive saw stack, slow filter bloom | LP 3 kHz ×2, per-channel chorus + M/S widening |
| T4 | Reverse Swell | Inharmonic cymbal + guitar chord, decayed then reversed | HP 180 Hz, heavy reverb (62 % wet) |
| T5 | High Pluck | Sine core + two inharmonic bell partials | Stereo delay (1/4 L, 1/8 R) + reverb |
| T6 | Kick | Pitch-swept sine (135→46 Hz) + HP click, tanh soft-clip | Overdrive +2 dB, clipper, 62 Hz bell |
| T7 | Snare / Rim | Bandpassed noise crack + 331/1740 Hz tonal body, 115 ms | **Bone dry, dead centre.** HP + 2.6 kHz presence only |
| T8 | Hi-Hat | 6-oscillator metallic stack + noise, HP 7.2 kHz | Phaser (15 % mix), 9 kHz air shelf |
| T9 | Perc / Open Hat | Same stack, longer decay + resonant woodblock | Small room, panned 15 % right |
| T10 | Crash / Impact | 33 Hz sub boom + dark inharmonic crash | HP 24 Hz, very long tail (room 0.97) |
| T11 | 808 Sub | Pure sine driven by a per-sample frequency curve | Distortion +5 dB, +100 Hz bell, **sidechain duck** |
| T12 | Vocal Textures | Formant-synthesized vowel, resampled 2:1 (−12 semitones) | 100 % wet reverb, LP 2 kHz ×2 |
| T13 | Lead Synth | 5× detuned saw stack + pulse + sub octave through a block-stepped filter sweep | Drive +3 dB, 350 Hz dip, LP 3.8 kHz, per-channel decorrelation |

---

## T13 — the lead synth (Godzilla-informed sound design)

**Research.** *Godzilla* (Eminem ft. Juice WRLD, 2020, prod. D.A. Got That Dope)
is **E♭ minor at 166 BPM** (83 half-time), progression **E♭m–G♭**. Its synth is
characterised as a growling, monstrous line that evolves *through filtering and
register shifts*.

**What was borrowed and what was not.** The *patch* is modelled on it —
a detuned saw stack for the growl, a hard filter-swept pluck for the bite, drive
for aggression. The *riff is original*: transcribing a recognisable melodic hook
into a releasable beat is the part that carries copyright exposure, and the line
had to be rewritten for G minor at 104 BPM regardless.

**Implementation.** `synth.lead_pluck()` sums five saws detuned ±18 cents plus a
pulse layer and a sub octave, then runs them through `_sweep_lp()` — a lowpass
whose coefficients are recomputed per 256-sample block while the biquad state
carries across the boundary, so the cutoff glides from 7 kHz down to 620 Hz
without zipper noise.

The two behaviours the research called out are both sequenced in: the filter
opens (620 Hz → 1350 Hz) in the back half of each chorus, and the riff jumps an
octave for the final two bars.

**Placement.** Choruses only, plus a rising 2-beat pickup into choruses B and C.
It is kept out of the verses on purpose — the riff occupies the same range the
rap needs, and the entire mix is built around leaving that range empty. It also
routes through the harmonic bed bus, so it takes the same 1.6 kHz / 3.2 kHz
pocket carve as everything else melodic.

Measured effect: chorus-to-verse contrast in the musical band went from
**2.1 dB to 2.5 dB**, and the 1–5 kHz pocket stayed the quietest region of the
spectrum.

---

## Phase 2 — the 56-bar grid

```
bars  1–8    CHORUS
bars  9–24   VERSE   (16)
bars 25–32   CHORUS
bars 33–48   VERSE   (16)
bars 49–56   CHORUS
```

Chord loop, one bar each: **Gm → Eb → Bb → F** (i–VI–III–VII).

- **T4** crests exactly on each chorus downbeat. Chorus 1 sits at bar 1, so its
  lead-in falls off the front of the grid — a double-length swell placed two bars
  early leaves precisely the final bar audible, still peaking on the downbeat.
- **T5** plays D5→Bb4→G4 on the "and of 3 / 4 / and of 4" of alternating chorus
  bars, threading between the kick and snare rather than over them.
- **T6** hits beat 1 and the "and" of 2, plus beat 4 in choruses (and the "and"
  of 4 every fourth bar). 142 triggers total.
- **T7** lands on beat 3 of all 56 bars.
- **T8** runs steady 1/8ths with 1/32 rolls filling beat 4 of every 2nd and 4th
  bar. 616 hits.
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
RMS      -15.84 dBFS
LUFS-I   -15.28            (~-9.3 LUFS if normalized to 0 dBFS)
crest      9.84 dB         transients intact
L/R corr  +0.919           mono-safe
clipped samples: 0         NaN/Inf: none
```

**Octave tilt** — the 1–5 kHz vocal pocket is the quietest part of the spectrum
by design:

```
   20-60   Hz   -5.17
   60-120  Hz   -2.30   <- 808 lives here
  120-250  Hz  -13.40
  250-500  Hz  -14.63
  500-1000 Hz  -16.72
 1000-2000 Hz  -21.46   <- vocal pocket
 2000-4000 Hz  -25.21   <- vocal pocket
 4000-8000 Hz  -27.86
 8000-16000Hz  -32.43
```

**Section lift** — broadband RMS barely moves (the 808 dominates it), so the
verse/chorus contrast is measured in the musical band, where it is ~2.1 dB:

```
chorus  300 Hz-6 kHz  -27.4 dBFS
verse   300 Hz-6 kHz  -29.5 dBFS
```

**Groove grid** — every hit is placed at a sample-exact index; onset detection
confirms the counts and sub-millisecond jitter:

```
kick   142 onsets (expect 142)
snare   56 onsets (expect  56)
hat    616 onsets (expect 616)
```

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
