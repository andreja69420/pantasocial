# Volis me takvu — instrumental

Instrumental bed (matrica) for the song "Volis me takvu" — **G minor, 4/4,
87 BPM, 100 bars, 4:35.9**. No vocals: the arrangement is written to leave room
for them.

Every sound comes from a recording of a real instrument. No AI music
generation, no synthesized waveforms, no sine tones, no General MIDI sound
sets. Each part is sequenced programmatically, humanized, and rendered through
a freely-licensed multi-sampled library with `sfizz_render`.

Key, tempo, section lengths and the harmonic shape were all measured off a
reference mix the artist supplied (see *Matched to the reference* below) rather
than chosen.

## Files

| Path | What |
|---|---|
| `volis-me-takvu-matrica-128.mp3` | The audio matrica — 128 kbps, 48 kHz, stereo |
| `volis-me-takvu-MIDI.zip` | MIDI bundle for a DAW: combined file, per-part files, notes |
| `volis-me-takvu-LOGIC.mid` | Type 1 MIDI, 11 named tracks, tempo and key embedded |
| `matrica-struktura.txt` | Section map with timecodes |
| `midi/` | 12 humanized MIDI parts, one per instrument, plus the melody guide |
| `mix.wav` | 24-bit/48 kHz master (not committed; regenerate with `tools/track/mix.py`) |
| `stems/` | 11 stems, 24-bit/48 kHz WAV (not committed; regenerate with `tools/track/render.py`) |
| `everything-i-never-said*.mp3` | First, unrelated through-composed piece, kept for reference |

The WAV stems (~460 MB) are deliberately not committed. `tools/track/`
regenerates them — the composition is seeded, so output is deterministic.

## Reproducing

```
python3 tools/track/remap_upright.py   # one-off: fix the piano library mapping
python3 tools/track/compose.py         # write humanized MIDI parts
python3 tools/track/render.py          # render each part through its library
python3 tools/track/mix.py             # mix, master, limit to -14 LUFS
python3 tools/track/verify.py          # duration/clipping/tempo/key/section checks
python3 tools/track/export_logic.py    # combined multi-track MIDI for a DAW
```

Requires `sfizz_render` on PATH plus `mido`, `numpy`, `scipy`, `soundfile`,
`librosa`, `pedalboard`, `pyloudnorm`, and the sample libraries below under
`/home/user/samples`.

## Matched to the reference

Two reference mixes were supplied — one instrumental, one with a guide vocal.
Everything structural was measured, not guessed:

- **87 BPM.** Taken from the interval between identical sections (77.23 s over
  28 bars → 87.01 BPM), which is far more reliable than beat tracking — that
  reported 86.13.
- **G minor.** Chroma ranks B♭, F, G, E♭, i.e. **Gm – E♭ – B♭ – F**. That is the
  same shape as the original Em – C – G – D sketch, so the arrangement is
  written in E minor and transposed +3 semitones at note-emit time.
- **16-bar verses, 12-bar hooks.** Measured at 43.9 s and 33.11 s. Twelve bars
  fits four sung lines at two bars each, then the tag twice.
- **Solo piano opening.** The reference has no drums until 27.4 s. The vocal
  enters at 1.30 s, the piano answers 1.1 s later on a downbeat.
- **The piano enters on the tonic.** Its first bar reads G/D/B♭, then E♭, B♭, F.
  Bar 0 is left empty for the vocal pickup, so the chord loop starts at bar 1 —
  otherwise the first audible chord lands on E♭ and the track opens in the
  relative major.

## Arrangement

Chord loop Gm – E♭ – B♭ – F, one bar per chord, looping throughout. Every
section begins on the tonic; the last two bars hold it.

| Section | Bars | Time | Arrangement |
|---|---|---|---|
| HOOK 1 | 0–8 | 0:00–0:25 | Solo piano. Bar 0 empty, piano from bar 1 |
| VERSE 1 | 9–24 | 0:25–1:09 | Rhythm section only — sparse, heavy, space for the rap |
| HOOK 2 | 25–36 | 1:09–1:42 | Full kit, bass, acoustic, strings |
| VERSE 2 | 37–52 | 1:42–2:26 | Clean electric arpeggios and cellos come in |
| HOOK 3 | 53–64 | 2:26–2:59 | Double-tracked power chords, full strings |
| VERSE 3 | 65–80 | 2:59–3:43 | Fullest verse — violins added |
| BREAK | 81–84 | 3:43–3:54 | Drums out, cymbal wash and strings, fill into the last hook |
| HOOK 4 | 85–96 | 3:54–4:28 | Biggest section |
| outro | 97–99 | 4:28–4:36 | Parts fall away in turn, ends on solo piano |

Hook lines sit at bar +0, +2, +4, +6 with the tag at +8 and +10. Verse lines are
one per bar, sixteen to a verse.

Written so a lead vocal fits on top: verses drop the competing melodic lines
and keep the piano low and sparse, the acoustic guitar's melodic hook plays only
under the sung hook and never over the rap, and the piano's falling figure is
reserved for the opening, the outro and the gaps at the end of each hook line.
Section fader moves (opening −3.0 dB, verses ≈ −1.2 dB, hooks ≈ +1.2 dB) keep
the song breathing rather than sitting at one level.

## Sample libraries used

| Part | Library | Licence |
|---|---|---|
| Piano | **FreePats Upright Piano KW** — Kawai upright recorded in a living room with a Zoom H1, roughly where a player's head would be. Remapped by `tools/track/remap_upright.py`. | CC0 1.0 |
| Acoustic guitar | **FreePats FS Seagull Steel-String** — assembled by roberto@zenvoid.org from FlameStudios' FS Seagull samples, 2 velocity layers per semitone. | GPLv3+ with the FreePats special exception (compositions made with the sounds are *not* covered by the GPL) |
| Drums | **AVL Drumkits 1.0 — Black Pearl 5pc** — real Pearl kit, Sabian/Zildjian cymbals, 5 velocity layers per zone, separate centre and edge zones. Bandshed / GMaq. | GPL |
| Bass (sub + electric) | **Karoryfer Black & Blue Basses** — two 5-string electric basses: blue solidbody with a pick (sub octave), black hollowbody with fingers (doubling). 2000+ samples. | CC0 1.0 |
| Electric guitars | **Karoryfer Black & Green Guitars** — green Gretsch Anniversary and black Höfner Club hollowbodies, 1500+ samples. | CC0 1.0 |
| Strings | **Virtual Playing Orchestra 3.3** (Paul Battersby) — 1st violin section sustain from Sonatina Symphonic Orchestra, cello section sustain from No Budget Orchestra (Jeff Glatt). | VPO imposes no restrictions on making music, including commercially; component samples are CC Sampling Plus 1.0 / CC-BY-SA 4.0 / CC0 |

Also downloaded and auditioned but not used: Salamander Grand Piano V3
(Yamaha C5, CC-BY 3.0) and YDP Grand Piano (Zenph Yamaha Disklavier, CC-BY 3.0).
Total downloaded ≈ 4.5 GB, all free and openly licensed, all script-downloadable
with no account, installer or paywall.

Note on the GPL material: the FreePats bank carries the standard FreePats
exception, which explicitly exempts compositions made with the sounds from the
GPL. AVL Drumkits is GPL without such an exception, which covers the sample
library itself rather than recorded output; if this is ever released
commercially, worth confirming that reading with Bandshed.

## Two library problems that had to be fixed

**The piano mapping.** `Upright Piano KW` ships stretching some samples +3
semitones from their recorded pitch. Piano strings are inharmonic and that
inharmonicity does not transpose, so a 3-semitone stretch reads as the
instrument being out of tune. `remap_upright.py` remaps every key to its nearest
recorded pitch (worst case now ±2) and disables `loop_continuous` on the low
samples, which otherwise sustain forever instead of decaying. It has 2 velocity
layers against Salamander's 16 — that part cannot be fixed by remapping.

**`pedalboard`'s Limiter applies auto-makeup gain**, always pushing up to its
threshold. It drove the master to 0 dBFS and −9.9 LUFS. `mix.py` uses a
hand-written look-ahead limiter that only ever attenuates, which is why the
dynamics survive at PLR 12.8 dB.

## How it was humanized

Straight quantized MIDI through samples sounds fake, so before rendering:

- **Velocity** randomized ±12 around musical accents (downbeats loud, backbeats
  next, offbeats softest). A final pass over the time-ordered stream guarantees
  **zero consecutive notes share a velocity**.
- **Timing** jitter of ±8 ms per note; nothing sits on the grid.
- **Strums** are strummed, not blocked: notes offset 8–20 ms in sequence,
  low-to-high for downstrokes, high-to-low for upstrokes.
- **No pitch is ever restruck while it is still ringing.** This one matters:
  under the sustain pedal, restriking a sounding note starts a second copy of
  the same sample over the first and the two comb-filter — it reads as the piano
  being out of tune. A real piano just re-strikes a vibrating string; a sampler
  cannot. So the chord yields the note to the melody rather than doubling it,
  both voicings are deduped, note lengths are clamped inside their bar, and
  where the melody legitimately repeats a pitch the earlier note is released and
  the pedal blips first. Verified at 0 overlapping same-pitch pairs out of 372.
- **Drums** force each repeated hit into a different velocity layer than the one
  before it, so no two consecutive hits reuse the same sample; backbeats
  alternate between the snare's centre and edge zones, with ghost notes between.
- **Strings** enter with staggered per-voice lead-ins and overlap into the next
  bar so the pad breathes instead of moving in lockstep.
- **Guitars** at the peaks are genuinely double-tracked: the same part performed
  twice with independent timing and velocity, rendered through two different
  real guitars and panned opposite — not one take copied.

## Measured

```
duration    4:39.00 file (4:35.9 of music plus the piano tail)
peak        -1.20 dBFS   no clipped samples
loudness    -14.03 LUFS integrated
dynamics    PLR 12.8 dB, LRA 10.3 LU
tempo       86.1 BPM detected (true 87; detector resolution on a sparse mix)
key         G minor — G leads the opening chord and the track resolves to G
```

Bar 0 reads as digital silence by design — it is the space the vocal enters in.
`verify.py` reports it rather than failing on it.

Fully instrumental. There are no vocals in any of the source libraries.
