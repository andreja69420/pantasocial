# Everything I Never Said

Instrumental track, E minor, 4/4, 87 BPM, 4:03.

Every sound in this track comes from a recording of a real instrument. No AI
music generation, no synthesized waveforms, no sine tones, no General MIDI
sound sets. Each part was sequenced programmatically, humanized, and rendered
through a freely-licensed multi-sampled library with `sfizz_render`.

## Files

| Path | What |
|---|---|
| `everything-i-never-said.mp3` | The deliverable — 320 kbps CBR, 48 kHz, stereo |
| `mix.wav` | 24-bit/48 kHz master (not committed; regenerate with `tools/track/mix.py`) |
| `stems/` | 11 stems, 24-bit/48 kHz WAV (not committed; regenerate with `tools/track/render.py`) |
| `midi/` | 11 humanized MIDI parts, one per instrument |

The WAV stems (463 MB) are deliberately not committed. `tools/track/` regenerates
them bit-for-bit — the composition is seeded, so the output is deterministic.

## Reproducing

```
python3 tools/track/compose.py    # write humanized MIDI parts
python3 tools/track/render.py     # render each part through its sampled library
python3 tools/track/mix.py        # mix, master, limit to -14 LUFS
python3 tools/track/verify.py     # duration/clipping/tempo/key/instrumentation checks
```

Requires `sfizz_render` on PATH plus `mido`, `numpy`, `scipy`, `soundfile`,
`librosa`, `pedalboard`, `pyloudnorm`, and the sample libraries below under
`/home/user/samples`.

## Arrangement

Chord loop Em – C – G – D, one bar per chord, running continuously across the
whole track (87 bars = 240.000 s), resolving to E minor for the last three bars.

| Section | Bars | Time | Content |
|---|---|---|---|
| A | 0–7 | 0:00–0:22 | Solo piano, sparse and exposed |
| B | 8–23 | 0:22–1:06 | Acoustic guitar takes the lead, sub bass, soft rimshot |
| C | 24–31 | 1:06–1:28 | Full drums drop in, strings enter — first lift |
| D | 32–47 | 1:28–2:12 | Pulled back to piano, acoustic and bass; clean electric arpeggios |
| E | 48–57 | 2:12–2:40 | Peak — double-tracked power chords, full strings |
| F | 58–64 | 2:40–2:59 | Breakdown — piano and strings only, drums cut |
| G | 65–79 | 2:59–3:41 | Final section, everything in, crashes, layered guitars |
| H | 80–86 | 3:41–4:00 | Outro — kit, bass, guitar and strings fall away in turn, ends on solo piano |

## Sample libraries used

| Part | Library | Licence |
|---|---|---|
| Piano | **Salamander Grand Piano V3** — Yamaha C5, two AKG C414 in AB, 48 kHz/24-bit, 16 velocity layers, release and resonance samples. Alexander Holm. | CC-BY 3.0 |
| Acoustic guitar | **FreePats FS Seagull Steel-String Acoustic** — assembled by roberto@zenvoid.org from FlameStudios' FS Seagull samples, 2 velocity layers per semitone. | GPLv3+ with the FreePats special exception (compositions made with the sounds are *not* covered by the GPL) |
| Drums | **AVL Drumkits 1.0 — Black Pearl 5pc** — real Pearl kit, Sabian/Zildjian cymbals, 5 velocity layers per zone, separate centre and edge zones. Bandshed / GMaq. | GPL |
| Bass (sub + electric) | **Karoryfer Black & Blue Basses** — two 5-string electric basses: blue solidbody played with a pick (sub octave), black hollowbody played with fingers (doubling). 2000+ samples. | CC0 1.0 |
| Electric guitars | **Karoryfer Black & Green Guitars** — green Gretsch Anniversary and black Höfner Club hollowbodies, 1500+ samples. | CC0 1.0 |
| Strings | **Virtual Playing Orchestra 3.3** (Paul Battersby) — 1st violin section sustain from Sonatina Symphonic Orchestra, cello section sustain from No Budget Orchestra (Jeff Glatt). | VPO imposes no restrictions on making music, including commercially; component samples are CC Sampling Plus 1.0 / CC-BY-SA 4.0 / CC0 |

Total downloaded: **4.2 GB**. Everything above is free and openly licensed,
downloadable by script, with no account, installer or paywall.

Note on the GPL-licensed material: the FreePats bank carries the standard
FreePats exception, which explicitly exempts compositions made with the sounds
from the GPL. AVL Drumkits is GPL without such an exception, which covers the
sample library itself rather than recorded output; if this track is ever
released commercially it is worth confirming that reading with Bandshed.

## How it was humanized

Straight quantized MIDI through samples sounds fake, so before rendering:

- **Velocity** randomized ±12 around musical accents (downbeats loud, backbeats
  next, offbeats softest). A final pass over the time-ordered stream guarantees
  **zero consecutive notes share a velocity** — verified across all 3,015 notes.
- **Timing** jitter of ±8 ms per note; nothing sits on the grid.
- **Strums** are strummed, not blocked: notes offset 8–20 ms in sequence,
  low-to-high for downstrokes, high-to-low for upstrokes (81 strummed chords,
  mean spread 26.6 ms across three strings ≈ 13 ms per string).
- **Piano** chords are rolled by hand-like amounts (12–26 ms), the sustain pedal
  is driven on CC64 per chord, and soft passages are written at genuinely low
  velocities so Salamander's quiet velocity layers actually trigger.
- **Drums** force each repeated hit into a different velocity layer than the one
  before it, so no two consecutive hits reuse the same sample; backbeats
  alternate between the snare's centre and edge zones, with ghost notes between.
- **Strings** enter with staggered per-voice lead-ins and overlap into the next
  bar so the pad breathes instead of starting and stopping in lockstep.
- **Guitars** at the peaks are genuinely double-tracked: the same part performed
  twice with independent timing and velocity, rendered through two different
  real guitars and panned opposite — not one take copied.

## Measured

```
duration    4:03.00      (target 3:55–4:05)
peak        -0.84 dBFS   no clipped samples
loudness    -14.01 LUFS integrated
dynamics    PLR 13.2 dB, LRA 9.0 LU
tempo       86.1 BPM detected (target 87)
key         E minor — profile match plus tonic E at both the intro and the final chord
```

Fully instrumental. There are no vocals in any of the source libraries.
