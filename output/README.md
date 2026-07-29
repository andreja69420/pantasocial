# Volis me takvu — instrumental

Instrumental bed for the song "Volis me takvu" — E minor, 4/4, 87 BPM, 4:05,
88 bars in song form. No vocals: the arrangement leaves room for them.

Every sound in this track comes from a recording of a real instrument. No AI
music generation, no synthesized waveforms, no sine tones, no General MIDI
sound sets. Each part was sequenced programmatically, humanized, and rendered
through a freely-licensed multi-sampled library with `sfizz_render`.

## Files

| Path | What |
|---|---|
| `volis-me-takvu-instrumental-320.mp3` | 320 kbps CBR, 48 kHz, stereo |
| `volis-me-takvu-instrumental-128.mp3` | 128 kbps version of the same master |
| `volis-me-takvu-timing.txt` | Timecode for every lyric line |
| `everything-i-never-said*.mp3` | Earlier through-composed version, kept for reference |
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

Chord loop Em – C – G – D, one bar per chord, looping throughout. Every section
is a multiple of 4 bars so each one starts on Em; the last two bars hold the
tonic. Hook lines take 2 bars each, verse lines 1 bar each.

| Section | Bars | Time | Arrangement |
|---|---|---|---|
| intro | 0–3 | 0:00–0:11 | Piano alone, rim pickup into the hook |
| HOOK 1 | 4–11 | 0:11–0:33 | Full kit, bass, acoustic, strings enter |
| VERSE 1 | 12–27 | 0:33–1:17 | Rhythm section only — sparse, heavy, space for the rap |
| HOOK 2 | 28–35 | 1:17–1:39 | As hook 1, pushed harder |
| VERSE 2 | 36–51 | 1:39–2:23 | Clean electric arpeggios and cellos come in |
| HOOK 3 | 52–59 | 2:23–2:45 | Double-tracked power chords, full strings |
| VERSE 3 | 60–75 | 2:45–3:29 | Fullest verse — violins added |
| HOOK 4 | 76–83 | 3:29–3:51 | Biggest section |
| outro | 84–87 | 3:51–4:03 | Parts fall away in turn, ends on solo piano |

Written so a lead vocal fits on top: verses drop the competing melodic lines
and keep the piano low and sparse, the acoustic guitar's melodic hook only
plays under the sung hook (never over the rap), and the piano's falling figure
is reserved for the intro, the outro and the gaps at the end of each hook line.
Section fader moves (intro −3.5 dB, verses ≈ −1.2 dB, hooks ≈ +1.2 dB) keep the
song breathing rather than sitting at one level.

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
duration    4:05.00
peak        -1.20 dBFS   no clipped samples
loudness    -14.04 LUFS integrated
dynamics    PLR 12.8 dB, LRA 7.5 LU
tempo       86.1 BPM detected (target 87)
key         E minor — E leads the opening chord and the track resolves to E
```

Fully instrumental. There are no vocals in any of the source libraries.
