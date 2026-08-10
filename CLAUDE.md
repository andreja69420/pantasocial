# Instrumentals — Project Context for Claude Code

## What this project is

This repository is where instrumentals for songs get built — fully synthesized,
mixed, and mastered from first principles in Python (no samples). It is not a
web app and has no unrelated storefront/marketing content; everything in here
should serve the music.

Each track lives in its own subdirectory (e.g. `LTYL_Modern_12Track/`) with its
own `README.md` documenting the production notes for that track: reference
research, synthesis design, arrangement, mix/master chain, and measured QA
results. Read that README before touching a track's code — it's the spec.

## Current track: LTYL_Modern_12Track

`LTYL_Modern_12Track/` — "Love The Way You Lie — Modern Rework", a 56-bar dark
trap/drill instrumental reimagining of the 2010 Eminem/Rihanna ballad. 90 BPM,
G minor, 48 kHz / 24-bit, mastered to −6 dBFS peak to leave headroom for a raw
vocal on top. Full production notes, DSP chain, and QA measurements are in
`LTYL_Modern_12Track/README.md` — that file is long and detailed; treat it as
authoritative over anything summarized here.

### Layout

```
LTYL_Modern_12Track/
  synth.py       DSP primitives + per-track sound generators (all synthesized, no samples)
  arrange.py     56-bar sequencer: chord/root tables, per-zone track gains, sidechain triggers
  mixdsp.py      Per-track Pedalboard effect chains, sidechain ducking, bus summing, master chain
  build.py       End-to-end render -> LoveTheWayYouLie_Modern12Track.wav + stems/
  analyze.py     Automated QA pass: levels, spectral tilt, dynamics, groove grid, pitch/glide -> analysis.png
  README.md      Full production writeup — research, design decisions, measured results
  analysis.png   QA chart output from analyze.py
  venv/          Local virtualenv (gitignored, not committed)
  stems/         Rendered per-track stems, gitignored (regenerate with build.py)
  stems_raw/     Raw pre-mix stems, gitignored
  assets/        Any fetched/cached source material, gitignored
```

Top-level repo also carries the rendered output of the current track:
`LoveTheWayYouLie_Modern12Track.wav` (24-bit master, committed) and
`LoveTheWayYouLie_Modern12Track.mp3`. `LoveTheWayYouLie_Modern12Track_16bit.wav`
is a derived, gitignored artifact from `build.py` — don't hand-edit it.

### Toolchain

```
cd LTYL_Modern_12Track
python3 -m venv venv && ./venv/bin/pip install pydub numpy scipy pedalboard requests librosa
./venv/bin/python build.py      # renders the wav + stems/
./venv/bin/python analyze.py    # QA pass + analysis.png
```

Always use the track's own `venv`, not a system Python — `pedalboard`, `librosa`,
etc. are pinned there.

## Working conventions

- **Determinism matters.** Seeds must be derived from a stable hash (e.g.
  FNV-1a), never Python's built-in `hash()` — it's salted per process and was a
  real bug fixed in this project (see the README's "Also fixed along the way"
  note). A render must be byte-reproducible run to run.
- **Synthesize, don't sample**, unless a track's README says otherwise. Every
  instrument here is generated from DSP primitives so pitch and timing stay
  exact and license-clean.
- **Arrangement is data, not conditionals.** Per-zone track presence belongs in
  a table like `arrange.LAYERS`, not scattered `if section == ...` branches —
  it's what makes the QA scripts able to derive "expected" counts instead of
  hard-coding them.
- **Every mix/arrangement claim should be measured.** This project verifies
  itself: onset counts vs. expected, spectral tilt per band, LUFS/true-peak,
  pitch deviation in cents, sidechain envelope shape, etc., via `analyze.py`.
  When you change the DSP or arrangement, re-run the QA pass and update the
  numbers in the README rather than asserting the change worked.
- **Leave room for the vocal.** The 1–5 kHz "vocal pocket" is deliberately the
  quietest region of the mix, and verses must measure less dense in that band
  than choruses. Don't add melodic content in that range without checking the
  occupancy numbers in the README.
- Master peak target is stated per-track in its README (this track: exactly
  −6.00 dBFS) — headroom is intentional, not a mistake to "fix" toward 0 dBFS.

## What NOT to do

- Do not reintroduce unrelated storefront/marketing/web-app content into this
  repo. This is a music production repo only.
- Do not commit `venv/`, `stems/`, `stems_raw/`, `assets/`, or `__pycache__/` —
  they're gitignored on purpose; regenerate them from `build.py` instead.
- Do not hand-edit the derived 16-bit wav — it comes from `build.py` off the
  committed 24-bit master.
- Do not swap a synthesized instrument for a sampled one, or add a new
  dependency, without a strong reason — the point of this pipeline is
  from-scratch, license-clean, exactly-in-key synthesis.
