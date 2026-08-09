"""56-bar arrangement grid at 104 BPM in G minor.

Produces 12 mono buffers (one per track) plus the kick trigger times that the
mix stage needs to build the 808 sidechain envelope.
"""
from __future__ import annotations

import numpy as np

import synth as S
from synth import SR, nf

BPM = 104.0
BEAT = 60.0 / BPM                 # 0.576923 s
BAR = 4.0 * BEAT                  # 2.307692 s
BARS = 56
TAIL = 4.0                        # seconds of room for reverb/808 decay
TOTAL = int((BARS * BAR + TAIL) * SR)

# Chorus(8) -> Verse(16) -> Chorus(8) -> Verse(16) -> Chorus(8)
SECTIONS = [("chorus", 0, 8), ("verse", 8, 24), ("chorus", 24, 32),
            ("verse", 32, 48), ("chorus", 48, 56)]
CHORUS_STARTS = [0, 24, 48]


def section_of(bar: int) -> str:
    for kind, a, b in SECTIONS:
        if a <= bar < b:
            return kind
    return "verse"


# i - VI - III - VII in G minor, one bar each, looping.
#
# 808 roots are octave-placed to stay inside 45-90 Hz rather than following the
# chords literally down to Eb1 (39 Hz): a strict descent would put half the
# bassline below what a phone or laptop can reproduce, and the octave jumps give
# the 60 ms portamento something audible to slide across.
PROGRESSION = [
    {"name": "Gm", "guitar": ["G3", "Bb3", "D4", "G4"], "piano": ["G2", "D3", "G3", "Bb3", "D4"],
     "pad": ["G3", "Bb3", "D4"], "root": "G1"},     # 49.0 Hz
    {"name": "Eb", "guitar": ["Eb3", "G3", "Bb3", "Eb4"], "piano": ["Eb2", "Bb2", "Eb3", "G3", "Bb3"],
     "pad": ["Eb3", "G3", "Bb3"], "root": "Eb2"},   # 77.8 Hz
    {"name": "Bb", "guitar": ["Bb2", "D3", "F3", "Bb3"], "piano": ["Bb1", "F2", "Bb2", "D3", "F3"],
     "pad": ["Bb2", "D3", "F3"], "root": "Bb1"},    # 58.3 Hz
    {"name": "F",  "guitar": ["F3", "A3", "C4", "F4"], "piano": ["F2", "C3", "F3", "A3", "C4"],
     "pad": ["F3", "A3", "C4"], "root": "F2"},      # 87.3 Hz
]


def bar_time(bar: int, beat: float = 0.0) -> float:
    """Absolute seconds for a 0-indexed bar and 0-indexed beat offset."""
    return bar * BAR + beat * BEAT


def place(buf: np.ndarray, x: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(round(at * SR))
    if i < 0:                      # events scheduled before the downbeat get clipped
        x = x[-i:]
        if len(x):                 # ...and re-faded, or the cut leaves a click at t=0
            f = min(int(0.015 * SR), len(x))
            x = x.copy()
            x[:f] *= np.linspace(0.0, 1.0, f)
        i = 0
    if i >= len(buf) or len(x) == 0:
        return
    n = min(len(x), len(buf) - i)
    buf[i:i + n] += x[:n] * gain


# T13 lead riff. Original melodic content written for this track — the synth
# *design* is modelled on the Godzilla intro (detuned saw growl, hard filter-
# swept pluck), but the line itself outlines our own Gm-Eb-Bb-F loop rather
# than transcribing anyone's hook.
#
# Eight staccato 16ths per bar, syncopated so the figure pushes against the
# kick instead of doubling it.
RIFF_SLOTS = [0, 2, 3, 6, 8, 10, 11, 14]          # in 16ths
RIFF_CELLS = {
    "Gm": ["G4", "G4", "Bb4", "D5", "G4", "Bb4", "D5", "C5"],
    "Eb": ["Eb4", "Eb4", "G4", "Bb4", "Eb4", "G4", "Bb4", "D5"],
    "Bb": ["Bb3", "Bb3", "D4", "F4", "Bb3", "D4", "F4", "Eb4"],
    "F":  ["F4", "F4", "A4", "C5", "F4", "A4", "C5", "Bb4"],
}


def _up_octave(name: str) -> str:
    k = len(name)
    while name[k - 1].isdigit():
        k -= 1
    return f"{name[:k]}{int(name[k:]) + 1}"


def build() -> tuple[dict[str, np.ndarray], list[float]]:
    tracks = {f"T{i}": np.zeros(TOTAL) for i in range(1, 14)}
    kick_times: list[float] = []

    # ---------------------------------------------------------- asset cache
    # Two guitar lengths: the downbeat strum rings, the arpeggio is choked
    # short so eight overlapping voices per bar never turn to mud.
    guitar, guitar_short = {}, {}
    for ch in PROGRESSION:
        for n in ch["guitar"]:
            if n not in guitar:
                sd = abs(hash(n)) % 10_000
                guitar[n] = S.guitar_note(n, 2.2, seed=sd)
                guitar_short[n] = S.guitar_note(n, 0.95, seed=sd)
    chords = [S.synth_chord(ch["piano"], 3.0, seed=500 + i)
              for i, ch in enumerate(PROGRESSION)]
    pads = [S.pad_chord(ch["pad"], BAR * 1.12, seed=100 + i) for i, ch in enumerate(PROGRESSION)]

    swell = S.reverse_swell(BAR, seed=7)
    # Chorus 1 sits at bar 0, so its lead-in falls off the front of the grid.
    # A 2-bar swell placed 2 bars early leaves exactly the final bar audible,
    # still cresting precisely on the downbeat.
    swell_intro = S.reverse_swell(BAR * 2, seed=7)
    plucks = {n: S.pluck(n, 1.1, seed=200 + i) for i, n in enumerate(("D5", "Bb4", "G4"))}
    kick_s = S.kick()
    rim_s = S.rimshot()
    hat_s = S.hihat()
    hat_soft = S.hihat(0.042, seed=33)
    ohat_s = S.open_hat()
    wood_s = S.woodblock()
    impact_s = S.impact()
    chops = {n: S.vocal_chop(n, 2.6, seed=300 + i) for i, n in enumerate(("G4", "Bb4", "D5"))}

    # ------------------------------------------------- T1/T2/T3 harmonic bed
    for bar in range(BARS):
        ch = PROGRESSION[bar % 4]
        t0 = bar_time(bar)
        sect = section_of(bar)

        # Verses pull the whole harmonic bed back so the rap sits on top of the
        # drums; choruses open it back up. This is the main verse/chorus lift.
        lift = 1.0 if sect == "chorus" else 0.72

        # T1 guitar: strum on the downbeat + a 4-note arpeggio through the bar
        for i, n in enumerate(ch["guitar"]):
            place(tracks["T1"], guitar[n], t0 + i * 0.011, 0.62 * lift)   # strum spread
        for i, beat in enumerate((1.0, 2.0, 3.0, 3.5)):
            n = ch["guitar"][(i + 1) % len(ch["guitar"])]
            place(tracks["T1"], guitar_short[n], t0 + beat * BEAT, 0.34 * lift)

        # T2 synth chord: beat 1 only
        place(tracks["T2"], chords[bar % 4], t0, 0.50 * lift)

        # T3 pad: sustained, one chord per bar
        place(tracks["T3"], pads[bar % 4], t0, 0.58 * lift)

    # ------------------------------------------------------ T4 reverse swell
    for cs in CHORUS_STARTS:
        if cs == 0:
            place(tracks["T4"], swell_intro, bar_time(cs) - 2 * BAR, 0.75)
        else:
            place(tracks["T4"], swell, bar_time(cs) - BAR, 0.75)

    # ---------------------------------------------------------- T5 high pluck
    # Dark 3-note counter-melody, choruses only, sitting in the offbeat gaps.
    # Chorus 1 has the lead synth carrying the melody, so the pluck stays sparse
    # there. Choruses 2 and 3 have no synth, so the pluck plays every bar and
    # becomes their melodic signature instead.
    for kind, a, b in SECTIONS:
        if kind != "chorus":
            continue
        dense = a != 0
        for bar in range(a, b):
            if not dense and (bar - a) % 2:
                continue
            t0 = bar_time(bar)
            for n, beat, g in (("D5", 2.5, 0.55), ("Bb4", 3.0, 0.45), ("G4", 3.5, 0.50)):
                place(tracks["T5"], plucks[n], t0 + beat * BEAT, g * (1.1 if dense else 1.0))

    # ------------------------------------------------------------ T6/T7 drums
    for bar in range(BARS):
        t0 = bar_time(bar)
        sect = section_of(bar)
        pos = [0.0, 1.5]                                        # beat 1 + "and" of 2
        if sect == "chorus":
            pos.append(3.0)                                     # beat 4 drive
            if (bar % 4) == 3:
                pos.append(3.5)
        for p in pos:
            t = t0 + p * BEAT
            place(tracks["T6"], kick_s, t, 0.92)
            kick_times.append(t)

        place(tracks["T7"], rim_s, t0 + 2.0 * BEAT, 0.85)       # beat 3, every bar

    # ---------------------------------------------------------------- T8 hats
    for bar in range(BARS):
        t0 = bar_time(bar)
        roll = (bar % 4) in (1, 3)                              # every 2nd and 4th bar
        vel = 1.0 if section_of(bar) == "chorus" else 0.85
        for i in range(8):
            beat = i * 0.5
            if roll and beat >= 3.0:
                continue
            g = 0.62 if i % 2 == 0 else 0.42
            place(tracks["T8"], hat_s if i % 2 == 0 else hat_soft, t0 + beat * BEAT, g * vel)
        if roll:
            for j in range(8):                                  # 1/32 roll on beat 4
                beat = 3.0 + j * 0.125
                place(tracks["T8"], hat_soft, t0 + beat * BEAT, (0.30 + 0.045 * j) * vel)

    # ---------------------------------------------------------------- T9 perc
    for bar in range(BARS):
        t0 = bar_time(bar)
        place(tracks["T9"], ohat_s, t0 + 1.5 * BEAT, 0.44)      # open hat, "and" of 2
        place(tracks["T9"], wood_s, t0 + 0.5 * BEAT, 0.30)
        place(tracks["T9"], wood_s, t0 + 2.5 * BEAT, 0.34)
        if section_of(bar) == "chorus":
            place(tracks["T9"], wood_s, t0 + 3.5 * BEAT, 0.26)

    # -------------------------------------------------------------- T10 impact
    for cs in CHORUS_STARTS:
        place(tracks["T10"], impact_s, bar_time(cs), 0.85)

    # ----------------------------------------------------------------- T11 808
    # One root per bar (plus a chorus retrigger), each note gliding out of the
    # previous pitch over 60 ms.
    events: list[tuple[float, float, float]] = []               # (start, dur, freq)
    for bar in range(BARS):
        ch = PROGRESSION[bar % 4]
        f = nf(ch["root"])
        t0 = bar_time(bar)
        if section_of(bar) == "chorus":
            events.append((t0, 2.5 * BEAT + 0.06, f))
            events.append((t0 + 2.5 * BEAT, 1.5 * BEAT + 0.06, f))
        else:
            events.append((t0, 4.0 * BEAT + 0.06, f))

    prev_f = None
    for start, dur, f in events:
        n = int(dur * SR)
        freqs = np.full(n, f)
        seg = S.sub808(freqs, dur, glide_ms=60.0, start_freq=prev_f)
        gain = 0.95 if section_of(int(start // BAR)) == "chorus" else 0.95 * 10 ** (-2.0 / 20)
        gain *= (60.0 / f) ** 0.25          # even out perceived level across octaves
        place(tracks["T11"], seg, start, gain)
        prev_f = f

    # -------------------------------------------------------- T12 vocal chops
    # Sparse: one long chop every 8 bars, alternating pitch.
    order = ["G4", "Bb4", "D5", "Bb4"]
    for i, bar in enumerate(range(0, BARS, 8)):
        beat = 2.0 if section_of(bar) == "verse" else 0.0
        place(tracks["T12"], chops[order[i % len(order)]], bar_time(bar, beat), 0.42)
    for i, bar in enumerate(range(4, BARS, 8)):
        place(tracks["T12"], chops[order[(i + 2) % len(order)]], bar_time(bar, 3.0), 0.26)

    # ----------------------------------------------------------- T13 lead riff
    # Choruses only, plus a 2-beat pickup into choruses B and C. Keeping it out
    # of the verses is deliberate: this riff lives in the same range the rap
    # needs, and the whole mix is built around leaving that range empty.
    lead: dict[tuple[str, int, int], np.ndarray] = {}

    def lead_note(name: str, bright: int, seed: int) -> np.ndarray:
        key = (name, bright, 0)
        if key not in lead:
            lead[key] = S.lead_pluck(name, 0.22, seed=seed,
                                     f_lo=620.0 if bright == 0 else 1350.0,
                                     f_hi=7000.0 if bright == 0 else 9000.0)
        return lead[key]

    for kind, a, b in SECTIONS:
        if kind != "chorus" or a != 0:      # opening statement only
            continue
        for bar in range(a, b):
            pos = bar - a
            cell = RIFF_CELLS[PROGRESSION[bar % 4]["name"]]
            # Filter opens in the back half, then the last two bars jump an
            # octave: the "register shift" the original leans on for lift.
            bright = 1 if pos >= 4 else 0
            octv = pos >= 6
            for slot, note in zip(RIFF_SLOTS, cell):
                n = _up_octave(note) if octv else note
                g = 0.62 if slot in (0, 8) else 0.44
                place(tracks["T13"], lead_note(n, bright, 400 + hash(n) % 500),
                      bar_time(bar, slot * 0.25), g)

    return tracks, sorted(kick_times)
