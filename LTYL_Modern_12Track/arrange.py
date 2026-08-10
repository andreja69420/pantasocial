"""56-bar arrangement grid at 104 BPM in G minor.

Produces 12 mono buffers (one per track) plus the kick trigger times that the
mix stage needs to build the 808 sidechain envelope.
"""
from __future__ import annotations

import numpy as np

import synth as S
from synth import SR, nf

BPM = 90.0
BEAT = 60.0 / BPM                 # 0.666667 s
BAR = 4.0 * BEAT                  # 2.666667 s
BARS = 56
TAIL = 4.0                        # seconds of room for reverb/808 decay
TOTAL = int((BARS * BAR + TAIL) * SR)

# Chorus(8) -> Verse(16) -> Chorus(8) -> Verse(16) -> Chorus(8)
SECTIONS = [("chorus", 0, 8), ("verse", 8, 24), ("chorus", 24, 32),
            ("verse", 32, 48), ("chorus", 48, 56)]
CHORUS_STARTS = [0, 24, 48]

# Each 16-bar verse is split in half so the arrangement can drop and rebuild
# inside it instead of holding one static texture for 43 seconds.
ZONES = [(0, 8, "chorus1"), (8, 16, "verse1a"), (16, 24, "verse1b"),
         (24, 32, "chorus2"), (32, 40, "verse2a"), (40, 48, "verse2b"),
         (48, 56, "chorus3")]

# Per-track presence, zone by zone. 0.0 means the track is silent there.
# This is the arrangement: chorus 1 is one instrument alone, verse 1 builds,
# chorus 2 is everything, verse 2 drops out and rebuilds, chorus 3 is everything.
LAYERS = {
    "T1":  {"verse1a": 0.90, "verse1b": 1.00, "chorus2": 1.00, "verse2a": 0.70, "verse2b": 0.95, "chorus3": 1.00},
    "T2":  {"verse1b": 0.55, "chorus2": 1.00, "verse2b": 0.60, "chorus3": 1.00},
    "T3":  {"verse1b": 0.70, "chorus2": 1.00, "verse2b": 0.80, "chorus3": 1.00},
    "T5":  {"chorus2": 1.00, "chorus3": 1.00},
    "T6":  {"verse1a": 1.00, "verse1b": 1.00, "chorus2": 1.00, "verse2a": 1.00, "verse2b": 1.00, "chorus3": 1.00},
    "T7":  {"verse1a": 1.00, "verse1b": 1.00, "chorus2": 1.00, "verse2a": 1.00, "verse2b": 1.00, "chorus3": 1.00},
    "T8":  {"verse1a": 0.55, "verse1b": 1.00, "chorus2": 1.00, "verse2a": 0.50, "verse2b": 1.00, "chorus3": 1.00},
    "T9":  {"verse1b": 1.00, "chorus2": 1.00, "verse2b": 1.00, "chorus3": 1.00},
    "T11": {"verse1a": 1.00, "verse1b": 1.00, "chorus2": 1.00, "verse2a": 1.00, "verse2b": 1.00, "chorus3": 1.00},
    "T12": {"verse1a": 0.80, "verse1b": 1.00, "chorus2": 1.00, "verse2a": 0.80, "verse2b": 1.00, "chorus3": 1.00},
    # chorus1 is boosted because the synth is alone there: its mix level was set
    # to sit inside an 11-track chorus, which left the solo intro ~16 dB down.
    "T13": {"chorus1": 3.00, "chorus2": 1.00, "chorus3": 1.00},
}


def section_of(bar: int) -> str:
    for kind, a, b in SECTIONS:
        if a <= bar < b:
            return kind
    return "verse"


def zone_of(bar: int) -> tuple[str, int]:
    for a, b, name in ZONES:
        if a <= bar < b:
            return name, a
    return ZONES[-1][2], ZONES[-1][0]


def lg(track: str, bar: int) -> float:
    """Arrangement gain for a track in a given bar (0.0 = not playing)."""
    return LAYERS.get(track, {}).get(zone_of(bar)[0], 0.0)


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

        # T1 guitar: strum on the downbeat + a 4-note arpeggio through the bar
        if (g1 := lg("T1", bar)):
            for i, n in enumerate(ch["guitar"]):
                place(tracks["T1"], guitar[n], t0 + i * 0.011, 0.62 * g1)  # strum spread
            for i, beat in enumerate((1.0, 2.0, 3.0, 3.5)):
                n = ch["guitar"][(i + 1) % len(ch["guitar"])]
                place(tracks["T1"], guitar_short[n], t0 + beat * BEAT, 0.34 * g1)

        # T2 synth chord: beat 1 only
        if (g2 := lg("T2", bar)):
            place(tracks["T2"], chords[bar % 4], t0, 0.50 * g2)

        # T3 pad: sustained, one chord per bar
        if (g3 := lg("T3", bar)):
            place(tracks["T3"], pads[bar % 4], t0, 0.58 * g3)

    # ------------------------------------------------------ T4 reverse swell
    # Chorus 1 is a bare solo instrument, so it gets no lead-in — the swells
    # announce the two full choruses instead.
    for cs in CHORUS_STARTS:
        if cs != 0:
            place(tracks["T4"], swell, bar_time(cs) - BAR, 0.75)

    # ---------------------------------------------------------- T5 high pluck
    # Dark 3-note counter-melody, choruses only, sitting in the offbeat gaps.
    # Chorus 1 has the lead synth carrying the melody, so the pluck stays sparse
    # there. Choruses 2 and 3 have no synth, so the pluck plays every bar and
    # becomes their melodic signature instead.
    for bar in range(BARS):
        g5 = lg("T5", bar)
        if not g5 or (bar % 2):                                 # alternating bars
            continue
        t0 = bar_time(bar)
        for n, beat, g in (("D5", 2.5, 0.55), ("Bb4", 3.0, 0.45), ("G4", 3.5, 0.50)):
            place(tracks["T5"], plucks[n], t0 + beat * BEAT, g * g5)

    # ------------------------------------------------------------ T6/T7 drums
    for bar in range(BARS):
        t0 = bar_time(bar)
        sect = section_of(bar)
        pos = [0.0, 1.5]                                        # beat 1 + "and" of 2
        if sect == "chorus":
            pos.append(3.0)                                     # beat 4 drive
            if (bar % 4) == 3:
                pos.append(3.5)
        if (g6 := lg("T6", bar)):
            for p in pos:
                t = t0 + p * BEAT
                place(tracks["T6"], kick_s, t, 0.92 * g6)
                kick_times.append(t)                            # sidechain follows real kicks only

        if (g7 := lg("T7", bar)):
            place(tracks["T7"], rim_s, t0 + 2.0 * BEAT, 0.85 * g7)   # beat 3

    # ---------------------------------------------------------------- T8 hats
    for bar in range(BARS):
        t0 = bar_time(bar)
        roll = (bar % 4) in (1, 3)                              # every 2nd and 4th bar
        vel = (1.0 if section_of(bar) == "chorus" else 0.85) * lg("T8", bar)
        if not vel:
            continue
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
        g9 = lg("T9", bar)
        if not g9:
            continue
        t0 = bar_time(bar)
        place(tracks["T9"], ohat_s, t0 + 1.5 * BEAT, 0.44 * g9)  # open hat, "and" of 2
        place(tracks["T9"], wood_s, t0 + 0.5 * BEAT, 0.30 * g9)
        place(tracks["T9"], wood_s, t0 + 2.5 * BEAT, 0.34 * g9)
        if section_of(bar) == "chorus":
            place(tracks["T9"], wood_s, t0 + 3.5 * BEAT, 0.26 * g9)

    # -------------------------------------------------------------- T10 impact
    for cs in CHORUS_STARTS:
        if cs != 0:                                             # not the bare chorus 1
            place(tracks["T10"], impact_s, bar_time(cs), 0.85)

    # ----------------------------------------------------------------- T11 808
    # One root per bar (plus a chorus retrigger), each note gliding out of the
    # previous pitch over 60 ms.
    events: list[tuple[float, float, float]] = []               # (start, dur, freq)
    for bar in range(BARS):
        if not lg("T11", bar):
            continue
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
        if (g := lg("T12", bar)):
            beat = 2.0 if section_of(bar) == "verse" else 0.0
            place(tracks["T12"], chops[order[i % len(order)]], bar_time(bar, beat), 0.42 * g)
    for i, bar in enumerate(range(4, BARS, 8)):
        if (g := lg("T12", bar)):
            place(tracks["T12"], chops[order[(i + 2) % len(order)]], bar_time(bar, 3.0), 0.26 * g)

    # ----------------------------------------------------------- T13 lead riff
    # Choruses only, plus a 2-beat pickup into choruses B and C. Keeping it out
    # of the verses is deliberate: this riff lives in the same range the rap
    # needs, and the whole mix is built around leaving that range empty.
    lead: dict[tuple[str, str], np.ndarray] = {}

    def lead_note(name: str, mode: str) -> np.ndarray:
        key = (name, mode)
        if key not in lead:
            seed = 400 + abs(hash(name)) % 500
            if mode == "soft":
                # Solo-intro voicing. Alone at the top of a melancholic record,
                # the aggressive patch would set entirely the wrong tone, so the
                # same oscillator stack is run with the filter mostly shut, a
                # slower sweep, less detune and almost no drive — haunting
                # rather than snarling. The hard version returns in chorus 2.
                lead[key] = S.lead_pluck(name, 0.62, seed=seed, voices=4,
                                         detune_cents=11.0, f_hi=3000.0,
                                         f_lo=420.0, sweep_tau=0.11, drive=1.25)
            else:
                lead[key] = S.lead_pluck(name, 0.22, seed=seed,
                                         f_lo=620.0 if mode == "dark" else 1350.0,
                                         f_hi=7000.0 if mode == "dark" else 9000.0)
        return lead[key]

    SOFT_SLOTS = [0, 6, 8, 14]              # sparse — lets 8 solo bars breathe
    for bar in range(BARS):
        g13 = lg("T13", bar)
        if not g13:
            continue
        zname, zstart = zone_of(bar)
        pos = bar - zstart
        cell = RIFF_CELLS[PROGRESSION[bar % 4]["name"]]
        # Filter opens in the back half, then the last two bars jump an octave:
        # the "register shift" the original leans on for lift.
        octv = pos >= 6
        if zname == "chorus1":
            for slot, note in zip(SOFT_SLOTS, (cell[0], cell[3], cell[4], cell[7])):
                n = _up_octave(note) if octv else note
                place(tracks["T13"], lead_note(n, "soft"),
                      bar_time(bar, slot * 0.25), 0.58 * g13)
        else:
            mode = "bright" if pos >= 4 else "dark"
            for slot, note in zip(RIFF_SLOTS, cell):
                n = _up_octave(note) if octv else note
                g = 0.62 if slot in (0, 8) else 0.44
                place(tracks["T13"], lead_note(n, mode),
                      bar_time(bar, slot * 0.25), g * g13)

    return tracks, sorted(kick_times)
