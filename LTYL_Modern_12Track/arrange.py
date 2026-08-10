"""56-bar arrangement grid at 90 BPM in G minor.

Produces 13 mono buffers (one per track) plus the kick trigger times that the
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
    # chorus1 is boosted because T13 is alone there and its mix level is set to
    # sit inside an 11-track chorus. The factor is far smaller than the synth
    # needed: a five-note piano chord is a much bigger sound than a filtered
    # single-note pluck, and 3.0 put the intro 6.5 dB above the full choruses.
    # Strings enter with the build and carry the full sections, the way the
    # 2010 record uses them. Never in the bare chorus 1.
    "T14": {"verse1b": 0.55, "chorus2": 1.00, "verse2b": 0.65, "chorus3": 1.00},
    "T13": {"chorus1": 2.20, "verse1a": 0.62, "verse1b": 0.70, "chorus2": 1.00,
            "verse2a": 0.52, "verse2b": 0.66, "chorus3": 1.00},
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
     "pad": ["G3", "Bb3", "D4"], "strings": ["D4", "G4", "Bb4", "D5"], "root": "G1"},
    {"name": "Eb", "guitar": ["Eb3", "G3", "Bb3", "Eb4"], "piano": ["Eb2", "Bb2", "Eb3", "G3", "Bb3"],
     "pad": ["Eb3", "G3", "Bb3"], "strings": ["Eb4", "G4", "Bb4", "Eb5"], "root": "Eb2"},
    {"name": "Bb", "guitar": ["Bb2", "D3", "F3", "Bb3"], "piano": ["Bb1", "F2", "Bb2", "D3", "F3"],
     "pad": ["Bb2", "D3", "F3"], "strings": ["D4", "F4", "Bb4", "D5"], "root": "Bb1"},
    {"name": "F",  "guitar": ["F3", "A3", "C4", "F4"], "piano": ["F2", "C3", "F3", "A3", "C4"],
     "pad": ["F3", "A3", "C4"], "strings": ["C4", "F4", "A4", "C5"], "root": "F2"},
]


def chord_index(bar: int) -> int:
    """Which chord is sounding in this bar.

    The progression moves every TWO bars, not every bar. The piano brief is
    written around 2-bar chord sections ("3 hits per chord, 1 chord per 2
    bars"), and harmonic rhythm has to be global — if only the piano slowed
    down it would sit on Gm while the guitar and 808 had already moved to Eb.
    All section boundaries (bars 0, 8, 24, 32, 48) land on Gm under this.
    """
    return (bar // 2) % 4


def _fnv(key: str) -> float:
    """Deterministic [-1, 1] from a string.

    Python's built-in hash() is salted per process, so seeds derived from it
    changed on every run and the render was not byte-reproducible. FNV-1a is
    stable across processes and machines.
    """
    h = 2166136261
    for c in key:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    return (h / 0xFFFFFFFF) * 2.0 - 1.0


def seed_of(key: str, span: int) -> int:
    return int((_fnv(key) + 1.0) * 0.5 * span)


# Micro-timing. Real drums are never dead on the grid; 100% quantisation is a
# large part of what makes programmed drums read as programmed. Offsets are
# deterministic per (track, bar, position), so the render stays reproducible.
HUMANIZE_MS = {"T6": 2.5, "T7": 2.0, "T8": 4.5, "T9": 5.5, "T13": 4.0}
SWING = 0.56          # offbeat 8ths pushed 6% of a beat late


def humanize(track: str, bar: int, pos: float) -> float:
    ms = HUMANIZE_MS.get(track, 0.0)
    return _fnv(f"{track}|{bar}|{pos}") * ms / 1000.0 if ms else 0.0


def vel(track: str, bar: int, pos: float, depth: float = 0.16) -> float:
    """Deterministic velocity jitter around 1.0."""
    return 1.0 + _fnv(f"v{track}|{bar}|{pos}") * depth


def swing(beat: float, ratio: float = SWING) -> float:
    """Push offbeat 8ths later. Straight 8ths sit at .5; swung sit at `ratio`."""
    return beat + (ratio - 0.5) if round(beat * 2) % 2 else beat


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


# T13 piano stabs. Three-note close voicings — one hand — deliberately kept in
# the F4-Eb5 register: no low keys held down, nothing to muddy the 808 or crowd
# the low mids. Voice-led so the top line moves by step and common tones hold:
#   Gm(G4 Bb4 D5) -> Eb(G4 Bb4 Eb5) -> Bb(F4 Bb4 D5) -> F(F4 A4 C5)
STAB_VOICINGS = {
    "Gm": ["G4", "Bb4", "D5"],
    "Eb": ["G4", "Bb4", "Eb5"],
    "Bb": ["F4", "Bb4", "D5"],
    "F":  ["F4", "A4", "C5"],
}

# Three hits spread across the whole 2-bar chord section rather than crammed
# into the first bar. Beat offsets are 0-indexed over 8 beats: bar 1 downbeat,
# the "and" of 3 in bar 1, then beat 2 of bar 2 — leaving 3 beats to breathe
# before the chord switches. The original is a midtempo ballad whose piano sets
# a somber tone; hits packed into four beats read as busy rather than somber.
STAB_HITS = [0.0, 2.5, 5.0]


def build() -> tuple[dict[str, np.ndarray], list[float]]:
    tracks = {f"T{i}": np.zeros(TOTAL) for i in range(1, 15)}
    kick_times: list[float] = []

    # ---------------------------------------------------------- asset cache
    # Two guitar lengths: the downbeat strum rings, the arpeggio is choked
    # short so eight overlapping voices per bar never turn to mud. The electric
    # sustains far longer than the acoustic did, so the arpeggio is choked
    # harder (0.95 -> 0.70 s) to keep the same amount of space.
    guitar, guitar_short = {}, {}
    for ch in PROGRESSION:
        for n in ch["guitar"]:
            if n not in guitar:
                sd = seed_of(n, 10_000)
                guitar[n] = S.electric_note(n, 2.4, seed=sd, release=0.55)
                guitar_short[n] = S.electric_note(n, 0.80, seed=sd, release=0.22)
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
    rim_s = S.snare_layered()
    hats = [S.hihat(0.055, seed=31, pitch=1.00),
            S.hihat(0.047, seed=33, pitch=1.09),
            S.hihat(0.062, seed=37, pitch=0.93)]
    ohat_s = S.open_hat()
    wood_s = S.woodblock()
    impact_s = S.impact()
    chops = {n: S.vocal_chop(n, 2.6, seed=300 + i) for i, n in enumerate(("G4", "Bb4", "D5"))}
    strings = [S.strings_chord(ch["strings"], BAR * 2.15, seed=600 + i)
               for i, ch in enumerate(PROGRESSION)]

    # ------------------------------------------------- T1/T2/T3 harmonic bed
    for bar in range(BARS):
        ch = PROGRESSION[chord_index(bar)]
        t0 = bar_time(bar)
        sect = section_of(bar)

        # T1 guitar: strum on the downbeat + a 4-note arpeggio through the bar
        if (g1 := lg("T1", bar)):
            for i, n in enumerate(ch["guitar"]):
                place(tracks["T1"], guitar[n], t0 + i * 0.011, 0.62 * g1)  # strum spread
            # Four arpeggio notes under the choruses, two under the verses. At
            # 8 events/bar the guitar was the densest melodic source, and the
            # verses have to leave room for a vocal.
            arp = (1.0, 2.0, 3.0, 3.5) if sect == "chorus" else (2.0, 3.5)
            for i, beat in enumerate(arp):
                n = ch["guitar"][(i + 1) % len(ch["guitar"])]
                place(tracks["T1"], guitar_short[n], t0 + beat * BEAT, 0.34 * g1)

        # T2 synth chord: beat 1 only
        if (g2 := lg("T2", bar)):
            place(tracks["T2"], chords[chord_index(bar)], t0, 0.50 * g2)

        # T3 pad: sustained, one chord per bar
        if (g3 := lg("T3", bar)):
            place(tracks["T3"], pads[chord_index(bar)], t0, 0.58 * g3)

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
                t = t0 + p * BEAT + humanize("T6", bar, p)
                place(tracks["T6"], kick_s, t, 0.92 * g6 * vel("T6", bar, p, 0.10))
                kick_times.append(t)                            # sidechain follows real kicks only

        if (g7 := lg("T7", bar)):
            t = t0 + 2.0 * BEAT + humanize("T7", bar, 2.0)      # beat 3
            place(tracks["T7"], rim_s, t, 0.85 * g7 * vel("T7", bar, 2.0, 0.08))

    # ---------------------------------------------------------------- T8 hats
    for bar in range(BARS):
        t0 = bar_time(bar)
        base = (1.0 if section_of(bar) == "chorus" else 0.85) * lg("T8", bar)
        if not base:
            continue
        zname, zstart = zone_of(bar)
        roll32 = (bar % 4) in (1, 3)                            # every 2nd and 4th bar
        trip = (bar - zstart) == 7                              # triplet fill closes each zone

        for i in range(8):                                      # straight 8ths, swung
            beat = swing(i * 0.5)
            if (roll32 or trip) and beat >= 3.0:
                continue
            # accent the downbeats, duck the offbeats, then jitter both
            accent = 0.62 if i % 2 == 0 else 0.40
            pitch = i % 3                                       # rotate hat timbre
            place(tracks["T8"], hats[pitch], t0 + beat * BEAT + humanize("T8", bar, beat),
                  accent * base * vel("T8", bar, beat))

        if trip:                                                # 1/8 triplets on beat 4
            for j in range(6):
                b = 3.0 + j * (1.0 / 3.0)
                place(tracks["T8"], hats[(j + 1) % 3], t0 + b * BEAT + humanize("T8", bar, b),
                      (0.34 + 0.05 * j) * base * vel("T8", bar, b))
        elif roll32:                                            # 1/32 roll on beat 4
            for j in range(8):
                b = 3.0 + j * 0.125
                place(tracks["T8"], hats[j % 3], t0 + b * BEAT + humanize("T8", bar, b),
                      (0.30 + 0.045 * j) * base * vel("T8", bar, b, 0.10))

    # ---------------------------------------------------------------- T9 perc
    for bar in range(BARS):
        g9 = lg("T9", bar)
        if not g9:
            continue
        t0 = bar_time(bar)
        for src, b, g in ((ohat_s, 1.5, 0.44), (wood_s, 0.5, 0.30), (wood_s, 2.5, 0.34)):
            sb = swing(b)
            place(tracks["T9"], src, t0 + sb * BEAT + humanize("T9", bar, b),
                  g * g9 * vel("T9", bar, b))
        if section_of(bar) == "chorus":
            sb = swing(3.5)
            place(tracks["T9"], wood_s, t0 + sb * BEAT + humanize("T9", bar, 3.5),
                  0.26 * g9 * vel("T9", bar, 3.5))

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
        ch = PROGRESSION[chord_index(bar)]
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
    keys: dict[tuple[str, str], np.ndarray] = {}

    def key_note(name: str, touch: str, ring: str) -> np.ndarray:
        key = (name, touch, ring)
        if key not in keys:
            seed = 700 + seed_of(name, 900)
            # Velocity is a timbre control on a real piano, not just a level:
            # the soft touch is darker, not merely quieter. `ring` sets how long
            # the note is allowed to sound before the damper lands.
            # Length includes the damper release, so the note is still ringing
            # freely for dur-release and is then damped rather than truncated.
            dur, rel = (3.4, 0.70) if ring == "long" else (1.75, 0.40)
            keys[key] = S.steinway_note(name, dur, seed=seed, release=rel,
                                        velocity=0.50 if touch == "soft" else 0.82)
        return keys[key]

    # Chord stabs: three hits in the first bar of each 2-bar chord section,
    # then the second bar breathes. Over an 8-bar chorus that is 4 chords x 3
    # hits = 12 stabs, which is exactly the brief.
    for bar in range(BARS):
        g13 = lg("T13", bar)
        if not g13 or bar % 2:              # stabs live in the first bar only
            continue
        ch = PROGRESSION[chord_index(bar)]
        voicing = STAB_VOICINGS[ch["name"]]
        touch = "soft" if zone_of(bar)[0] == "chorus1" else "hard"
        for hit, beat in enumerate(STAB_HITS):
            # The third stab is the one that rings through the empty bar; the
            # first two are choked short so they read as stabs, not chords.
            ring = "long" if hit == 2 else "short"
            accent = (0.62, 0.50, 0.56)[hit]
            for i, n in enumerate(voicing):
                place(tracks["T13"], key_note(n, touch, ring),
                      bar_time(bar, beat) + i * 0.006, accent * g13)

    # ------------------------------------------------------------ T14 strings
    for bar in range(0, BARS, 2):                               # one chord per 2 bars
        if (g14 := lg("T14", bar)):
            place(tracks["T14"], strings[chord_index(bar)], bar_time(bar), 0.52 * g14)

    return tracks, sorted(kick_times)
