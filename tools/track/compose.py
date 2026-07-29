"""
Compose "Everything I Never Said" - E minor, 4/4, 87 BPM, 87 bars = 240.000s

Writes one humanized MIDI file per instrument into output/midi/.
Every part is written for a real multi-sampled library; nothing here is
synthesized.
"""
import os
import random

import mido

random.seed(20260729)

BPM = 87
SPB = 60.0 / BPM              # 0.689655 s per beat
BAR = 4 * SPB                 # 2.758621 s per bar
TPB = 480                     # ticks per beat
TEMPO = mido.bpm2tempo(BPM)
TOTAL_BARS = 87               # 87 * BAR = 240.000 s exactly
TAIL = 7.0                    # let final piano + reverb tails decay

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))),
                   "output", "midi")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
# Harmony: Em - C - G - D, one bar each, looping continuously across the whole
# track (bar % 4), so the cycle never breaks even where sections start midway.
# --------------------------------------------------------------------------
CH = ["Em", "C", "G", "D"]


FINAL_CADENCE = 84  # from here the loop resolves and sits on the tonic


def chord_at(bar):
    # The outro has to land on E minor rather than wherever the loop happens
    # to be, so the last three bars hold the tonic while parts drop away.
    if bar >= FINAL_CADENCE:
        return "Em"
    return CH[bar % 4]


# Section boundaries in bars. Chosen so the wall-clock lands on the requested
# arrangement timecodes (bar b starts at b * 2.758621 s).
SECTIONS = {
    "A": (0, 8),     # 0:00.0 - 0:22.1  solo piano
    "B": (8, 24),    # 0:22.1 - 1:06.2  acoustic guitar lead, sub bass, rimshot
    "C": (24, 32),   # 1:06.2 - 1:28.3  full drums + strings, first lift
    "D": (32, 48),   # 1:28.3 - 2:12.4  pulled back, rebuilding
    "E": (48, 58),   # 2:12.4 - 2:40.0  peak: electric guitars + strings
    "F": (58, 65),   # 2:40.0 - 2:59.3  breakdown: piano + strings only
    "G": (65, 80),   # 2:59.3 - 3:40.7  final section, everything in
    "H": (80, 87),   # 3:40.7 - 4:00.0  outro, falls away to solo piano
}


def sec_bars(name, until=None):
    """Bars of a section. `until` truncates it, which is how the outro drops
    instruments one at a time."""
    a, b = SECTIONS[name]
    return range(a, min(b, until) if until is not None else b)


# Outro: who stops when, so the track thins out one instrument at a time
# and the last bar is piano alone.
OUTRO_END = {
    "drums": 83,            # 3:49 - kit goes first
    "bass_sub": 84,         # 3:52
    "acoustic_guitar": 85,  # 3:54
    "strings_violin": 85,   # 3:54
    "strings_cello": 86,    # 3:57
}


def bar_t(bar, beat=0.0):
    """Absolute seconds for a bar + beat offset."""
    return bar * BAR + beat * SPB


# --------------------------------------------------------------------------
# Humanizing track writer
# --------------------------------------------------------------------------
class Track:
    """Collects notes/CCs and writes a humanized MIDI file.

    Humanization applied here rather than at note-entry so every part gets
    the same treatment: velocity jitter with no two consecutive velocities
    equal, timing jitter, and note-length variation.
    """

    def __init__(self, name, jitter=0.008, len_var=0.10):
        self.name = name
        self.jitter = jitter
        self.len_var = len_var
        self.notes = []   # (t, note, vel, dur)
        self.ccs = []     # (t, cc, val)
        self._last_vel = None
        self._last_bucket = {}

    def n(self, t, note, vel, dur, jitter=None, exact=False):
        """Add one note. vel is the musical target; jitter is applied here."""
        if jitter is None:
            jitter = self.jitter
        vel = self._vary_vel(vel)
        if not exact:
            t = t + random.uniform(-jitter, jitter)
            dur = dur * random.uniform(1.0 - self.len_var, 1.0 + self.len_var)
        t = max(0.0, t)
        self.notes.append((t, int(note), vel, max(0.03, dur)))

    def drum(self, t, key, vel, dur=0.22, jitter=0.008):
        """Drum hit that forces a different velocity layer than the previous
        hit on the same zone, so repeated hits never reuse the same sample.

        AVL layer edges: 1-26, 27-52, 53-77, 78-102, 103-127.
        """
        vel = int(max(1, min(127, vel + random.randint(-12, 12))))
        b = self._bucket(vel)
        if self._last_bucket.get(key) == b:
            vel = self._shift_bucket(vel, b)
        self._last_bucket[key] = self._bucket(vel)
        if vel == self._last_vel:
            vel = max(1, min(127, vel + random.choice((-2, -1, 1, 2))))
        self._last_vel = vel
        t = max(0.0, t + random.uniform(-jitter, jitter))
        self.notes.append((t, int(key), vel, dur))

    @staticmethod
    def _bucket(v):
        for i, hi in enumerate((26, 52, 77, 102, 127)):
            if v <= hi:
                return i
        return 4

    @staticmethod
    def _shift_bucket(v, b):
        # nudge into an adjacent velocity layer, staying musically close
        edges = [(1, 26), (27, 52), (53, 77), (78, 102), (103, 127)]
        cand = [i for i in (b - 1, b + 1) if 0 <= i <= 4]
        lo, hi = edges[random.choice(cand)]
        if b < 4 and v > 100:
            lo, hi = edges[max(0, b - 1)]
        return random.randint(lo, hi)

    def _vary_vel(self, vel):
        v = int(round(vel + random.uniform(-12, 12)))
        v = max(1, min(127, v))
        if v == self._last_vel:
            v = max(1, min(127, v + random.choice((-3, -2, -1, 1, 2, 3))))
        self._last_vel = v
        return v

    def strum(self, t, notes, vel, dur, down=True, spread=(0.008, 0.020),
              accent_top=0):
        """Real strum: notes offset in sequence, low->high for a downstroke,
        high->low for an upstroke."""
        seq = sorted(notes) if down else sorted(notes, reverse=True)
        step = random.uniform(*spread)
        for i, nn in enumerate(seq):
            v = vel + (accent_top if nn == max(seq) else 0)
            # strings struck later in the stroke ring marginally shorter
            self.n(t + i * step * random.uniform(0.8, 1.2), nn, v,
                   dur - i * 0.01, jitter=0.003)

    def cc(self, t, num, val):
        self.ccs.append((max(0.0, t), num, int(max(0, min(127, val)))))

    def save(self):
        mf = mido.MidiFile(ticks_per_beat=TPB)
        tr = mido.MidiTrack()
        mf.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=self.name, time=0))
        tr.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
        tr.append(mido.MetaMessage("time_signature", numerator=4,
                                   denominator=4, time=0))

        ev = []
        for (t, note, vel, dur) in self.notes:
            ev.append((t, 1, mido.Message("note_on", note=note, velocity=vel)))
            ev.append((t + dur, 0,
                       mido.Message("note_off", note=note, velocity=0)))
        for (t, num, val) in self.ccs:
            ev.append((t, 0, mido.Message("control_change",
                                          control=num, value=val)))
        ev.sort(key=lambda x: (x[0], x[1]))

        # Velocities are varied as notes are written, but parts are written
        # voice by voice rather than in time order. Sweep once more over the
        # final time-ordered stream so no two consecutive note-ons share a
        # velocity - identical back-to-back velocities are the giveaway.
        prev = None
        for (_, kind, msg) in ev:
            if kind != 1:
                continue
            if msg.velocity == prev:
                step = random.choice((-3, -2, -1, 1, 2, 3))
                msg.velocity = max(1, min(127, msg.velocity + step))
            prev = msg.velocity

        last = 0
        for (t, _, msg) in ev:
            tick = int(round(mido.second2tick(t, TPB, TEMPO)))
            msg.time = max(0, tick - last)
            last = max(last, tick)
            tr.append(msg)

        end = int(round(mido.second2tick(TOTAL_BARS * BAR + TAIL, TPB, TEMPO)))
        tr.append(mido.MetaMessage("end_of_track", time=max(1, end - last)))
        path = os.path.join(OUT, f"{self.name}.mid")
        mf.save(path)
        print(f"  {self.name:<16} {len(self.notes):>5} notes  "
              f"{len(self.ccs):>4} cc  -> {os.path.basename(path)}")
        return path


def accent(beat):
    """Musical accent weighting: downbeats loudest, backbeats next,
    offbeats softest."""
    b = beat % 4
    if abs(b - 0) < 0.01:
        return 8
    if abs(b - 2) < 0.01:
        return 3
    if abs(b % 1) < 0.01:
        return 0
    return -7


# ==========================================================================
# PIANO  (Salamander Grand Piano, 16 velocity layers)
# ==========================================================================
PIANO_VOICE = {
    "Em": (40, [59, 64, 67]),
    "C":  (36, [60, 64, 67]),
    "G":  (43, [59, 62, 67]),
    "D":  (38, [57, 62, 66]),
}
# Falling melodic figure, one 4-bar phrase: (beat, note)
PIANO_FIG = {
    0: [(0.0, 71), (2.0, 67)],
    1: [(0.0, 64), (1.5, 67), (3.0, 64)],
    2: [(0.0, 62), (2.0, 59)],
    3: [(0.0, 57), (2.0, 66)],
}


def build_piano():
    t = Track("piano", jitter=0.009, len_var=0.14)
    # dynamic profile per section: (chord vel, melody vel, density)
    prof = {
        "A": (44, 52, "sparse"),
        "B": (40, 0,  "pad"),
        "C": (58, 64, "pad"),
        "D": (50, 60, "sparse"),
        "E": (74, 84, "pad"),
        "F": (48, 58, "sparse"),
        "G": (80, 92, "pad"),
        "H": (44, 50, "sparse"),
    }
    for name, (cv, mv, mode) in prof.items():
        for bar in sec_bars(name):
            ch = chord_at(bar)
            root, upper = PIANO_VOICE[ch]
            t0 = bar_t(bar)

            # sustain pedal: down just after the chord lands, up just before
            # the next chord so voicings ring but do not smear together
            t.cc(t0 - 0.030, 64, 0)
            t.cc(t0 + 0.045, 64, 127)

            # left hand root, softly rolled into the chord
            t.n(t0, root, cv + accent(0) - 4, BAR * 0.95)
            if mode == "pad":
                t.n(t0 + 0.012, root + 7, cv - 8, BAR * 0.9)

            # right hand voicing, rolled like a real hand (not blocked)
            roll = random.uniform(0.012, 0.026)
            for i, nn in enumerate(upper):
                t.n(t0 + 0.02 + i * roll, nn, cv + accent(0) - i * 3,
                    BAR * random.uniform(0.72, 0.95), jitter=0.004)

            if mode == "pad" and random.random() < 0.6:
                # mid-bar re-voicing on beat 3
                t.n(bar_t(bar, 2.0), upper[random.randint(0, 2)],
                    cv + accent(2) - 6, BAR * 0.45)

            # falling melodic figure
            if mv:
                for (beat, note) in PIANO_FIG[bar % 4]:
                    if mode == "sparse" and beat not in (0.0, 2.0):
                        continue
                    t.n(bar_t(bar, beat), note, mv + accent(beat),
                        SPB * random.uniform(1.1, 1.9))

    # closing gesture: the track ends on solo piano, last Em let ring
    last = TOTAL_BARS - 1
    t.cc(bar_t(last) + 0.05, 64, 127)
    t.n(bar_t(last, 2.0), 40, 34, 6.0)
    t.n(bar_t(last, 2.0) + 0.03, 52, 30, 5.8)
    t.n(bar_t(last, 2.0) + 0.07, 59, 28, 5.6)
    t.n(bar_t(last, 2.0) + 0.11, 64, 26, 5.4)
    t.cc(bar_t(TOTAL_BARS) + 4.5, 64, 0)
    return t


# ==========================================================================
# ACOUSTIC STEEL-STRING GUITAR  (FreePats FS Seagull) - the main hook
# ==========================================================================
AG_VOICE = {
    "Em": [40, 47, 52, 55, 59, 64],
    "C":  [48, 52, 55, 60, 64, 67],
    "G":  [43, 47, 50, 55, 59, 67],
    "D":  [50, 57, 62, 66, 69, 74],
}
# The hook: two melody notes per bar (beats 1 and 3) over the 4-bar loop
AG_HOOK = {0: (59, 64), 1: (67, 64), 2: (62, 67), 3: (66, 69)}
AG_HOOK_ALT = {0: (64, 67), 1: (72, 67), 2: (67, 71), 3: (69, 74)}

# fingerpicked 16th grid: (16th slot, voicing index)
AG_PATTERN = [(0, 0), (2, 3), (6, 2), (7, 4), (8, 1), (10, 3), (14, 2), (15, 4)]


def build_acoustic():
    t = Track("acoustic_guitar", jitter=0.007, len_var=0.18)
    # present from 0:22 (bar 8) to the end, minus the F breakdown
    plan = {"B": 78, "C": 92, "D": 84, "E": 100, "G": 104, "H": 74}
    for name, base in plan.items():
        for bar in sec_bars(name, OUTRO_END["acoustic_guitar"]
                            if name == "H" else None):
            ch = chord_at(bar)
            voice = AG_VOICE[ch]
            t0 = bar_t(bar)
            sixteenth = SPB / 4.0
            phrase_hi = (bar // 4) % 2 == 1
            hook = (AG_HOOK_ALT if phrase_hi else AG_HOOK)[bar % 4]

            for (slot, idx) in AG_PATTERN:
                beat = slot / 4.0
                v = base + accent(beat) - (4 if slot in (7, 15) else 0)
                t.n(t0 + slot * sixteenth, voice[idx], v,
                    SPB * random.uniform(0.75, 1.5))

            # the hook itself, on beats 1 and 3, played louder than the picking
            for k, beat in enumerate((0.0, 2.0)):
                t.n(bar_t(bar, beat) + random.uniform(0.002, 0.010),
                    hook[k], base + 14 + accent(beat),
                    SPB * random.uniform(1.6, 2.4))

            # phrase-end strum instead of a picked bar
            if bar % 8 == 7 and name in ("C", "E", "G"):
                t.strum(bar_t(bar, 3.0), voice[:5], base + 6, SPB * 1.6,
                        down=(bar % 16 == 7), accent_top=6)

    # outro: acoustic thins out then stops before the final piano
    return t


# ==========================================================================
# DRUMS  (AVL Black Pearl, 5 velocity layers per zone)
# ==========================================================================
K, SD, SD_EDGE, RIM, HH, HH_OPEN = 36, 38, 40, 37, 42, 46
TOM_HI, TOM_MID, TOM_LO = 47, 45, 41
CRASH1, CRASH2, RIDE = 49, 57, 51


class DrumKit:
    """Routes hits to two stems so the kick can keep its sub while the rest of
    the kit is high-passed. Velocity-layer alternation is tracked per zone, so
    splitting does not weaken the no-identical-repeat guarantee."""

    def __init__(self):
        self.kick = Track("drums_kick")
        self.kit = Track("drums_kit")

    def drum(self, t, key, vel, dur=0.22, jitter=0.008):
        target = self.kick if key == K else self.kit
        target.drum(t, key, vel, dur, jitter)


def build_drums():
    t = DrumKit()
    # Section B: soft rimshot only, no kit
    for bar in sec_bars("B"):
        t0 = bar_t(bar)
        for beat in (1.0, 3.0):
            t.drum(t0 + beat * SPB, RIM, 46 + accent(beat))
        if bar % 4 == 3:
            t.drum(bar_t(bar, 3.5), RIM, 34)

    def full_bar(bar, kick_v, snare_v, hats, ghost=True, ride=False):
        t0 = bar_t(bar)
        # deep kick: beat 1 and the "and of 3"; never busy
        t.drum(t0, K, kick_v + 6, 0.5)
        t.drum(bar_t(bar, 2.5), K, kick_v - 4, 0.5)
        if bar % 4 in (1, 3):
            t.drum(bar_t(bar, 3.75), K, kick_v - 12, 0.4)
        # hard backbeat, alternating centre / edge zone so no two are alike
        for i, beat in enumerate((1.0, 3.0)):
            zone = SD if (bar + i) % 2 == 0 else SD_EDGE
            t.drum(bar_t(bar, beat), zone, snare_v, 0.45)
        if ghost:
            for beat in (1.75, 2.25, 3.5):
                if random.random() < 0.42:
                    t.drum(bar_t(bar, beat), SD_EDGE, 22, 0.2)
        if hats:
            for beat in (0.0, 1.0, 2.0, 3.0):
                key = HH_OPEN if (beat == 3.0 and bar % 4 == 3) else HH
                t.drum(bar_t(bar, beat) + 0.005, key,
                       50 + accent(beat), 0.3)
        if ride:
            for beat in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
                t.drum(bar_t(bar, beat), RIDE, 54 + accent(beat), 0.4)

    def fill(bar):
        """Tom fill over the last two beats."""
        toms = [TOM_HI, TOM_HI, TOM_MID, TOM_MID, TOM_LO, TOM_LO]
        starts = [3.0, 3.25, 3.5, 3.75]
        random.shuffle(toms)
        for i, beat in enumerate(starts):
            t.drum(bar_t(bar, beat), toms[i], 78 + i * 6, 0.35)

    def crash(bar, key=CRASH1, v=104):
        t.drum(bar_t(bar), key, v, 2.2)

    # C: full drums drop in, first big lift
    crash(24, CRASH1, 106)
    for bar in sec_bars("C"):
        full_bar(bar, 96, 100, hats=True)
        if bar == 31:
            fill(bar)

    # D: pulled back - kick and rim, no backbeat snare for the first half
    for bar in sec_bars("D"):
        t0 = bar_t(bar)
        if bar < 40:
            t.drum(t0, K, 78, 0.5)
            for beat in (1.0, 3.0):
                t.drum(bar_t(bar, beat), RIM, 48 + accent(beat))
        else:
            full_bar(bar, 88, 92, hats=True, ghost=True)
        if bar == 47:
            fill(bar)

    # E: peak
    crash(48, CRASH2, 112)
    for bar in sec_bars("E"):
        full_bar(bar, 106, 110, hats=False, ride=True)
        if bar == 51:
            crash(bar, CRASH1, 92)
        if bar == 57:
            fill(bar)

    # F: breakdown - drums cut out entirely (one crash to mark the drop)
    crash(58, CRASH2, 88)

    # G: final section, everything in, crashes
    crash(65, CRASH1, 112)
    for bar in sec_bars("G"):
        full_bar(bar, 108, 112, hats=(bar % 8 < 4), ride=(bar % 8 >= 4))
        if bar in (69, 73, 77):
            crash(bar, CRASH2 if bar % 8 == 5 else CRASH1, 100)
        if bar == 79:
            fill(bar)

    # H: outro - instruments fall away one by one; drums go first
    crash(80, CRASH1, 96)
    for bar in range(80, 83):
        t.drum(bar_t(bar), K, 84, 0.5)
        t.drum(bar_t(bar, 1.0), SD if bar % 2 else SD_EDGE, 88, 0.45)
        t.drum(bar_t(bar, 3.0), SD_EDGE if bar % 2 else SD, 84, 0.45)
    t.drum(bar_t(83), K, 70, 0.6)
    return [t.kick, t.kit]


# ==========================================================================
# BASS  (Karoryfer Black & Blue Basses - real electric basses)
#   sub octave  = babyblue solidbody played with a pick
#   doubling    = darkblack hollowbody played with the fingers
# ==========================================================================
BASS_ROOT = {"Em": 28, "C": 36, "G": 31, "D": 38}


def bass_line(bar):
    """(beat, semitone offset from root) - locked to the kick."""
    ch = chord_at(bar)
    pat = [(0.0, 0), (2.5, 0)]
    if bar % 4 in (1, 3):
        pat.append((3.75, 7 if ch in ("Em", "G") else 5))
    if bar % 8 == 7:
        pat.append((3.0, 12))
    return pat


def build_bass(name, octave, base_vel, sections):
    t = Track(name, jitter=0.006, len_var=0.12)
    for sname in sections:
        for bar in sec_bars(sname, OUTRO_END.get(name) if sname == "H"
                            else None):
            root = BASS_ROOT[chord_at(bar)] + octave
            for (beat, off) in bass_line(bar):
                dur = SPB * (2.4 if beat == 0.0 else 1.2)
                t.n(bar_t(bar, beat), root + off,
                    base_vel + accent(beat), dur)
    return t


# ==========================================================================
# STRINGS  (Virtual Playing Orchestra - Sonatina / No Budget Orchestra)
#   long sustains, overlapping so they breathe rather than start in lockstep
# ==========================================================================
STR_VIOLIN = {"Em": [67, 71], "C": [64, 72], "G": [62, 67], "D": [66, 69]}
STR_CELLO = {"Em": [40, 47], "C": [36, 43], "G": [43, 50], "D": [38, 45]}


def build_strings(name, voicing, base_vel, sections, swell_from=48):
    t = Track(name, jitter=0.022, len_var=0.08)
    for sname in sections:
        for bar in sec_bars(sname, OUTRO_END.get(name) if sname == "H"
                            else None):
            ch = chord_at(bar)
            notes = voicing[ch]
            # swell: sections in the second half push harder
            v = base_vel + (14 if bar >= swell_from else 0)
            v += int(6 * ((bar % 4) / 3.0))
            for i, nn in enumerate(notes):
                # players do not enter together; each voice leans in
                lead = random.uniform(-0.045, 0.055) + i * random.uniform(0.01, 0.04)
                # overlap into the next bar so the pad never gaps
                dur = BAR * random.uniform(1.06, 1.22)
                t.n(bar_t(bar) + lead, nn, v - i * 4, dur, jitter=0.012)
    return t


# ==========================================================================
# ELECTRIC GUITAR  (Karoryfer Black & Green Guitars)
#   green Gretsch  = clean arpeggios mid-track
#   black Hofner   = power chords at the two peaks only
# ==========================================================================
EG_ARP = {
    "Em": [52, 59, 64, 67],
    "C":  [48, 55, 60, 64],
    "G":  [43, 50, 55, 62],
    "D":  [50, 57, 62, 66],
}
EG_POWER = {"Em": [40, 47, 52], "C": [48, 55, 60],
            "G": [43, 50, 55], "D": [50, 57, 62]}


def build_electric_clean():
    t = Track("electric_clean", jitter=0.008, len_var=0.20)
    # clean arpeggios mid-track (section D) and again under the final section
    for sname, base in (("D", 72), ("G", 78)):
        for bar in sec_bars(sname):
            arp = EG_ARP[chord_at(bar)]
            order = [0, 1, 2, 3, 2, 1, 3, 2]
            for i, idx in enumerate(order):
                beat = i * 0.5
                t.n(bar_t(bar, beat), arp[idx], base + accent(beat),
                    SPB * random.uniform(1.2, 2.2))
    return t


def build_electric_power(name="electric_power"):
    """Called twice to double-track the peaks: the same part performed again
    with independent timing/velocity, rendered through a different real
    guitar and panned opposite. That is how double-tracking actually works -
    not a copy of one take."""
    t = Track(name, jitter=0.006, len_var=0.10)
    # power chords only at the two peaks: E (2:12-2:40) and G (2:59-3:40)
    for sname, base in (("E", 96), ("G", 100)):
        for bar in sec_bars(sname):
            ch = EG_POWER[chord_at(bar)]
            # downstroke on 1, upstroke pickup into 3, downstroke on 3
            t.strum(bar_t(bar), ch, base + 6, BAR * 0.55, down=True,
                    spread=(0.010, 0.018), accent_top=4)
            t.strum(bar_t(bar, 1.75), ch, base - 16, SPB * 0.5, down=False,
                    spread=(0.008, 0.014))
            t.strum(bar_t(bar, 2.0), ch, base, BAR * 0.42, down=True,
                    spread=(0.010, 0.020), accent_top=4)
            if bar % 4 == 3:
                t.strum(bar_t(bar, 3.5), ch, base - 8, SPB * 0.6, down=False,
                        spread=(0.008, 0.016))
    return t


# ==========================================================================
if __name__ == "__main__":
    print(f"Composing: {TOTAL_BARS} bars @ {BPM} BPM = "
          f"{TOTAL_BARS * BAR:.3f}s ({TOTAL_BARS * BAR / 60:.2f} min)")
    for k, (a, b) in SECTIONS.items():
        print(f"  {k}  bars {a:>3}-{b - 1:<3}  "
              f"{bar_t(a) // 60:.0f}:{bar_t(a) % 60:05.2f} -> "
              f"{bar_t(b) // 60:.0f}:{bar_t(b) % 60:05.2f}")
    print()

    tracks = [
        build_piano(),
        build_acoustic(),
        *build_drums(),
        build_bass("bass_sub", 0, 88, ["B", "C", "D", "E", "G", "H"]),
        build_bass("bass_electric", 12, 84, ["C", "D", "E", "G"]),
        build_strings("strings_violin", STR_VIOLIN, 62,
                      ["C", "E", "F", "G", "H"]),
        build_strings("strings_cello", STR_CELLO, 66,
                      ["C", "D", "E", "F", "G", "H"]),
        build_electric_clean(),
        build_electric_power("electric_power"),
        build_electric_power("electric_power2"),
    ]
    for tr in tracks:
        tr.save()
    print(f"\nMIDI written to {OUT}")
