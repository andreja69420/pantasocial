"""
Compose the instrumental for "Volis me takvu" - E minor, 4/4, 87 BPM.

Song form, 88 bars = 242.759 s:

    intro    bars  0-3    0:00.0 - 0:11.0   piano alone
    HOOK 1   bars  4-11   0:11.0 - 0:33.1   4 sung lines, 2 bars each + tag
    VERSE 1  bars 12-27   0:33.1 - 1:17.2   16 lines, 1 bar each
    HOOK 2   bars 28-35   1:17.2 - 1:39.3
    VERSE 2  bars 36-51   1:39.3 - 2:23.5
    HOOK 3   bars 52-59   2:23.5 - 2:45.5
    VERSE 3  bars 60-75   2:45.5 - 3:29.7
    HOOK 4   bars 76-83   3:29.7 - 3:51.7
    outro    bars 84-87   3:51.7 - 4:02.8   falls away to solo piano

The arrangement leaves room for a lead vocal: verses stay rhythmic and drop
the competing melodic lines, hooks lift underneath the sung part, and the
track builds hook by hook.

Writes one humanized MIDI file per instrument into output/midi/.
Every part is written for a real multi-sampled library; nothing is synthesized.
"""
import os
import random

import mido

random.seed(20260729)

# Matched to the reference: 87 BPM, G minor. The chord loop is written in E
# minor and transposed up 3 semitones at note-emit time, so Em-C-G-D becomes
# Gm-Eb-Bb-F - the same shape the reference uses.
TRANSPOSE = 3

BPM = 87
SPB = 60.0 / BPM              # 0.689655 s per beat
BAR = 4 * SPB                 # 2.758621 s per bar
TPB = 480                     # ticks per beat
TEMPO = mido.bpm2tempo(BPM)
TOTAL_BARS = 100
TAIL = 7.0                    # let the final piano and reverb tails decay

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output", "midi")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
# Harmony: Em - C - G - D, one bar per chord, looping throughout. Every
# section length is a multiple of 4 bars, so each one starts on Em.
# --------------------------------------------------------------------------
CH = ["Em", "C", "G", "D"]
FINAL_CADENCE = 98  # the last two bars hold the tonic instead of the loop


def loop_pos(bar):
    """Position in the 4-bar chord loop. Bar 0 is an empty lead-in so the
    singer can take the pickup, so the loop starts on bar 1 - which puts the
    piano's first audible chord on the tonic, as in the reference."""
    return (bar - 1) % 4


def chord_at(bar):
    if bar >= FINAL_CADENCE:
        return "Em"
    return CH[loop_pos(bar)]


# name -> (first bar, end bar, kind, intensity 0..1)
# Section lengths taken from the reference: 16-bar verses, 12-bar hooks
# (4 sung lines at 2 bars each, then the tag twice), a sparse opening hook
# standing in for an intro, and a 4-bar break before the last hook.
SECTIONS = {
    "hook1":  (0, 9, "hook_soft", 0.35),
    "verse1": (9, 25, "verse", 0.55),
    "hook2":  (25, 37, "hook", 0.78),
    "verse2": (37, 53, "verse", 0.66),
    "hook3":  (53, 65, "hook", 0.90),
    "verse3": (65, 81, "verse", 0.78),
    "break":  (81, 85, "break", 0.30),
    "hook4":  (85, 97, "hook", 1.00),
    "outro":  (97, 100, "outro", 0.20),
}

# Outro: who stops when, so the track thins out one instrument at a time.
OUTRO_END = {
    "drums": 98, "bass_sub": 98,
    "acoustic_guitar": 99, "strings_violin": 99, "strings_cello": 100,
}


def sec_bars(name, until=None):
    a, b, _, _ = SECTIONS[name]
    return range(a, min(b, until) if until is not None else b)


def level(name):
    return SECTIONS[name][3]


def bar_t(bar, beat=0.0):
    return bar * BAR + beat * SPB


# --------------------------------------------------------------------------
# Humanizing track writer
# --------------------------------------------------------------------------
class Track:
    """Collects notes/CCs and writes a humanized MIDI file.

    Humanization lives here rather than at note entry so every part gets the
    same treatment: velocity jitter with no two consecutive velocities equal,
    timing jitter, and note-length variation.
    """

    def __init__(self, name, jitter=0.008, len_var=0.10):
        self.name = name
        self.jitter = jitter
        self.len_var = len_var
        self.notes = []   # (t, note, vel, dur)
        self.ccs = []     # (t, cc, val)
        self._last_vel = None
        self._last_bucket = {}

    def n(self, t, note, vel, dur, jitter=None):
        if jitter is None:
            jitter = self.jitter
        vel = self._vary_vel(vel)
        t = t + random.uniform(-jitter, jitter)
        dur = dur * random.uniform(1.0 - self.len_var, 1.0 + self.len_var)
        self.notes.append((max(0.0, t), int(note) + TRANSPOSE, vel,
                           max(0.03, dur)))

    def drum(self, t, key, vel, dur=0.22, jitter=0.008):
        """Drum hit forced into a different velocity layer than the previous
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
        self.notes.append((max(0.0, t + random.uniform(-jitter, jitter)),
                           int(key), vel, dur))

    @staticmethod
    def _bucket(v):
        for i, hi in enumerate((26, 52, 77, 102, 127)):
            if v <= hi:
                return i
        return 4

    @staticmethod
    def _shift_bucket(v, b):
        edges = [(1, 26), (27, 52), (53, 77), (78, 102), (103, 127)]
        cand = [i for i in (b - 1, b + 1) if 0 <= i <= 4]
        lo, hi = edges[random.choice(cand)]
        if b < 4 and v > 100:
            lo, hi = edges[max(0, b - 1)]
        return random.randint(lo, hi)

    def _vary_vel(self, vel):
        v = max(1, min(127, int(round(vel + random.uniform(-12, 12)))))
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

        # Parts are written voice by voice, not in time order, so sweep the
        # final stream once more: identical back-to-back velocities are the
        # single biggest giveaway that a performance was sequenced.
        prev = None
        for (_, k, msg) in ev:
            if k != 1:
                continue
            if msg.velocity == prev:
                msg.velocity = max(1, min(127, msg.velocity + random.choice(
                    (-3, -2, -1, 1, 2, 3))))
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
              f"{len(self.ccs):>4} cc")
        return path


def accent(beat):
    """Downbeats loudest, backbeats next, offbeats softest."""
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
# The falling figure is the piano's own hook. It only plays where no vocal is
# competing with it: the intro, the outro, and as an answer in the gap at the
# end of each sung hook line.
PIANO_FIG = {
    0: [(0.0, 71), (2.0, 67)],
    1: [(0.0, 64), (1.5, 67), (3.0, 64)],
    2: [(0.0, 62), (2.0, 59)],
    3: [(0.0, 57), (2.0, 66)],
}
ANSWER = [71, 67, 64, 62, 67, 64, 59, 62]


def build_piano():
    t = Track("piano", jitter=0.009, len_var=0.14)
    for name, (a, b, k, lv) in SECTIONS.items():
        for bar in sec_bars(name):
            # the piano comes in a bar late so the vocal opens alone
            if k == "hook_soft" and bar == a:
                continue
            root, upper = PIANO_VOICE[chord_at(bar)]
            t0 = bar_t(bar)
            cv = int(34 + 52 * lv)

            t.cc(t0 - 0.030, 64, 0)
            t.cc(t0 + 0.045, 64, 127)

            if k == "verse":
                # low, sparse and out of the way - the rap owns the midrange
                t.n(t0, root, cv - 4, BAR * 0.9)
                t.n(t0 + 0.014, root + 7, cv - 10, BAR * 0.8)
                if bar % 2 == 0:
                    roll = random.uniform(0.014, 0.028)
                    for i, nn in enumerate(upper[:2]):
                        t.n(t0 + 0.02 + i * roll, nn - 12, cv - 8 - i * 3,
                            BAR * random.uniform(0.6, 0.85), jitter=0.004)
                continue

            # intro, hooks and outro: full voicing, rolled like a real hand
            t.n(t0, root, cv + accent(0) - 4, BAR * 0.95)
            if k in ("hook", "hook_soft", "break"):
                t.n(t0 + 0.012, root + 7, cv - 8, BAR * 0.9)
            roll = random.uniform(0.012, 0.026)
            for i, nn in enumerate(upper):
                t.n(t0 + 0.02 + i * roll, nn, cv + accent(0) - i * 3,
                    BAR * random.uniform(0.72, 0.95), jitter=0.004)

            if k == "outro" or (k == "hook_soft" and bar < a + 4):
                # exposed opening and the ending: the piano's own falling hook
                for (beat, note) in PIANO_FIG[loop_pos(bar)]:
                    t.n(bar_t(bar, beat), note, cv + 12 + accent(beat),
                        SPB * random.uniform(1.1, 1.9))
            elif k in ("hook", "hook_soft") and (bar - a) % 2 == 1:
                idx = ((bar - a) // 2) % len(ANSWER)
                t.n(bar_t(bar, 3.0), ANSWER[idx], cv + 6,
                    SPB * random.uniform(1.2, 1.8))

    # final Em, let ring
    last = TOTAL_BARS - 1
    t.cc(bar_t(last) + 0.05, 64, 127)
    for i, nn in enumerate((40, 52, 59, 64)):
        t.n(bar_t(last, 2.0) + i * 0.037, nn, 34 - i * 2, 6.0 - i * 0.2)
    t.cc(bar_t(TOTAL_BARS) + 4.5, 64, 0)
    return t


# ==========================================================================
# ACOUSTIC STEEL-STRING GUITAR  (FreePats FS Seagull)
# ==========================================================================
AG_VOICE = {
    "Em": [40, 47, 52, 55, 59, 64],
    "C":  [48, 52, 55, 60, 64, 67],
    "G":  [43, 47, 50, 55, 59, 67],
    "D":  [50, 57, 62, 66, 69, 74],
}
AG_HOOK = {0: (59, 64), 1: (67, 64), 2: (62, 67), 3: (66, 69)}
AG_HOOK_ALT = {0: (64, 67), 1: (72, 67), 2: (67, 71), 3: (69, 74)}
AG_PATTERN = [(0, 0), (2, 3), (6, 2), (7, 4), (8, 1), (10, 3), (14, 2), (15, 4)]


def build_acoustic():
    t = Track("acoustic_guitar", jitter=0.007, len_var=0.18)
    for name, (a, b, k, lv) in SECTIONS.items():
        for bar in sec_bars(name, OUTRO_END["acoustic_guitar"]
                            if k == "outro" else None):
            # the opening hook stays bare for its first half
            if k in ("hook_soft", "break"):
                continue
            voice = AG_VOICE[chord_at(bar)]
            t0 = bar_t(bar)
            sixteenth = SPB / 4.0
            base = int(58 + 46 * lv)

            for (slot, idx) in AG_PATTERN:
                beat = slot / 4.0
                # verses stay in the lower voicing so the vocal keeps the top
                if k == "verse" and idx >= 4:
                    idx -= 2
                v = base + accent(beat) - (4 if slot in (7, 15) else 0)
                t.n(t0 + slot * sixteenth, voice[idx], v,
                    SPB * random.uniform(0.75, 1.5))

            # the melodic hook plays under the sung hook, never over the rap
            if k in ("hook", "hook_soft", "outro"):
                hook = (AG_HOOK_ALT if (bar // 4) % 2 else AG_HOOK)[loop_pos(bar)]
                for j, beat in enumerate((0.0, 2.0)):
                    t.n(bar_t(bar, beat) + random.uniform(0.002, 0.010),
                        hook[j], base + 8 + accent(beat),
                        SPB * random.uniform(1.6, 2.4))

            if k == "hook" and bar == b - 1:
                t.strum(bar_t(bar, 3.0), voice[:5], base + 6, SPB * 1.6,
                        down=((bar // 8) % 2 == 0), accent_top=6)
    return t


# ==========================================================================
# DRUMS  (AVL Black Pearl, 5 velocity layers per zone)
# ==========================================================================
K, SD, SD_EDGE, RIM, HH, HH_OPEN = 36, 38, 40, 37, 42, 46
TOM_HI, TOM_MID, TOM_LO = 47, 45, 41
CRASH1, CRASH2, RIDE, SPLASH = 49, 57, 51, 55


class DrumKit:
    """Routes hits to two stems so the kick keeps its sub while the rest of
    the kit is high-passed. Layer alternation is tracked per zone, so the
    split does not weaken the no-identical-repeat guarantee."""

    def __init__(self):
        self.kick = Track("drums_kick")
        self.kit = Track("drums_kit")

    def drum(self, t, key, vel, dur=0.22, jitter=0.008):
        (self.kick if key == K else self.kit).drum(t, key, vel, dur, jitter)


def build_drums():
    d = DrumKit()

    def groove(bar, lv, hats, ride=False, ghost=True):
        kv, sv = int(72 + 40 * lv), int(76 + 38 * lv)
        t0 = bar_t(bar)
        # deep kick, never busy: beat 1 and the "and of 3"
        d.drum(t0, K, kv + 6, 0.5)
        d.drum(bar_t(bar, 2.5), K, kv - 4, 0.5)
        if loop_pos(bar) in (1, 3):
            d.drum(bar_t(bar, 3.75), K, kv - 12, 0.4)
        # hard backbeat, alternating centre / edge so no two are alike
        for i, beat in enumerate((1.0, 3.0)):
            d.drum(bar_t(bar, beat), SD if (bar + i) % 2 == 0 else SD_EDGE,
                   sv, 0.45)
        if ghost:
            for beat in (1.75, 2.25, 3.5):
                if random.random() < 0.40:
                    d.drum(bar_t(bar, beat), SD_EDGE, 22, 0.2)
        if hats:
            for beat in (0.0, 1.0, 2.0, 3.0):
                key = HH_OPEN if (beat == 3.0 and loop_pos(bar) == 3) else HH
                d.drum(bar_t(bar, beat) + 0.005, key,
                       int(42 + 16 * lv) + accent(beat), 0.3)
        if ride:
            for beat in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
                d.drum(bar_t(bar, beat), RIDE,
                       int(46 + 14 * lv) + accent(beat), 0.4)

    def fill(bar):
        toms = [TOM_HI, TOM_HI, TOM_MID, TOM_MID, TOM_LO, TOM_LO]
        random.shuffle(toms)
        for i, beat in enumerate((3.0, 3.25, 3.5, 3.75)):
            d.drum(bar_t(bar, beat), toms[i], 78 + i * 6, 0.35)

    for name, (a, b, k, lv) in SECTIONS.items():
        if k == "hook_soft":
            # solo piano, exactly as the reference: no kit at all
            continue
        if k == "break":
            # drums out, cymbal wash only, then a fill into the last hook
            d.drum(bar_t(a), CRASH2, 74, 3.0)
            d.drum(bar_t(a + 2), SPLASH, 62, 2.0)
            fill(b - 1)
            continue
        if k == "outro":
            d.drum(bar_t(a), CRASH1, 96, 2.4)
            for bar in range(a, OUTRO_END["drums"]):
                d.drum(bar_t(bar), K, 84, 0.5)
                d.drum(bar_t(bar, 1.0), SD if bar % 2 else SD_EDGE, 88, 0.45)
                d.drum(bar_t(bar, 3.0), SD_EDGE if bar % 2 else SD, 84, 0.45)
            continue

        d.drum(bar_t(a), CRASH1 if k == "verse" else CRASH2,
               int(88 + 24 * lv), 2.4)
        for bar in sec_bars(name):
            if k == "hook":
                groove(bar, lv, hats=(bar - a < 6), ride=(bar - a >= 6))
                if bar == a + 6:
                    d.drum(bar_t(bar), CRASH1, int(84 + 18 * lv), 2.0)
            else:
                # verse: steady and heavy, the grid the vocal sits on
                groove(bar, lv, hats=True, ghost=True)
        fill(b - 1)
    return [d.kick, d.kit]


# ==========================================================================
# BASS  (Karoryfer Black & Blue Basses - real electric basses)
#   sub octave = blue solidbody played with a pick
#   doubling   = black hollowbody played with the fingers
# ==========================================================================
BASS_ROOT = {"Em": 28, "C": 36, "G": 31, "D": 38}


def build_bass(name, octave, base_vel, kinds):
    t = Track(name, jitter=0.006, len_var=0.12)
    for sname, (a, b, k, lv) in SECTIONS.items():
        if k not in kinds:
            continue
        for bar in sec_bars(sname, OUTRO_END.get(name) if k == "outro"
                            else None):
            ch = chord_at(bar)
            root = BASS_ROOT[ch] + octave
            pat = [(0.0, 0), (2.5, 0)]          # locked to the kick
            if loop_pos(bar) in (1, 3):
                pat.append((3.75, 7 if ch in ("Em", "G") else 5))
            if bar % 8 == 0:
                pat.append((3.0, 12))
            for (beat, off) in pat:
                t.n(bar_t(bar, beat), root + off,
                    int(base_vel * (0.75 + 0.3 * lv)) + accent(beat),
                    SPB * (2.4 if beat == 0.0 else 1.2))
    return t


# ==========================================================================
# STRINGS  (Virtual Playing Orchestra)
#   long sustains, overlapping so they breathe rather than move in lockstep
# ==========================================================================
STR_VIOLIN = {"Em": [67, 71], "C": [64, 72], "G": [62, 67], "D": [66, 69]}
STR_CELLO = {"Em": [40, 47], "C": [36, 43], "G": [43, 50], "D": [38, 45]}


def build_strings(name, voicing, base_vel, plan):
    """plan maps section name -> level multiplier; absent means tacet."""
    t = Track(name, jitter=0.022, len_var=0.08)
    for sname, mult in plan.items():
        a, b, k, lv = SECTIONS[sname]
        for bar in sec_bars(sname, OUTRO_END.get(name) if k == "outro"
                            else None):
            notes = voicing[chord_at(bar)]
            v = int((base_vel + 16 * lv) * mult) + int(6 * (loop_pos(bar) / 3.0))
            for i, nn in enumerate(notes):
                lead = (random.uniform(-0.045, 0.055)
                        + i * random.uniform(0.01, 0.04))
                t.n(bar_t(bar) + lead, nn, v - i * 4,
                    BAR * random.uniform(1.06, 1.22), jitter=0.012)
    return t


# ==========================================================================
# ELECTRIC GUITAR  (Karoryfer Black & Green Guitars)
# ==========================================================================
EG_ARP = {
    "Em": [52, 59, 64, 67], "C": [48, 55, 60, 64],
    "G":  [43, 50, 55, 62], "D": [50, 57, 62, 66],
}
EG_POWER = {"Em": [40, 47, 52], "C": [48, 55, 60],
            "G": [43, 50, 55], "D": [50, 57, 62]}


def build_electric_clean():
    """Clean arpeggios add motion under the later verses."""
    t = Track("electric_clean", jitter=0.008, len_var=0.20)
    for sname in ("verse2", "verse3"):
        lv = level(sname)
        for bar in sec_bars(sname):
            arp = EG_ARP[chord_at(bar)]
            for i, idx in enumerate([0, 1, 2, 3, 2, 1, 3, 2]):
                beat = i * 0.5
                t.n(bar_t(bar, beat), arp[idx],
                    int(62 + 20 * lv) + accent(beat),
                    SPB * random.uniform(1.2, 2.2))
    return t


def build_electric_power(name="electric_power"):
    """Power chords hold up the last two hooks. Called twice to double-track:
    the same part performed again with independent timing and velocity, then
    rendered through a different real guitar - not one take copied."""
    t = Track(name, jitter=0.006, len_var=0.10)
    for sname in ("hook3", "hook4"):
        lv = level(sname)
        base = int(78 + 24 * lv)
        for bar in sec_bars(sname):
            ch = EG_POWER[chord_at(bar)]
            t.strum(bar_t(bar), ch, base + 6, BAR * 0.55, down=True,
                    spread=(0.010, 0.018), accent_top=4)
            t.strum(bar_t(bar, 1.75), ch, base - 16, SPB * 0.5, down=False,
                    spread=(0.008, 0.014))
            t.strum(bar_t(bar, 2.0), ch, base, BAR * 0.42, down=True,
                    spread=(0.010, 0.020), accent_top=4)
            if loop_pos(bar) == 3:
                t.strum(bar_t(bar, 3.5), ch, base - 8, SPB * 0.6, down=False,
                        spread=(0.008, 0.016))
    return t


# ==========================================================================
if __name__ == "__main__":
    print(f"Composing: {TOTAL_BARS} bars @ {BPM} BPM = "
          f"{TOTAL_BARS * BAR:.3f}s\n")
    for k, (a, b, kd, lv) in SECTIONS.items():
        print(f"  {k:<8} bars {a:>3}-{b - 1:<3} "
              f"{int(bar_t(a) // 60)}:{bar_t(a) % 60:05.2f} -> "
              f"{int(bar_t(b) // 60)}:{bar_t(b) % 60:05.2f}   {kd}")
    print()

    tracks = [
        build_piano(),
        build_acoustic(),
        *build_drums(),
        build_bass("bass_sub", 0, 92,
                   ("hook", "verse", "break", "outro")),
        build_bass("bass_electric", 12, 88,
                   ("hook", "verse")),
        # violins carry the hooks; cellos also underpin the later verses
        build_strings("strings_violin", STR_VIOLIN, 58, {
            "hook2": 0.92, "hook3": 1.0,
            "verse3": 0.62, "break": 1.0, "hook4": 1.0,
            "outro": 0.55}),
        build_strings("strings_cello", STR_CELLO, 62, {
            "hook2": 0.95, "verse2": 0.58,
            "hook3": 1.0, "verse3": 0.70, "break": 1.0,
            "hook4": 1.0, "outro": 0.60}),
        build_electric_clean(),
        build_electric_power("electric_power"),
        build_electric_power("electric_power2"),
    ]
    for tr in tracks:
        tr.save()
    print(f"\nMIDI written to {OUT}")
