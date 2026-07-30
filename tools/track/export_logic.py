"""Merge the per-instrument MIDI parts into one multi-track file for a DAW.

Writes a Type 1 MIDI: track 0 is a conductor track carrying tempo and time
signature, then one named track per instrument. Dropping it into Logic (or any
DAW) gives every part on its own labelled track, ready to have instruments
assigned.

No program-change messages are written, so the DAW never forces a General MIDI
sound - you pick the instrument per track. The drum parts are merged onto MIDI
channel 10 and keep their GM note numbers (36 kick, 38 snare, 42 closed hat,
41/45/47 toms, 49/57 crashes, 51 ride, 37 side stick), so a drum instrument
maps them without editing.
"""
import os
import sys

import mido

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIDI = os.path.join(ROOT, "output", "midi")
BPM = 87
TPB = 480

# (track name, source files, MIDI channel)
TRACKS = [
    ("Piano",            ["piano"],                       0),
    ("Acoustic Guitar",  ["acoustic_guitar"],             1),
    ("Bass Sub",         ["bass_sub"],                    2),
    ("Bass Electric",    ["bass_electric"],               3),
    ("Strings Violin",   ["strings_violin"],              4),
    ("Strings Cello",    ["strings_cello"],               5),
    ("Electric Clean",   ["electric_clean"],              6),
    ("Electric Power L", ["electric_power"],              7),
    ("Electric Power R", ["electric_power2"],             8),
    ("Drums",            ["drums_kick", "drums_kit"],     9),   # ch 10
    ("Hook Melody",      ["hook_melody_guide"],          10),
]


def read_events(name):
    """Absolute-time messages from one part file, tempo/meta stripped."""
    path = os.path.join(MIDI, f"{name}.mid")
    if not os.path.exists(path):
        return None
    out = []
    t = 0.0
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.is_meta:
            continue
        out.append((t, msg))
    return out


def main():
    mf = mido.MidiFile(type=1, ticks_per_beat=TPB)
    tempo = mido.bpm2tempo(BPM)

    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="Volis me takvu",
                                      time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=4,
                                      denominator=4, time=0))
    conductor.append(mido.MetaMessage("key_signature", key="Gm", time=0))
    mf.tracks.append(conductor)

    total, missing = 0, []
    for label, sources, channel in TRACKS:
        events = []
        for src in sources:
            ev = read_events(src)
            if ev is None:
                missing.append(src)
                continue
            events += ev
        if not events:
            continue
        events.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1))

        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=label, time=0))
        last = 0
        notes = 0
        for (t, msg) in events:
            tick = int(round(mido.second2tick(t, TPB, tempo)))
            msg = msg.copy(channel=channel, time=max(0, tick - last))
            last = max(last, tick)
            track.append(msg)
            if msg.type == "note_on" and msg.velocity > 0:
                notes += 1
        track.append(mido.MetaMessage("end_of_track", time=1))
        mf.tracks.append(track)
        total += notes
        print(f"  {label:<18} ch {channel + 1:<3} {notes:>5} notes")

    out = os.path.join(ROOT, "output", "volis-me-takvu-LOGIC.mid")
    mf.save(out)
    print(f"\n{out}")
    print(f"  {len(mf.tracks) - 1} instrument tracks, {total} notes, "
          f"{BPM} BPM, 4/4, G minor")
    if missing:
        print(f"  note: no file for {missing}")
        sys.exit(1)


if __name__ == "__main__":
    main()
