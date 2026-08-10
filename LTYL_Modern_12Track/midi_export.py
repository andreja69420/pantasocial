"""Export the melodic/harmonic instruments as standard MIDI files for Logic
Pro (or any DAW): one .mid per instrument, so each can be dropped onto its
own track and played back with a real/virtual instrument instead of this
project's synthesis.

    python midi_export.py

Writes midi/*.mid. Re-render the audio in Logic, then drop the bounced stems
back into stems_raw/ (or send them over) for remixing/mastering here — this
script only exports the composition, not the sound design.

Requires `mido` (pip install mido); nothing else in this file touches numpy
or scipy, so it runs in well under a second.
"""
from __future__ import annotations

import os

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

import arrange
import synth as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "midi")

# (filename, event-list key, GM program number, channel, display name)
# Program numbers are General MIDI defaults so the files play back sanely
# even before you swap in your own patch: 0 Grand Piano, 25 Steel Guitar,
# 49 Strings Ensemble, 41 Violin, 89 Pad (warm), 33 Fingered Bass.
INSTRUMENTS = [
    ("piano_RH.mid",  "piano_RH", 0,  0, "Grand Piano — right hand (stabs)"),
    ("piano_LH.mid",  "piano_LH", 0,  1, "Grand Piano — left hand (two-hand zones only)"),
    ("guitar.mid",    "guitar",   25, 2, "Electric/Acoustic Guitar"),
    ("pad.mid",       "pad",      89, 3, "Synth Pad"),
    ("strings.mid",   "strings",  49, 4, "String Ensemble"),
    ("violin.mid",    "violin",   41, 5, "Solo Violin"),
    ("bass808.mid",   "bass808",  33, 6, "808 Bass"),
]


def write_midi(path: str, notes: list[tuple[float, float, str, float]],
               bpm: float, program: int, channel: int,
               ticks_per_beat: int = 480) -> int:
    """`notes` is `[(start_sec, dur_sec, note_name, velocity_0to1), ...]`."""
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(Message("program_change", program=program, channel=channel, time=0))

    ticks_per_sec = ticks_per_beat * bpm / 60.0
    events: list[tuple[int, int, int, int]] = []   # (tick, is_note_on, note, velocity)
    for start, dur, name, vel in notes:
        n = S.midi_number(name)
        v = max(1, min(127, round(vel * 127)))
        t_on = round(start * ticks_per_sec)
        t_off = round((start + dur) * ticks_per_sec)
        if t_off <= t_on:
            t_off = t_on + 1
        events.append((t_on, 1, n, v))
        events.append((t_off, 0, n, 0))
    # note_off before note_on at the same tick, so a re-struck pitch doesn't
    # get its fresh note_on immediately cancelled by the previous note's off.
    events.sort(key=lambda e: (e[0], e[1]))

    last_tick = 0
    for tick, is_on, note, vel in events:
        delta = tick - last_tick
        last_tick = tick
        kind = "note_on" if is_on else "note_off"
        track.append(Message(kind, note=note, velocity=vel, time=delta, channel=channel))

    mid.save(path)
    return len(notes)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    events = arrange.schedule_events()

    print("=" * 60)
    print(f"  MIDI export — {arrange.BPM:.0f} BPM, {arrange.BARS} bars, G minor")
    print("=" * 60)
    for fname, key, program, channel, label in INSTRUMENTS:
        notes = events.get(key, [])
        path = os.path.join(OUT_DIR, fname)
        n = write_midi(path, notes, arrange.BPM, program, channel)
        print(f"    {fname:<16} {n:>4} notes   {label}")

    print(f"\n  wrote {len(INSTRUMENTS)} files to {os.path.relpath(OUT_DIR, HERE)}/")
    print("  Import each into its own Logic track, pick an instrument, bounce,")
    print("  and send the audio back for remixing/mastering into the master.")


if __name__ == "__main__":
    main()
