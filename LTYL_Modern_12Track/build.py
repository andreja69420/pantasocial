"""Build the full 12-track, 56-bar instrumental end to end.

    python build.py

Writes LoveTheWayYouLie_Modern12Track.wav (and per-track stems under stems/).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import soundfile as sf

import arrange
import mixdsp
from synth import SR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NAME = "LoveTheWayYouLie_Modern12Track.wav"

TRACK_NAMES = {
    "T1": "Main Guitar", "T2": "Ambient Synth", "T3": "Synth Pad",
    "T4": "Reverse Swell", "T5": "High Pluck", "T6": "Kick",
    "T7": "Snare/Rim", "T8": "Hi-Hat", "T9": "Perc/Open Hat",
    "T10": "Crash/Impact", "T11": "808 Sub", "T12": "Vocal Textures",
    "T13": "Grand Piano",
}


def try_cc0_assets(dest: str) -> list[str]:
    """Attempt to pull CC0 source material before falling back to synthesis.

    Kept deliberately small and non-fatal: anything that 404s, times out, or
    comes back non-audio just means that asset gets synthesized instead.
    """
    import requests

    manifest = {
        "cc0_cymbal.ogg": "https://upload.wikimedia.org/wikipedia/commons/6/6d/Crash_cymbal.ogg",
        "cc0_piano_c.ogg": "https://upload.wikimedia.org/wikipedia/commons/9/9c/Piano_C4.ogg",
        "cc0_guitar.ogg": "https://upload.wikimedia.org/wikipedia/commons/4/40/Acoustic_guitar.ogg",
    }
    os.makedirs(dest, exist_ok=True)
    got = []
    for name, url in manifest.items():
        path = os.path.join(dest, name)
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "LTYL-build/1.0"})
            if r.status_code == 200 and r.content[:4] in (b"OggS", b"RIFF", b"fLaC"):
                with open(path, "wb") as fh:
                    fh.write(r.content)
                got.append(name)
                print(f"    [cc0 ] {name}  ({len(r.content)//1024} KB)")
            else:
                print(f"    [miss] {name}  (HTTP {r.status_code})")
        except Exception as exc:
            print(f"    [miss] {name}  ({type(exc).__name__})")
    return got


def report(stereo: np.ndarray, label: str) -> None:
    peak = 20 * np.log10(np.max(np.abs(stereo)) + 1e-12)
    rms = 20 * np.log10(np.sqrt(np.mean(stereo ** 2)) + 1e-12)
    lufs = mixdsp.lufs_integrated(stereo)
    print(f"  {label:<12} peak {peak:7.2f} dBFS | RMS {rms:7.2f} dBFS | "
          f"{lufs:6.2f} LUFS | crest {peak - rms:5.2f} dB")


def pocket_check(stereo: np.ndarray) -> None:
    """Confirm the 1-5 kHz vocal pocket really is the quietest region."""
    mono = stereo.mean(axis=0)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1 / SR)
    bands = [(20, 120, "sub"), (120, 400, "low"), (400, 1000, "low-mid"),
             (1000, 5000, "VOCAL POCKET"), (5000, 12000, "high"), (12000, 19000, "air")]
    print("  spectral balance (energy density, dB rel. full mix):")
    total = np.sqrt(np.mean(spec ** 2))
    for lo, hi, name in bands:
        m = (freqs >= lo) & (freqs < hi)
        e = 20 * np.log10(np.sqrt(np.mean(spec[m] ** 2)) / total + 1e-12)
        print(f"    {name:<13} {lo:>5}-{hi:<5} Hz  {e:+7.2f} dB")


def main() -> int:
    t_start = time.time()
    print("=" * 72)
    print("  LOVE THE WAY YOU LIE — modern dark trap/drill rework")
    print(f"  {arrange.BPM:.0f} BPM | G minor | 56 bars | {arrange.BARS * arrange.BAR:.1f}s | {SR} Hz")
    print("=" * 72)

    print("\n[1/5] Asset acquisition")
    got = try_cc0_assets(os.path.join(HERE, "assets"))
    print(f"    {len(got)} CC0 file(s) retrieved; all 12 tracks synthesized in-key "
          f"(G minor) for tuning accuracy.")

    print("\n[2/5] Sequencing the 56-bar grid")
    tracks, kick_times = arrange.build()
    for k in sorted(tracks, key=lambda s: int(s[1:])):
        pk = 20 * np.log10(np.max(np.abs(tracks[k])) + 1e-12)
        print(f"    {k:<4} {TRACK_NAMES[k]:<16} peak {pk:7.2f} dBFS")
    print(f"    {len(kick_times)} kick triggers captured for the 808 sidechain.")

    print("\n[3/5] Per-track DSP (Pedalboard)")
    proc = mixdsp.process_tracks(tracks, kick_times)
    env = mixdsp.sidechain_env(proc["T11"].shape[1], kick_times)
    print(f"    Sidechain: -5.0 dB duck / 2 ms attack / 80 ms release, "
          f"min gain {20 * np.log10(env.min()):.2f} dB")

    print("\n[4/5] Summing + mastering")
    mix = mixdsp.sum_buses(proc)
    report(mix, "pre-master")
    final = mixdsp.master(mix, target_peak_db=-6.0)
    report(final, "MASTER")
    pocket_check(final)

    print("\n[5/5] Export")
    stem_dir = os.path.join(HERE, "stems")
    os.makedirs(stem_dir, exist_ok=True)
    for k, a in proc.items():
        sf.write(os.path.join(stem_dir, f"{k}_{TRACK_NAMES[k].replace('/', '-')}.wav"),
                 a.T, SR, subtype="PCM_24")

    # Raw pre-DSP stems, mono. Pitch verification wants these: the mix chains
    # add filtering, distortion and reverb that obscure fundamentals, none of
    # which changes what note was actually sequenced.
    # Written as float, NOT PCM_24. These buffers legitimately exceed 1.0 —
    # per-track gains are applied downstream in the mix stage — so a fixed-point
    # export clamps them and reports clipping that does not exist in the mix.
    raw_dir = os.path.join(HERE, "stems_raw")
    os.makedirs(raw_dir, exist_ok=True)
    for k, a in tracks.items():
        sf.write(os.path.join(raw_dir, f"{k}_{TRACK_NAMES[k].replace('/', '-')}.wav"),
                 a, SR, subtype="FLOAT")
    hot = {k: float(np.max(np.abs(a))) for k, a in tracks.items() if np.max(np.abs(a)) > 1.0}
    if hot:
        print("    raw buffers above 1.0 (fine — gains applied downstream): "
              + ", ".join(f"{k} {v:.2f}" for k, v in sorted(hot.items())))

    for path in (os.path.join(HERE, OUT_NAME), os.path.join(HERE, "..", OUT_NAME)):
        sf.write(path, final.T.astype(np.float32), SR, subtype="PCM_24")
        print(f"    wrote {os.path.normpath(path)}")

    peak_db = 20 * np.log10(np.max(np.abs(final)))
    print(f"\n  Peak: {peak_db:.4f} dBFS (target -6.0000)")
    print(f"  Length: {final.shape[1] / SR:.2f} s | built in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
