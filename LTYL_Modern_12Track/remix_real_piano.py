"""One-off A/B remix: swap the synthesized T13 grand piano for the real
Logic Pro bounce in real_stems/{piano_LH,piano_RH}.flac, calibrated so the
real piano sits at roughly the same level the synthesized one did (measured
in chorus 1, which is piano-solo, so it's a clean apples-to-apples window).

    python remix_real_piano.py

Writes LoveTheWayYouLie_Modern12Track_RealPiano.wav (+ .mp3) alongside the
main synthesized master, without touching it.
"""
from __future__ import annotations

import os

import numpy as np
import soundfile as sf

import arrange
import mixdsp
from synth import SR

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(HERE, "real_stems")
OUT_NAME = "LoveTheWayYouLie_Modern12Track_RealPiano"


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def load_real_piano() -> np.ndarray:
    lh, sr1 = sf.read(os.path.join(REAL_DIR, "piano_LH.flac"), always_2d=True)
    rh, sr2 = sf.read(os.path.join(REAL_DIR, "piano_RH.flac"), always_2d=True)
    assert sr1 == SR and sr2 == SR, f"expected {SR} Hz, got {sr1}/{sr2}"
    n = min(len(lh), len(rh))
    return (lh[:n] + rh[:n]).T          # (2, n), matches the (channels, samples) convention used everywhere else


def main() -> None:
    print("[1/4] Sequencing (synthesized tracks, needed for everything but T13)")
    tracks, kick_times = arrange.build()

    real_piano = load_real_piano()
    print(f"    real piano: {real_piano.shape[1] / SR:.2f}s, "
          f"peak {20*np.log10(np.max(np.abs(real_piano))):.2f} dBFS")

    print("[2/4] Calibrating real-piano level against the synthesized piano")
    proc_synth = mixdsp.process_tracks(tracks, kick_times)
    proc_real_uncalibrated = mixdsp.process_tracks(tracks, kick_times, real_stereo={"T13": real_piano})

    c0, c1 = int(0 * SR), int(arrange.bar_time(8) * SR)   # chorus 1 -- piano solo, nothing else
    target = rms_db(proc_synth["T13"][:, c0:c1])
    current = rms_db(proc_real_uncalibrated["T13"][:, c0:c1])
    correction_db = target - current
    calibrated_gain_db = mixdsp.TRACK_GAIN_DB["T13"] + correction_db
    print(f"    synth T13 chorus-1 RMS   {target:+.2f} dBFS")
    print(f"    real  T13 chorus-1 RMS   {current:+.2f} dBFS (before correction)")
    print(f"    correction               {correction_db:+.2f} dB  "
          f"(T13 gain {mixdsp.TRACK_GAIN_DB['T13']:.1f} -> {calibrated_gain_db:.1f} dB)")

    print("[3/4] Re-processing with the calibrated real piano")
    mixdsp.TRACK_GAIN_DB["T13"] = calibrated_gain_db
    proc = mixdsp.process_tracks(tracks, kick_times, real_stereo={"T13": real_piano})
    mix = mixdsp.sum_buses(proc)
    final = mixdsp.master(mix, target_peak_db=-6.0)

    peak = 20 * np.log10(np.max(np.abs(final)))
    rms = rms_db(final)
    lufs = mixdsp.lufs_integrated(final)
    print(f"    MASTER peak {peak:.3f} dBFS | RMS {rms:.2f} dBFS | {lufs:.2f} LUFS")

    print("[4/4] Export")
    wav_path = os.path.join(HERE, f"{OUT_NAME}.wav")
    sf.write(wav_path, final.T.astype(np.float32), SR, subtype="PCM_24")
    print(f"    wrote {wav_path}")


if __name__ == "__main__":
    main()
