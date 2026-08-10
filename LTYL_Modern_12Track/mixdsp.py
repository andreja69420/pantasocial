"""Mix and master stage: per-track Pedalboard chains, mathematical sidechain,
bus summing, master processing and loudness metering.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from pedalboard import (Pedalboard, Chorus, Clipping, Compressor, Delay, Distortion,
                        Gain, HighpassFilter, HighShelfFilter, Limiter, LowpassFilter,
                        PeakFilter, Phaser, Reverb)

from arrange import BEAT      # single source of tempo
from synth import SR


# ------------------------------------------------------------------- helpers

def db(x: float) -> float:
    return 10.0 ** (x / 20.0)


def pan(mono: np.ndarray, position: float = 0.0) -> np.ndarray:
    """Equal-power pan. position: -1 hard left .. 0 centre .. +1 hard right."""
    theta = (position + 1.0) * np.pi / 4.0
    return np.stack([mono * np.cos(theta) * np.sqrt(2) / 1.0,
                     mono * np.sin(theta) * np.sqrt(2) / 1.0]) * 0.7071


def run(board: Pedalboard, audio: np.ndarray) -> np.ndarray:
    out = board(audio.astype(np.float32), SR, reset=True)
    return np.asarray(out, dtype=np.float64)


def widen(st: np.ndarray, amount: float) -> np.ndarray:
    mid = (st[0] + st[1]) * 0.5
    side = (st[0] - st[1]) * 0.5 * amount
    return np.stack([mid + side, mid - side])


def sidechain_env(n: int, trigger_times, duck_db=-5.0, attack_ms=2.0,
                  release_ms=80.0) -> np.ndarray:
    """Gain curve that dips to `duck_db` on every trigger and recovers
    exponentially. Overlapping ducks combine by minimum, so a fast kick
    pattern never lets the 808 fully recover before the next hit."""
    g = np.ones(n)
    floor = db(duck_db)
    a = max(1, int(attack_ms / 1000 * SR))
    r = max(1, int(release_ms / 1000 * SR))
    # Power-curve recovery rather than a plain exponential: it holds the 808
    # near the floor through the kick transient and reaches unity exactly at
    # `release_ms`, which is the audible pump an exponential tail smears out.
    rel = floor + (1.0 - floor) * (np.arange(r) / r) ** 1.7
    shape = np.concatenate([np.linspace(1.0, floor, a, endpoint=False), rel])
    for t in trigger_times:
        i = int(round(t * SR))
        if i >= n:
            continue
        m = min(len(shape), n - i)
        g[i:i + m] = np.minimum(g[i:i + m], shape[:m])
    return g


def lufs_integrated(stereo: np.ndarray) -> float:
    """ITU-R BS.1770-4 gated integrated loudness."""
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    k = np.stack([lfilter(b2, a2, lfilter(b1, a1, ch)) for ch in stereo])

    block = int(0.4 * SR)
    hop = int(0.1 * SR)
    starts = range(0, max(1, k.shape[1] - block), hop)
    power = np.array([np.mean(k[:, s:s + block] ** 2, axis=1).sum() for s in starts])
    loud = -0.691 + 10 * np.log10(np.maximum(power, 1e-12))

    keep = loud > -70.0
    if not keep.any():
        return -np.inf
    rel = -0.691 + 10 * np.log10(power[keep].mean()) - 10.0
    keep &= loud > rel
    return -0.691 + 10 * np.log10(power[keep].mean()) if keep.any() else rel


# ------------------------------------------------------------ track chains

TRACK_GAIN_DB = {
    "T1": -11.0, "T2": -3.0, "T3": -13.0, "T4": -12.0, "T5": -12.0, "T6": -5.5,
    "T7": -6.0, "T8": -11.5, "T9": -14.5, "T10": -9.0, "T11": -11.0, "T12": -19.0,
    "T13": -18.0,
}


def process_tracks(tracks: dict[str, np.ndarray], kick_times) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    # T1 electric guitar. The pickup resonance at 2.7 kHz is what makes it read
    # as electric, but it lands squarely where the vocal needs presence, and
    # the long solid-body sustain piles energy up there rather than decaying
    # out of the way. So it is notched at 2.4 kHz and capped lower than the
    # cabinet already caps it — exactly what a mix engineer does to an electric
    # sitting under a rap vocal.
    out["T1"] = run(Pedalboard([
        HighpassFilter(100), LowpassFilter(2800),
        PeakFilter(cutoff_frequency_hz=2400, gain_db=-3.5, q=0.9),
        Chorus(rate_hz=0.45, depth=0.20, centre_delay_ms=6.5, feedback=0.08, mix=0.28),
        Reverb(room_size=0.58, damping=0.62, wet_level=0.21, dry_level=0.86, width=0.9),
    ]), pan(tracks["T1"], -0.22))

    # T2 synth chord — thinned out of the low end, drowned in a big hall
    out["T2"] = run(Pedalboard([
        HighpassFilter(150), HighpassFilter(150), LowpassFilter(6000),
        Reverb(room_size=0.92, damping=0.28, wet_level=0.40, dry_level=0.60, width=1.0),
    ]), pan(tracks["T2"], 0.12))

    # T3 pad — decorrelated per channel, then M/S widened
    padm = tracks["T3"]
    l = run(Pedalboard([Chorus(rate_hz=0.31, depth=0.35, centre_delay_ms=9.0, mix=0.5)]),
            np.stack([padm, padm]))[0]
    r = run(Pedalboard([Chorus(rate_hz=0.53, depth=0.30, centre_delay_ms=14.0, mix=0.5)]),
            np.stack([padm, padm]))[1]
    out["T3"] = run(Pedalboard([
        HighpassFilter(120), LowpassFilter(3000), LowpassFilter(3000),
        Reverb(room_size=0.80, damping=0.50, wet_level=0.30, dry_level=0.75, width=1.0),
    ]), widen(np.stack([l, r]), 1.7))

    # T4 reverse swell — heavy verb so it smears into the downbeat
    out["T4"] = run(Pedalboard([
        HighpassFilter(180), LowpassFilter(6500),
        Reverb(room_size=0.95, damping=0.20, wet_level=0.62, dry_level=0.45, width=1.0),
    ]), pan(tracks["T4"], 0.0))

    # T5 high pluck — stereo delay: 1/4 left, 1/8 right
    pl = tracks["T5"]
    dl = run(Pedalboard([Delay(delay_seconds=BEAT, feedback=0.34, mix=0.42)]),
             np.stack([pl, pl]))[0]
    dr = run(Pedalboard([Delay(delay_seconds=BEAT / 2, feedback=0.28, mix=0.36)]),
             np.stack([pl, pl]))[1]
    out["T5"] = run(Pedalboard([
        HighpassFilter(300),
        Reverb(room_size=0.78, damping=0.45, wet_level=0.34, dry_level=0.72, width=1.0),
    ]), np.stack([dl, dr]))

    # T6 kick — soft-clipped for trap weight, dead centre
    out["T6"] = run(Pedalboard([
        Distortion(drive_db=2.0), Clipping(threshold_db=-2.0),
        PeakFilter(cutoff_frequency_hz=62, gain_db=2.0, q=1.1),
        HighpassFilter(28),
    ]), pan(tracks["T6"], 0.0))

    # T7 snare/rim — BONE DRY, dead centre. No reverb, no delay, no width.
    out["T7"] = run(Pedalboard([
        HighpassFilter(180), PeakFilter(cutoff_frequency_hz=2600, gain_db=2.5, q=0.9),
    ]), pan(tracks["T7"], 0.0))

    # T8 hats — phaser for liquid movement
    out["T8"] = run(Pedalboard([
        HighpassFilter(600),
        Phaser(rate_hz=0.35, depth=0.6, centre_frequency_hz=2200, feedback=0.15, mix=0.15),
        HighShelfFilter(cutoff_frequency_hz=9000, gain_db=2.5, q=0.7),
    ]), pan(tracks["T8"], -0.10))

    # T9 perc — 15% right
    out["T9"] = run(Pedalboard([
        HighpassFilter(300),
        Reverb(room_size=0.35, damping=0.7, wet_level=0.10, dry_level=0.92, width=0.6),
    ]), pan(tracks["T9"], 0.15))

    # T10 impact — long tail under the chorus downbeat
    out["T10"] = run(Pedalboard([
        HighpassFilter(24),
        Reverb(room_size=0.97, damping=0.35, wet_level=0.42, dry_level=0.85, width=1.0),
    ]), pan(tracks["T10"], 0.0))

    # T11 808 — harmonics, +100 Hz shelf, then mathematical sidechain ducking
    # The lowpass sits at 700 Hz, not down at the fundamental: the point of the
    # distortion is the harmonic ladder that makes a 49 Hz root audible on a
    # phone speaker, and filtering at 180 Hz would throw that away again.
    # Q is deliberately broad (0.45): the roots span 49-87 Hz, and a tight bell
    # at 100 Hz would lift only the Eb2/F2 bars, making the bassline swing ~3 dB
    # between chords.
    sub = run(Pedalboard([
        Distortion(drive_db=5.0),
        PeakFilter(cutoff_frequency_hz=100, gain_db=3.5, q=0.45),
        LowpassFilter(700),
        HighpassFilter(26), HighpassFilter(26),
    ]), pan(tracks["T11"], 0.0))
    out["T11"] = sub * sidechain_env(sub.shape[1], kick_times,
                                     duck_db=-5.0, attack_ms=2.0, release_ms=80.0)

    # T12 vocal textures — 100% wet, heavily filtered, pure atmosphere
    wet = run(Pedalboard([
        Reverb(room_size=0.95, damping=0.30, wet_level=1.0, dry_level=0.0, width=1.0),
        LowpassFilter(2000), LowpassFilter(2000), HighpassFilter(220),
    ]), pan(tracks["T12"], -0.05))
    out["T12"] = wet

    # T13 grand piano stabs — no distortion and no chorus (chorus detunes the
    # unisons and turns a Steinway into a honky-tonk). Highpassed hard at 220 Hz
    # twice: the voicings already sit at F4 and above, and this guarantees no
    # low-end weight or sustain creeps under the 808. Reverb kept short so the
    # stabs stay tight rather than washing into each other.
    out["T13"] = run(Pedalboard([
        HighpassFilter(220), HighpassFilter(220),
        LowpassFilter(6500),
        Reverb(room_size=0.55, damping=0.55, wet_level=0.16, dry_level=0.92, width=0.95),
    ]), pan(tracks["T13"], 0.0))

    for k in out:
        out[k] = out[k] * db(TRACK_GAIN_DB[k])
    return out


# ------------------------------------------------------------- bus + master

BED = ("T1", "T2", "T3", "T4", "T5", "T12", "T13")
DRUMS = ("T6", "T7", "T8", "T9", "T10")


def sum_buses(proc: dict[str, np.ndarray]) -> np.ndarray:
    n = max(a.shape[1] for a in proc.values())

    def acc(keys):
        b = np.zeros((2, n))
        for k in keys:
            a = proc[k]
            b[:, :a.shape[1]] += a
        return b

    bed = acc(BED)
    # Carve the 1-5 kHz pocket out of the sustained beds so a raw, un-tuned
    # vocal has somewhere to sit without fighting the music.
    bed = run(Pedalboard([
        PeakFilter(cutoff_frequency_hz=1600, gain_db=-3.0, q=0.9),
        PeakFilter(cutoff_frequency_hz=3200, gain_db=-3.5, q=0.8),
    ]), bed)

    drums = acc(DRUMS)
    bass = acc(("T11",))
    return bed + drums + bass


def master(mix: np.ndarray, target_peak_db: float = -6.0) -> np.ndarray:
    # Trim into the bus so the limiter only ever catches peaks instead of
    # flattening the beat — a vocal needs the transients left intact.
    peak_in = np.max(np.abs(mix))
    out = run(Pedalboard([
        Gain(gain_db=-20 * np.log10(peak_in / db(-4.0)) if peak_in > 0 else 0.0),
        HighpassFilter(20), HighpassFilter(20),
        LowpassFilter(19000), LowpassFilter(19000),
        Compressor(threshold_db=-18.0, ratio=1.5, attack_ms=30.0, release_ms=100.0),
        Gain(gain_db=2.5),
        Limiter(threshold_db=-1.5, release_ms=120.0),
    ]), mix)

    out = np.tanh(out * 1.02) / np.tanh(1.02)          # final soft clip

    f = int(1.2 * SR)                                  # tail fade, no hard cut
    out[:, -f:] *= np.linspace(1.0, 0.0, f) ** 1.6
    peak = np.max(np.abs(out))
    return out * (db(target_peak_db) / peak) if peak > 0 else out
