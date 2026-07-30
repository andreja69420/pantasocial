"""Mix and master the rendered stems into the final track."""
import os

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import (Compressor, Gain, HighpassFilter, HighShelfFilter,
                        Limiter, LowShelfFilter, Pedalboard, Reverb)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEMS = os.path.join(ROOT, "output", "stems")
OUTDIR = os.path.join(ROOT, "output")
SR = 48000
BAR = 240.0 / 87.0
LENGTH = 279.0          # 100 bars of music (275.86 s) plus the piano tail
TARGET_LUFS = -14.0

# part -> mixing recipe
#   lufs  : gated loudness target for the stem before the bus (sets balance)
#   pan   : -1 hard left .. +1 hard right
#   width : stereo width multiplier (1.0 = as recorded)
#   hpf   : high-pass corner in Hz (0 = none; kick and bass stay full-range)
#   verb  : (room_size, wet) or None
#   comp  : (threshold_db, ratio) or None
MIX = {
    # forward in the mix
    "acoustic_guitar": dict(lufs=-16.5, pan=-0.18, width=1.15, hpf=95,
                            verb=(0.62, 0.16), comp=(-19, 2.2)),
    "piano":           dict(lufs=-16.8, pan=0.0, width=1.05, hpf=75,
                            verb=(0.68, 0.15), comp=(-22, 1.8)),
    # rhythm section, centred
    "drums_kick":      dict(lufs=-19.5, pan=0.0, width=0.35, hpf=0,
                            verb=None, comp=(-17, 3.5)),
    "drums_kit":       dict(lufs=-19.0, pan=0.0, width=1.0, hpf=110,
                            verb=(0.22, 0.09), comp=(-19, 3.0)),
    "bass_sub":        dict(lufs=-19.8, pan=0.0, width=0.15, hpf=0,
                            verb=None, comp=(-20, 3.2)),
    "bass_electric":   dict(lufs=-23.0, pan=0.0, width=0.4, hpf=0,
                            verb=None, comp=(-20, 2.8)),
    # wide
    "strings_violin":  dict(lufs=-23.5, pan=0.30, width=1.6,
                            hpf=120, verb=(0.86, 0.34), comp=None),
    "strings_cello":   dict(lufs=-22.5, pan=-0.28, width=1.5, hpf=95,
                            verb=(0.86, 0.30), comp=None),
    "electric_clean":  dict(lufs=-25.0, pan=0.45, width=1.3, hpf=120,
                            verb=(0.66, 0.22), comp=(-21, 2.5)),
    "electric_power":  dict(lufs=-23.0, pan=-0.62, width=1.2, hpf=110,
                            verb=(0.60, 0.16), comp=(-20, 2.6)),
    "electric_power2": dict(lufs=-23.2, pan=0.62, width=1.2, hpf=110,
                            verb=(0.60, 0.16), comp=(-20, 2.6)),
}

meter = pyln.Meter(SR)


def peak_limit(x, ceiling_db=-1.0, lookahead_ms=2.0, release_ms=150.0):
    """Transparent look-ahead peak limiter.

    pedalboard's Limiter applies auto-makeup gain (it always pushes up to the
    threshold), which would flatten the master. This only ever attenuates, so
    the dynamic range survives.
    """
    from scipy.ndimage import minimum_filter1d

    ceil = 10.0 ** (ceiling_db / 20.0)
    la = max(1, int(lookahead_ms * 1e-3 * SR))
    env = np.max(np.abs(x), axis=1)
    gain = np.minimum(1.0, ceil / np.maximum(env, 1e-9))
    # look-ahead: start ducking before the peak arrives
    gain = minimum_filter1d(gain, size=2 * la + 1, mode="nearest")
    # one-pole smoothing gives a musical release; taking the minimum against
    # the raw curve guarantees the smoothing can never let a peak through
    from scipy.signal import lfilter

    coeff = np.exp(-1.0 / (release_ms * 1e-3 * SR))
    smooth = lfilter([1.0 - coeff], [1.0, -coeff], gain, zi=None)
    gain = np.minimum(gain, smooth)
    return (x * gain[:, None]).astype(np.float32)


def load(part):
    d, sr = sf.read(os.path.join(STEMS, f"{part}.wav"), dtype="float32")
    assert sr == SR, f"{part}: unexpected sample rate {sr}"
    if d.ndim == 1:
        d = np.stack([d, d], axis=1)
    n = int(LENGTH * SR)
    if len(d) < n:
        d = np.pad(d, ((0, n - len(d)), (0, 0)))
    return d[:n]


def stereo_place(x, pan, width):
    """Constant-power placement with M/S width control."""
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5 * width
    left, right = mid + side, mid - side
    theta = (pan + 1.0) * (np.pi / 4.0)
    return np.stack([left * np.cos(theta), right * np.sin(theta)],
                    axis=1) * np.sqrt(2.0)


def swell_envelope(n):
    """Gentle lift across the song. Section-by-section string dynamics are set
    in the composition, so this only adds a slow overall rise plus a taper
    through the outro."""
    t = np.arange(n) / SR
    env = 0.82 + 0.18 * np.clip((t - 24.8) / 210.0, 0.0, 1.0)   # verse 1 -> hook 4
    env *= np.where(t > 267.6,
                    np.clip(1.0 - (t - 267.6) / 11.0, 0.15, 1.0), 1.0)
    return env.astype(np.float32)[:, None]


def process(part, cfg):
    x = load(part)
    board = Pedalboard([])
    if cfg["hpf"]:
        board.append(HighpassFilter(cutoff_frequency_hz=cfg["hpf"]))
    if part == "drums_kick":
        board.append(LowShelfFilter(cutoff_frequency_hz=90, gain_db=2.5))
        board.append(HighpassFilter(cutoff_frequency_hz=28))
    if part == "acoustic_guitar":
        board.append(HighShelfFilter(cutoff_frequency_hz=6500, gain_db=1.5))
    if cfg["comp"]:
        th, ratio = cfg["comp"]
        board.append(Compressor(threshold_db=th, ratio=ratio,
                                attack_ms=12, release_ms=180))
    if cfg["verb"]:
        room, wet = cfg["verb"]
        board.append(Reverb(room_size=room, damping=0.45, wet_level=wet,
                            dry_level=1.0 - wet * 0.35, width=0.95))
    x = board(x, SR)

    if part.startswith("strings"):
        x = x * swell_envelope(len(x))

    # set the balance by gated loudness so it is predictable, not guesswork
    measured = meter.integrated_loudness(x)
    if np.isfinite(measured):
        x = x * (10.0 ** ((cfg["lufs"] - measured) / 20.0))
    x = stereo_place(x, cfg["pan"], cfg["width"])
    return x.astype(np.float32), measured


def arrangement_automation(n):
    """Section-level fader moves: intro and outro sit back, verses drop under
    the hooks. This is ordinary verse/chorus automation - it also keeps the
    song's loudness range from collapsing into one flat level."""
    # (start bar, gain dB) for the 100-bar form
    moves = [(0, -3.0), (9, -1.3), (25, 1.0), (37, -1.2), (53, 1.2),
             (65, -1.0), (81, -2.0), (85, 1.5), (97, -2.5)]
    db = np.zeros(n, dtype=np.float32)
    for i, (bar, g) in enumerate(moves):
        t0 = bar * BAR
        t1 = moves[i + 1][0] * BAR if i + 1 < len(moves) else LENGTH
        db[int(t0 * SR):int(min(t1, LENGTH) * SR)] = g
    # smooth the steps so the moves ride in over ~0.6 s instead of jumping
    k = int(0.6 * SR)
    win = np.hanning(k)
    win /= win.sum()
    db = np.convolve(np.pad(db, (k, k), mode="edge"), win, mode="same")[k:-k]
    return (10.0 ** (db / 20.0)).astype(np.float32)[:, None]


def main():
    mixed = np.zeros((int(LENGTH * SR), 2), dtype=np.float32)
    print(f"{'stem':<18}{'measured':>10}{'target':>9}{'gain':>9}")
    print("-" * 48)
    for part, cfg in MIX.items():
        y, measured = process(part, cfg)
        gain = cfg["lufs"] - measured if np.isfinite(measured) else 0.0
        print(f"{part:<18}{measured:>9.1f}L{cfg['lufs']:>8.1f}L{gain:>+8.1f}dB")
        mixed[:len(y)] += y

    # gentle bus glue, then the limiter last
    bus = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=24),
        Compressor(threshold_db=-16, ratio=1.7, attack_ms=28, release_ms=260),
        HighShelfFilter(cutoff_frequency_hz=9000, gain_db=1.0),
    ])
    mixed = bus(mixed, SR)

    # master-fader moves, after the glue compressor so they survive it
    mixed = mixed * arrangement_automation(len(mixed))

    # Normalise to target, then let the limiter shave only what pokes above
    # the ceiling. Converges in a couple of passes because limiting this light
    # barely moves the integrated loudness.
    for _ in range(4):
        cur = meter.integrated_loudness(mixed)
        mixed = mixed * (10.0 ** ((TARGET_LUFS - cur) / 20.0))
        mixed = peak_limit(mixed, ceiling_db=-1.2)
        if abs(meter.integrated_loudness(mixed) - TARGET_LUFS) < 0.15:
            break

    # short fades so nothing clicks at the boundaries
    fi, fo = int(0.02 * SR), int(2.5 * SR)
    mixed[:fi] *= np.linspace(0, 1, fi)[:, None]
    mixed[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.6

    out = os.path.join(OUTDIR, "mix.wav")
    sf.write(out, mixed, SR, subtype="PCM_24")
    final = meter.integrated_loudness(mixed)
    peak = float(np.max(np.abs(mixed)))
    print(f"\nmix -> {out}")
    print(f"  duration {len(mixed) / SR:.2f}s  peak {peak:.4f} "
          f"({20 * np.log10(peak):.2f} dBFS)  integrated {final:.2f} LUFS")


if __name__ == "__main__":
    main()
