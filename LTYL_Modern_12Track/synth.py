"""Sound-design layer for the 12-track LTYL rework.

Every asset is synthesized from first principles rather than sampled, so the
whole kit is tuned to G minor by construction instead of being pitch-shifted
into key after the fact. All generators return mono float64 at SR.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, lfilter, resample_poly

SR = 48000

_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
       "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
       "Bb": 10, "B": 11}


def nf(name: str) -> float:
    """'Bb3' -> 233.08 Hz."""
    k = len(name)
    while name[k - 1].isdigit() or name[k - 1] == "-":
        k -= 1
    midi = 12 * (int(name[k:]) + 1) + _PC[name[:k]]
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


# ---------------------------------------------------------------- primitives

def t_axis(dur: float) -> np.ndarray:
    return np.arange(int(dur * SR)) / SR


def exp_env(n: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (tau * SR))


def adsr(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    s_n = max(0, n - a_n - d_n - r_n)
    return np.concatenate([
        np.linspace(0.0, 1.0, a_n, endpoint=False),
        np.linspace(1.0, s, d_n, endpoint=False),
        np.full(s_n, s),
        np.linspace(s, 0.0, n - a_n - d_n - s_n),
    ])[:n]


def _sos(kind: str, f, order=4):
    if kind == "band":
        return butter(order, [f[0] / (SR / 2), f[1] / (SR / 2)], btype="band", output="sos")
    return butter(order, f / (SR / 2), btype=kind, output="sos")


def lp(x, f, order=4):
    return sosfilt(_sos("low", min(f, SR / 2 * 0.98), order), x)


def hp(x, f, order=4):
    return sosfilt(_sos("high", f, order), x)


def bp(x, lo, hi, order=4):
    return sosfilt(_sos("band", (lo, min(hi, SR / 2 * 0.98)), order), x)


def norm(x, peak=1.0):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def fade(x, ms=6.0):
    """Kill DC clicks at both ends of a one-shot."""
    n = min(int(ms / 1000 * SR), len(x) // 2)
    if n < 2:
        return x
    w = np.linspace(0.0, 1.0, n)
    x = x.copy()
    x[:n] *= w
    x[-n:] *= w[::-1]
    return x


def noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


# ------------------------------------------------------- T1  acoustic guitar

def karplus(freq: float, dur: float, seed: int, decay: float = 0.9965,
            brightness: float = 0.55) -> np.ndarray:
    """Extended Karplus-Strong. The pick burst is pre-filtered so the attack
    reads as a fingered steel string rather than a noise splat."""
    n = int(dur * SR)
    delay = max(2, int(round(SR / freq)))
    rng = np.random.default_rng(seed)
    buf = rng.uniform(-1.0, 1.0, delay)
    buf = lfilter([brightness], [1.0, -(1.0 - brightness)], buf)
    buf /= np.max(np.abs(buf)) + 1e-12

    out = np.empty(n)
    idx = 0
    prev = 0.0
    for i in range(n):
        cur = buf[idx]
        out[i] = cur
        buf[idx] = decay * 0.5 * (cur + prev)
        prev = cur
        idx += 1
        if idx == delay:
            idx = 0

    body = bp(out, 90, 5200, order=2)          # guitar body resonance
    pick = noise(int(0.008 * SR), seed + 991) * exp_env(int(0.008 * SR), 0.0022)
    body[:len(pick)] += bp(pick, 1800, 6500, order=2) * 0.35
    return fade(norm(body, 0.9), 4.0)


def guitar_note(name: str, dur: float, seed: int) -> np.ndarray:
    return karplus(nf(name), dur, seed)


# ---------------------------------------------------------------- T2  piano

def _tri(f: float, t: np.ndarray, nharm: int, phase: float) -> np.ndarray:
    out = np.zeros(len(t))
    s = 1.0
    for k in range(1, nharm + 1, 2):
        if f * k > SR / 2 * 0.85:
            break
        out += s * np.sin(2 * np.pi * f * k * t + phase) / (k * k)
        s = -s
    return out * (8 / np.pi ** 2)


def synth_chord(names: list[str], dur: float, seed: int) -> np.ndarray:
    """Dark FM-bell chord — the harmonic anchor on beat 1.

    Replaces the acoustic piano this slot used to hold. A fast-decaying FM
    index gives a struck attack for definition, while a detuned triangle body
    carries the sustain; the whole thing is filtered dark so it reads as
    atmosphere rather than as a keyboard part.
    """
    t = t_axis(dur)
    rng = np.random.default_rng(seed)
    out = np.zeros(len(t))
    for name in names:
        f = nf(name)
        idx = 3.2 * np.exp(-t / 0.09)                     # struck attack
        mod = np.sin(2 * np.pi * f * 2.0 * t + rng.uniform(0, 2 * np.pi))
        voice = np.sin(2 * np.pi * f * t + idx * mod)
        for cents in (-5.0, 5.0):
            voice += 0.45 * _tri(f * 2 ** (cents / 1200), t, 16,
                                 rng.uniform(0, 2 * np.pi))
        out += voice * exp_env(len(t), dur * 0.30)
    out /= len(names)
    out = lp(out, 2200, order=2)
    out *= adsr(len(out), 0.012, 0.10, 0.75, 0.40)
    return fade(norm(out, 0.9), 6.0)


# ------------------------------------------------------------------ T3  pad

def _saw(f: float, t: np.ndarray, nharm: int, phase: float) -> np.ndarray:
    out = np.zeros(len(t))
    for k in range(1, nharm + 1):
        if f * k > SR / 2 * 0.85:
            break
        out += np.sin(2 * np.pi * f * k * t + phase * k) / k
    return out * (2 / np.pi)


def pad_chord(names: list[str], dur: float, seed: int) -> np.ndarray:
    """Detuned saw stack, slow bow-like attack, filtered dark."""
    t = t_axis(dur)
    rng = np.random.default_rng(seed)
    out = np.zeros(len(t))
    for name in names:
        f = nf(name)
        for cents in (-7.0, 0.0, 7.0):
            out += _saw(f * 2 ** (cents / 1200), t, 40, rng.uniform(0, 2 * np.pi))
    out /= len(names) * 3

    # slow filter bloom
    out = lp(out, 2600, order=2)
    out *= adsr(len(out), 0.55, 0.30, 0.80, 0.60)
    return fade(norm(out, 0.85), 20.0)


# -------------------------------------------------------- T4  reverse swell

def reverse_swell(dur: float, seed: int) -> np.ndarray:
    """A decaying cymbal/guitar hybrid, then flipped so it blooms into the hit."""
    n = int(dur * SR)
    t = t_axis(dur)
    cym = np.zeros(n)
    rng = np.random.default_rng(seed)
    for f in (317.0, 461.0, 613.0, 797.0, 1013.0, 1319.0):
        for m in (1.0, 1.63, 2.41, 3.17):
            cym += np.sin(2 * np.pi * f * m * t + rng.uniform(0, 2 * np.pi))
    cym = bp(cym, 900, 9000, order=2) / 24.0
    cym += bp(noise(n, seed + 3), 1200, 11000, order=2) * 0.55

    chord = np.zeros(n)
    for i, name in enumerate(("G3", "Bb3", "D4")):
        chord[:n] += karplus(nf(name), dur, seed + 40 + i)[:n]
    cym += chord * 0.45

    cym *= exp_env(n, dur * 0.38)
    return fade(norm(cym[::-1], 0.9), 8.0)


# ------------------------------------------------------------ T5  high pluck

def pluck(name: str, dur: float, seed: int) -> np.ndarray:
    """Sine-core bell pluck with two inharmonic partials for a glassy top."""
    f = nf(name)
    t = t_axis(dur)
    out = (np.sin(2 * np.pi * f * t) * exp_env(len(t), dur * 0.30)
           + 0.30 * np.sin(2 * np.pi * f * 2.76 * t) * exp_env(len(t), dur * 0.11)
           + 0.14 * np.sin(2 * np.pi * f * 5.40 * t) * exp_env(len(t), dur * 0.05))
    out *= adsr(len(out), 0.0016, 0.02, 0.85, 0.25)
    return fade(norm(out, 0.9), 3.0)


# ------------------------------------------------------------------ T6  kick

def kick(dur: float = 0.42) -> np.ndarray:
    n = int(dur * SR)
    t = t_axis(dur)
    f = 46.0 + (135.0 - 46.0) * np.exp(-t / 0.028)          # pitch drop
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * exp_env(n, 0.085)
    click = noise(int(0.005 * SR), 11) * exp_env(int(0.005 * SR), 0.0013)
    body[:len(click)] += hp(click, 2500, order=2) * 0.5
    body = np.tanh(body * 1.9) / np.tanh(1.9)               # soft clip in the box
    return fade(norm(body, 0.95), 3.0)


# ------------------------------------------------------------ T7  snare/rim

def rimshot(dur: float = 0.115) -> np.ndarray:
    """Sharp, dry, short. No tail — the tail is what eats a rap vocal."""
    n = int(dur * SR)
    t = t_axis(dur)
    crack = bp(noise(n, 23), 1500, 7000, order=3) * exp_env(n, 0.019)
    tone = (np.sin(2 * np.pi * 331.0 * t) * exp_env(n, 0.013) * 0.55
            + np.sin(2 * np.pi * 1740.0 * t) * exp_env(n, 0.007) * 0.35)
    out = crack * 0.85 + tone
    out = np.tanh(out * 1.5) / np.tanh(1.5)
    return fade(norm(out, 0.95), 2.0)


# ---------------------------------------------------------------- T8/T9  hats

def _metal(n: int, seed: int) -> np.ndarray:
    t = np.arange(n) / SR
    sq = np.zeros(n)
    for f in (2434.0, 3116.0, 3671.0, 4218.0, 5049.0, 5926.0):
        sq += np.sign(np.sin(2 * np.pi * f * t + seed * 0.11))
    return sq / 6.0


def hihat(dur: float = 0.055, seed: int = 31) -> np.ndarray:
    n = int(dur * SR)
    x = _metal(n, seed) * 0.55 + noise(n, seed) * 0.45
    x = hp(x, 7200, order=4) * exp_env(n, dur * 0.22)
    return fade(norm(x, 0.9), 1.5)


def open_hat(dur: float = 0.30, seed: int = 37) -> np.ndarray:
    n = int(dur * SR)
    x = _metal(n, seed) * 0.6 + noise(n, seed) * 0.4
    x = hp(x, 6200, order=4) * exp_env(n, dur * 0.30)
    return fade(norm(x, 0.9), 2.0)


def woodblock(dur: float = 0.075, seed: int = 41) -> np.ndarray:
    """Dark snap/woodblock — a resonant knock, not a clap."""
    n = int(dur * SR)
    t = t_axis(dur)
    tone = (np.sin(2 * np.pi * 812.0 * t) * exp_env(n, 0.012)
            + 0.5 * np.sin(2 * np.pi * 1571.0 * t) * exp_env(n, 0.006))
    snap = bp(noise(n, seed), 1900, 5200, order=3) * exp_env(n, 0.010)
    return fade(norm(tone * 0.7 + snap * 0.6, 0.9), 2.0)


# --------------------------------------------------------------- T10  impact

def impact(dur: float = 3.2) -> np.ndarray:
    n = int(dur * SR)
    t = t_axis(dur)
    f = 33.0 + (78.0 - 33.0) * np.exp(-t / 0.12)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * exp_env(n, 0.62)

    rng = np.random.default_rng(57)
    crash = np.zeros(n)
    for base in (412.0, 587.0, 733.0, 941.0, 1237.0, 1583.0):
        for m in (1.0, 1.71, 2.53, 3.44, 4.81):
            crash += np.sin(2 * np.pi * base * m * t + rng.uniform(0, 2 * np.pi))
    crash = bp(crash / 30.0, 800, 8500, order=2)
    crash += bp(noise(n, 59), 1500, 10000, order=2) * 0.5
    crash *= exp_env(n, 0.40)
    crash = lp(crash, 7000, order=2)                        # keep it dark

    return fade(norm(boom * 1.0 + crash * 0.55, 0.95), 4.0)


# ------------------------------------------------------------------ T11  808

def sub808(freqs: np.ndarray, dur: float, glide_ms: float = 60.0,
           start_freq: float | None = None) -> np.ndarray:
    """Pure sine 808 driven by a per-sample frequency curve.

    `start_freq` (the previous note's pitch) engages a `glide_ms` portamento
    ramp so overlapping notes bend into each other instead of retriggering.
    """
    n = len(freqs)
    f = freqs.copy()
    if start_freq is not None and glide_ms > 0:
        g = min(int(glide_ms / 1000 * SR), n)
        # exponential (musical) interpolation in the log-frequency domain
        f[:g] = np.exp(np.linspace(np.log(start_freq), np.log(f[0]), g))
    phase = 2 * np.pi * np.cumsum(f) / SR
    out = np.sin(phase)
    env = adsr(n, 0.006, 0.05, 0.90, min(0.22, dur * 0.35))
    return out * env


# ------------------------------------------------- T13  aggressive lead synth

def _square(f: float, t: np.ndarray, nharm: int, phase: float) -> np.ndarray:
    out = np.zeros(len(t))
    for k in range(1, nharm + 1, 2):
        if f * k > SR / 2 * 0.85:
            break
        out += np.sin(2 * np.pi * f * k * t + phase) / k
    return out * (4 / np.pi)


def _sweep_lp(x: np.ndarray, f_hi: float, f_lo: float, tau: float,
              block: int = 256) -> np.ndarray:
    """Time-varying lowpass — the filter envelope is what makes a pluck pluck.

    Coefficients are recomputed per block while the biquad state carries across
    the boundary, so the cutoff glides instead of producing the zipper noise a
    naive block-by-block refilter would.
    """
    out = np.zeros_like(x)
    zi = None
    for i in range(0, len(x), block):
        j = min(i + block, len(x))
        fc = float(np.clip(f_lo + (f_hi - f_lo) * np.exp(-(i / SR) / tau),
                           60.0, SR / 2 * 0.95))
        sos = butter(2, fc / (SR / 2), btype="low", output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        out[i:j], zi = sosfilt(sos, x[i:j], zi=zi)
    return out


def lead_pluck(name: str, dur: float, seed: int, voices: int = 5,
               detune_cents: float = 18.0, f_hi: float = 7000.0,
               f_lo: float = 620.0, sweep_tau: float = 0.055,
               drive: float = 2.4) -> np.ndarray:
    """Detuned saw/pulse stack through a fast downward filter sweep, then
    driven. The detune spread is the growl, the sweep is the attack bite."""
    f = nf(name)
    t = t_axis(dur)
    rng = np.random.default_rng(seed)

    x = np.zeros(len(t))
    spread = max(1.0, (voices - 1) / 2)
    for v in range(voices):
        cents = (v - (voices - 1) / 2) / spread * detune_cents
        x += _saw(f * 2 ** (cents / 1200), t, 48, rng.uniform(0, 2 * np.pi))
    x /= voices
    x += 0.32 * _square(f, t, 24, rng.uniform(0, 2 * np.pi))      # hollow edge
    x += 0.18 * _saw(f * 0.5, t, 24, rng.uniform(0, 2 * np.pi))   # sub octave

    x = _sweep_lp(x, f_hi, f_lo, sweep_tau)
    n = len(t)
    x *= np.minimum(1.0, np.arange(n) / max(1.0, 0.0015 * SR)) * exp_env(n, dur * 0.30)
    x = np.tanh(x * drive) / np.tanh(drive)
    return fade(norm(x, 0.9), 3.0)


# --------------------------------------------------------- T12  vocal chops

def vocal_chop(name: str, dur: float, seed: int, vowel=(690.0, 1180.0, 2560.0)) -> np.ndarray:
    """Formant-synthesized 'ahh', then resampled 2:1 so it drops exactly 12
    semitones. Shifting the formants with the pitch is what gives a real
    pitched-down chop its hollow, ethereal weight."""
    f0 = nf(name)
    t = t_axis(dur / 2)                                     # halved: 2x resample doubles it
    vib = 1.0 + 0.006 * np.sin(2 * np.pi * 4.7 * t)
    src = np.zeros(len(t))
    rng = np.random.default_rng(seed)
    for k in range(1, 45):
        fk = f0 * k
        if fk > SR / 2 * 0.85:
            break
        src += np.sin(2 * np.pi * fk * np.cumsum(vib) / SR + rng.uniform(0, 2 * np.pi)) / k

    voiced = np.zeros(len(t))
    for fc, g in zip(vowel, (1.0, 0.62, 0.32)):
        voiced += bp(src, fc * 0.86, fc * 1.16, order=2) * g
    voiced += bp(noise(len(t), seed + 5), 2000, 6000, order=2) * 0.02   # breath

    voiced *= adsr(len(voiced), 0.14, 0.18, 0.72, 0.45)
    down = resample_poly(voiced, 2, 1)                      # -12 semitones
    return fade(norm(down, 0.9), 25.0)
