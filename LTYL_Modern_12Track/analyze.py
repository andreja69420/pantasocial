"""Automated QA pass over the rendered master and stems.

    python analyze.py

Checks level balance, tonal tilt, arrangement dynamics, groove-grid accuracy
and that the 808 sidechain is measurably ducking, then draws a spectrogram.
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, hilbert, sosfilt

import arrange
import mixdsp
from arrange import BAR, BEAT
from synth import SR, nf as nf_hz

HERE = os.path.dirname(os.path.abspath(__file__))


def dbfs(x):
    return 20 * np.log10(np.max(np.abs(x)) + 1e-12)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def band_rms(mono, lo, hi):
    sos = butter(4, [lo / (SR / 2), min(hi, SR / 2 * 0.98) / (SR / 2)],
                 btype="band", output="sos")
    return rms_db(sosfilt(sos, mono))


def main() -> None:
    y, sr = sf.read(os.path.join(HERE, "LoveTheWayYouLie_Modern12Track.wav"),
                    always_2d=True)
    st = y.T
    mono = st.mean(axis=0)

    print("=" * 70)
    print("  MASTER")
    print("=" * 70)
    print(f"  peak      {dbfs(st):+.3f} dBFS")
    print(f"  RMS       {rms_db(st):+.2f} dBFS")
    print(f"  LUFS-I    {mixdsp.lufs_integrated(st):+.2f}")
    print(f"  crest     {dbfs(st) - rms_db(st):.2f} dB")
    print(f"  DC offset {np.mean(mono):+.2e}")
    corr = np.corrcoef(st[0], st[1])[0, 1]
    print(f"  L/R corr  {corr:+.3f}  (mono-safe if > 0)")
    print(f"  duration  {st.shape[1] / sr:.2f} s  ({st.shape[1] / sr / BAR:.1f} bars)")

    print("\n  Octave-band tilt (RMS, dB rel. broadband):")
    ref = rms_db(mono)
    for lo, hi in [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
                   (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]:
        v = band_rms(mono, lo, hi) - ref
        bar = "#" * max(0, int((v + 45) / 1.6))
        tag = "  <-- vocal pocket" if lo in (1000, 2000) else ""
        print(f"    {lo:>5}-{hi:<5} Hz {v:+7.2f}  {bar}{tag}")

    print("\n" + "=" * 70)
    print("  ARRANGEMENT DYNAMICS (per 8-bar block)")
    print("=" * 70)
    # Broadband RMS is dominated by the 808 and barely moves; the section lift
    # lives in the musical band, so measure there too.
    for z0, z1, name in arrange.ZONES:
        a, b = int(z0 * BAR * sr), int(z1 * BAR * sr)
        seg = mono[a:b]
        live = [t for t in sorted(arrange.LAYERS, key=lambda s: int(s[1:]))
                if arrange.lg(t, z0) > 0]
        print(f"    bars {z0+1:>2}-{z1:<2}  {name:<8} "
              f"broadband {rms_db(seg):+.2f} | 300Hz-6kHz {band_rms(seg, 300, 6000):+.2f} dBFS"
              f" | {len(live):>2} tracks")

    print("\n" + "=" * 70)
    print("  STEM BALANCE (post-DSP, pre-master)")
    print("=" * 70)
    stems = sorted(glob.glob(os.path.join(HERE, "stems", "*.wav")),
                   key=lambda p: int(os.path.basename(p).split("_")[0][1:]))
    for p in stems:
        s, _ = sf.read(p, always_2d=True)
        s = s.T
        w = (s[0] - s[1])
        width = rms_db(w) - rms_db(s.mean(axis=0))
        name = os.path.basename(p).replace(".wav", "")
        print(f"    {name:<24} peak {dbfs(s):7.2f} | RMS {rms_db(s):7.2f} | "
              f"width {width:+6.1f} dB")

    print("\n" + "=" * 70)
    print("  SIDECHAIN VERIFICATION (T11 808 vs T6 kick)")
    print("=" * 70)
    # 1. the envelope generator itself, in isolation
    env_t = mixdsp.sidechain_env(int(1.0 * sr), [0.2])
    floor_i = int(np.argmin(env_t))
    depth = 20 * np.log10(env_t.min())
    atk = (floor_i - int(0.2 * sr)) / sr * 1000
    rec = env_t[floor_i:]
    rel = (np.argmax(rec > mixdsp.db(-0.5)) / sr * 1000) if (rec > mixdsp.db(-0.5)).any() else np.nan
    print(f"    envelope depth      {depth:+.2f} dB   (spec -5.0)")
    print(f"    attack to floor     {atk:.2f} ms     (spec 2.0)")
    print(f"    release to -0.5 dB  {rel:.1f} ms     (spec 80.0 time-constant)")

    # 2. the rendered 808 stem, measured only at offbeat kicks — a kick landing
    #    on an 808 note onset has nothing sustaining to duck, so those would
    #    read as a level *rise* and poison the average.
    sub, _ = sf.read(os.path.join(HERE, "stems", "T11_808 Sub.wav"), always_2d=True)
    sub = sub.T.mean(axis=0)
    k = int(0.004 * sr)
    aenv = np.convolve(np.abs(sub), np.ones(k) / k, mode="same")
    onsets = {round(t, 4) for t in _note_onsets()}
    hits = [t for t in _kick_grid()
            if 20.0 < t < 110.0 and round(t, 4) not in onsets]
    w0, w1 = int(0.002 * sr), int(0.020 * sr)
    ratios = []
    for t in hits:
        i = int(t * sr)
        pre = aenv[i - int(0.030 * sr):i - int(0.004 * sr)].max(initial=0)
        post = aenv[i + w0:i + w1].max(initial=0)
        if pre > 1e-4:
            ratios.append(20 * np.log10((post + 1e-12) / pre))
    # A 50-87 Hz sine only completes ~1 cycle inside the probe window, so the
    # window necessarily catches the recovery ramp, not the floor. Compare
    # against what the envelope itself predicts over the same window.
    expected = 20 * np.log10(env_t[int(0.2 * sr) + w0:int(0.2 * sr) + w1].max())
    print(f"    offbeat kicks probed {len(ratios)}")
    print(f"    measured over +2..20ms {np.median(ratios):+.2f} dB median  "
          f"(envelope predicts {expected:+.2f} dB)")

    print("\n" + "=" * 70)
    print("  808 PORTAMENTO (instantaneous pitch across a chord change)")
    print("=" * 70)
    sos_lo = butter(4, 200 / (sr / 2), btype="low", output="sos")
    lowsub = sosfilt(sos_lo, sub)
    inst = np.abs(np.diff(np.unwrap(np.angle(hilbert(lowsub))))) * sr / (2 * np.pi)
    k = int(0.006 * sr)
    inst = np.convolve(inst, np.ones(k) / k, mode="same")
    roots = [arrange.PROGRESSION[i]["root"] for i in range(4)]
    for bar in (33, 34, 35):                      # mid-verse, notes fully sustained
        t0 = bar * BAR
        frm, to = roots[(bar - 1) % 4], roots[bar % 4]
        pts = [(-0.015, "before"), (0.000, "t=0"), (0.030, "+30ms"),
               (0.060, "+60ms"), (0.120, "+120ms")]
        vals = "  ".join(f"{lab}={inst[int((t0 + dt) * sr)]:6.1f}Hz" for dt, lab in pts)
        print(f"    bar {bar+1:>2}  {frm}->{to} "
              f"({nf_hz(frm):.1f}->{nf_hz(to):.1f} Hz)\n           {vals}")

    print("\n" + "=" * 70)
    print("  GROOVE GRID (onset alignment, ms from nearest 1/32)")
    print("=" * 70)
    # Smoothing must outrun the stem's own fundamental — a 46 Hz kick body has a
    # 22 ms period, so a 3 ms window "detects" every cycle of the decay.
    exp = _expected_counts()
    # Hats need a lower threshold than the drums: the arrangement drops hat
    # velocity to 0.5 in the stripped verses, which puts the quietest roll hits
    # under an 18 % gate even though they are sequenced correctly.
    for stem, label, expected, smooth, debounce, thr in (
            ("T6_Kick", "kick", exp["kick"], 0.030, 0.12, 0.18),
            ("T7_Snare-Rim", "snare", exp["snare"], 0.004, 0.20, 0.18),
            ("T8_Hi-Hat", "hat", exp["hat"], 0.002, 0.030, 0.12)):
        s, _ = sf.read(os.path.join(HERE, "stems", stem + ".wav"), always_2d=True)
        m = np.abs(s.T.mean(axis=0))
        w = max(1, int(smooth * sr))
        e = np.convolve(m, np.ones(w) / w, mode="same")
        # Pad a leading False: with mode="same" smoothing a hit at t=0 is
        # already above threshold at sample 0 and has no rising edge to find.
        above = np.r_[False, e > e.max() * thr]
        raw = np.flatnonzero(above[1:] & ~above[:-1]) / sr
        onsets, last = [], -1e9
        for o in raw:
            if o - last > debounce:
                onsets.append(o)
                last = o
        onsets = np.array(onsets)
        grid = BEAT / 8
        # An envelope crossing always lags the true onset; remove the constant
        # part of that lag so what is left is real timing jitter.
        off = np.array([o - round(o / grid) * grid for o in onsets])
        jit = np.abs(off - np.median(off)) * 1000
        flag = "OK " if len(onsets) == expected else "!! "
        print(f"    {flag}{label:<6} {len(onsets):>4} onsets (expect {expected:>3}) | "
              f"detect lag {np.median(off)*1000:+5.1f} ms | jitter max {jit.max():.3f} ms")

    # ------------------------------------------------------------- plots
    fig, ax = plt.subplots(3, 1, figsize=(16, 11),
                           gridspec_kw={"height_ratios": [1, 1, 2.2]})
    t = np.arange(len(mono)) / sr

    ax[0].plot(t, st[0], lw=0.3, color="#6366f1")
    ax[0].plot(t, st[1], lw=0.3, color="#a855f7", alpha=0.6)
    win = int(0.25 * sr)
    env = np.sqrt(np.convolve(mono ** 2, np.ones(win) / win, mode="same"))
    ax[0].plot(t, env * 3, color="#06b6d4", lw=1.4, label="RMS envelope (x3)")
    ax[0].plot(t, -env * 3, color="#06b6d4", lw=1.4)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_xlim(0, t[-1]); ax[0].set_ylim(-0.6, 0.6)
    ax[0].set_title("Master waveform — dashed = chorus downbeats")
    ax[0].set_ylabel("amplitude")

    # band energy over time: is the 1-5 kHz pocket actually holding open?
    for lo, hi, c, lab in ((20, 120, "#f59e0b", "sub 20-120"),
                           (120, 1000, "#a855f7", "body 120-1k"),
                           (1000, 5000, "#22d3ee", "VOCAL POCKET 1-5k"),
                           (5000, 16000, "#94a3b8", "top 5-16k")):
        b = sosfilt(butter(4, [lo / (sr / 2), min(hi, sr / 2 * 0.98) / (sr / 2)],
                           btype="band", output="sos"), mono)
        e = 20 * np.log10(np.sqrt(np.convolve(b ** 2, np.ones(win) / win,
                                              mode="same")) + 1e-9)
        ax[1].plot(t, e, color=c, lw=1.2, label=lab)
    ax[1].set_xlim(0, t[-1]); ax[1].set_ylim(-70, -10)
    ax[1].legend(loc="lower right", fontsize=8, ncol=4)
    ax[1].set_ylabel("band RMS (dBFS)")
    ax[1].set_title("Band energy over time")

    ax[2].specgram(mono, NFFT=4096, Fs=sr, noverlap=3072, cmap="magma",
                   vmin=-105, vmax=-35)
    ax[2].set_yscale("symlog", linthresh=100)
    ax[2].set_ylim(20, 20000)
    ax[2].axhline(1000, color="w", ls=":", lw=0.8)
    ax[2].axhline(5000, color="w", ls=":", lw=0.8)
    ax[2].set_ylabel("Hz  (dotted = 1-5 kHz vocal pocket)")
    ax[2].set_xlabel("seconds")

    for a_ in ax:
        for cs in arrange.CHORUS_STARTS:
            a_.axvline(cs * BAR, color="#06b6d4", ls="--", lw=1)
    fig.tight_layout()
    out = os.path.join(HERE, "analysis.png")
    fig.savefig(out, dpi=110)
    print(f"\n  wrote {out}")


def _note_onsets():
    """808 note start times (retriggers, where a duck has nothing to bite on)."""
    times = []
    for bar in range(arrange.BARS):
        if not arrange.lg("T11", bar):
            continue
        t0 = bar * BAR
        times.append(t0)
        if arrange.section_of(bar) == "chorus":
            times.append(t0 + 2.5 * BEAT)
    return times


def _kick_grid():
    """Re-derive kick trigger times from the arrangement rules (cheap)."""
    times = []
    for bar in range(arrange.BARS):
        if not arrange.lg("T6", bar):
            continue
        t0 = bar * BAR
        pos = [0.0, 1.5]
        if arrange.section_of(bar) == "chorus":
            pos.append(3.0)
            if (bar % 4) == 3:
                pos.append(3.5)
        times += [t0 + p * BEAT for p in pos]
    return sorted(times)


def _expected_counts():
    """Hit counts implied by the arrangement, so the grid check stays honest
    as the layer map changes."""
    snare = sum(1 for b in range(arrange.BARS) if arrange.lg("T7", b))
    hat = sum(14 if (b % 4) in (1, 3) else 8
              for b in range(arrange.BARS) if arrange.lg("T8", b))
    return {"kick": len(_kick_grid()), "snare": snare, "hat": hat}


if __name__ == "__main__":
    main()
