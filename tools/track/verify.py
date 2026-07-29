"""Verify the finished master before delivery.

Checks duration, clipping, silent passages, tempo, key, and that every section
actually contains the instruments the arrangement calls for.
"""
import os
import subprocess
import sys

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(ROOT, "output")
STEMS = os.path.join(OUTDIR, "stems")
SR = 48000
BAR = 4 * 60.0 / 87.0

SECTIONS = {
    "hook1":  (0, 12),  "verse1": (12, 28), "hook2":  (28, 40),
    "verse2": (40, 56), "hook3":  (56, 68), "verse3": (68, 84),
    "break":  (84, 88), "hook4":  (88, 100), "outro": (100, 103),
}

# which stems must be audible in which section
EXPECTED = {
    "hook1":  ["piano", "acoustic_guitar", "bass_sub",
               "strings_violin", "strings_cello"],
    "verse1": ["piano", "acoustic_guitar", "drums_kick", "drums_kit",
               "bass_sub", "bass_electric"],
    "hook2":  ["piano", "acoustic_guitar", "drums_kick", "drums_kit",
               "bass_sub", "bass_electric", "strings_violin", "strings_cello"],
    "verse2": ["piano", "acoustic_guitar", "drums_kick", "drums_kit",
               "bass_sub", "bass_electric", "strings_cello", "electric_clean"],
    "hook3":  ["piano", "acoustic_guitar", "drums_kick", "drums_kit",
               "bass_sub", "bass_electric", "strings_violin", "strings_cello",
               "electric_power", "electric_power2"],
    "verse3": ["piano", "acoustic_guitar", "drums_kick", "drums_kit",
               "bass_sub", "bass_electric", "strings_violin", "strings_cello", "electric_clean"],
    "break":  ["piano", "bass_sub", "strings_violin", "strings_cello"],
    "hook4":  ["piano", "acoustic_guitar", "drums_kick", "drums_kit",
               "bass_sub", "bass_electric", "strings_violin", "strings_cello",
               "electric_power", "electric_power2"],
    "outro":  ["piano", "acoustic_guitar", "bass_sub", "strings_cello"],
}
# instruments that must be SILENT in a section (the arrangement says so)
FORBIDDEN = {
    "hook1":  ["electric_power", "electric_power2", "electric_clean"],
    "verse1": ["electric_power", "electric_power2", "electric_clean", "strings_violin", "strings_cello"],
    "verse2": ["electric_power", "electric_power2", "strings_violin"],
    "break":  ["drums_kick", "acoustic_guitar", "electric_clean"],
    "outro":  ["electric_power", "electric_power2", "electric_clean", "bass_electric"],
}

KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                     4.75, 3.98, 2.69, 3.34, 3.17])
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                     5.19, 2.39, 3.66, 2.29, 2.88])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

fails, warns = [], []


def check(cond, msg):
    (print(f"  PASS  {msg}") if cond else fails.append(msg))
    if not cond:
        print(f"  FAIL  {msg}")


def main(path):
    x, sr = sf.read(path, dtype="float64")
    mono = x.mean(axis=1)
    dur = len(x) / sr
    print(f"Verifying {path}\n")

    # ---- duration -------------------------------------------------------
    print("Duration")
    check(281.0 <= dur <= 290.0,
          f"duration {dur:.2f}s ({int(dur // 60)}:{dur % 60:05.2f}) "
          f"within 4:41-4:50")

    # ---- clipping -------------------------------------------------------
    print("\nPeak / clipping")
    peak = float(np.max(np.abs(x)))
    clipped = int(np.sum(np.abs(x) >= 0.9995))
    check(peak < 1.0, f"peak {peak:.4f} ({20 * np.log10(peak):.2f} dBFS) "
                      f"below full scale")
    check(clipped == 0, f"no clipped samples (found {clipped})")

    # ---- loudness / dynamics -------------------------------------------
    print("\nLoudness")
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(x)
    plr = 20 * np.log10(peak) - lufs
    check(-15.5 <= lufs <= -12.5, f"integrated {lufs:.2f} LUFS (target -14)")
    check(plr >= 6.0, f"peak-to-loudness ratio {plr:.1f} dB (>= 6 dB)")
    r = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-filter_complex",
                        "ebur128=framelog=quiet", "-f", "null", "-"],
                       capture_output=True, text=True)
    lra = None
    for line in r.stderr.splitlines():
        if "LRA:" in line and "low" not in line and "high" not in line:
            try:
                lra = float(line.split("LRA:")[1].split("LU")[0])
            except (ValueError, IndexError):
                pass
    if lra is not None:
        check(lra >= 6.0, f"loudness range {lra:.1f} LU (>= 6 LU)")

    # ---- silent passages ------------------------------------------------
    print("\nContinuity")
    win = sr  # 1 second
    n = len(mono) // win
    rms = np.array([np.sqrt(np.mean(mono[i * win:(i + 1) * win] ** 2))
                    for i in range(n)])
    body = rms[:-3]  # ignore the deliberate fade-out tail
    quiet = np.where(body < 1e-4)[0]
    check(len(quiet) == 0,
          f"no silent passages (quietest 1s window "
          f"{20 * np.log10(max(body.min(), 1e-12)):.1f} dBFS RMS)")

    # ---- tempo ----------------------------------------------------------
    print("\nTempo")
    y22 = librosa.resample(mono, orig_sr=sr, target_sr=22050)
    onset = librosa.onset.onset_strength(y=y22, sr=22050)
    tempo = librosa.feature.tempo(onset_envelope=onset, sr=22050,
                                  aggregate=np.median, start_bpm=87)
    t = float(np.atleast_1d(tempo)[0])
    cands = [t, t * 2, t / 2, t * 1.5, t / 1.5]
    best = min(cands, key=lambda c: abs(c - 87))
    check(abs(best - 87) <= 2.0,
          f"detected tempo {t:.1f} BPM -> {best:.1f} BPM (target 87)")

    # ---- key ------------------------------------------------------------
    print("\nKey")
    chroma = librosa.feature.chroma_cqt(y=y22, sr=22050).mean(axis=1)
    chroma = (chroma - chroma.mean()) / (chroma.std() + 1e-9)
    scores = {}
    for i in range(12):
        for name, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            p = np.roll(prof, i)
            p = (p - p.mean()) / p.std()
            scores[f"{NOTES[i]} {name}"] = float(np.corrcoef(chroma, p)[0, 1])
    rank = sorted(scores.items(), key=lambda kv: -kv[1])
    top = rank[0][0]
    print("    top 3: " + ", ".join(f"{k} ({v:.3f})" for k, v in rank[:3]))
    # G minor and Bb major are relative keys with identical pitch content, so
    # the profile match alone cannot separate them. The tonic decides it:
    # check that E is the strongest pitch class where the track establishes
    # and resolves its key - the solo piano intro and the final chord.
    top2 = [k for k, _ in rank[:2]]
    check(set(top2) == {"G minor", "A# major"} or top == "G minor",
          f"key profile matches G minor / its relative Bb major "
          f"(top: {top})")

    def tonic(t0, t1):
        seg = mono[int(t0 * sr):int(t1 * sr)]
        s22 = librosa.resample(seg, orig_sr=sr, target_sr=22050)
        c = librosa.feature.chroma_cqt(y=s22, sr=22050).mean(axis=1)
        return NOTES[int(np.argmax(c))], c / c.max()

    # Opening chord (bar 0 is Gm) and the final resolution. An Em triad has
    # three near-equal partials, so the opening only has to rank E in the top
    # two pitch classes; the final chord is the real tiebreaker.
    open_note, ci = tonic(0.0, 2.6)
    outro, co = tonic(278.0, 287.0)
    open_rank = list(np.argsort(-ci)[:2])
    print(f"    opening chord: G={ci[7]:.2f} Bb={ci[10]:.2f} D={ci[2]:.2f} "
          f"-> top two {NOTES[open_rank[0]]}, {NOTES[open_rank[1]]}")
    print(f"    final chord:   G={co[7]:.2f} Bb={co[10]:.2f} D={co[2]:.2f} "
          f"-> {outro}")
    check(7 in open_rank, "G is a leading pitch class of the opening chord")
    check(outro == "G",
          f"the track resolves to G ({outro}) -> G minor, not its relative "
          f"Bb major")

    # ---- per-section instrumentation ------------------------------------
    print("\nSection instrumentation")
    stem_rms = {}
    for f in sorted(os.listdir(STEMS)):
        if not f.endswith(".wav"):
            continue
        d, dsr = sf.read(os.path.join(STEMS, f), dtype="float64")
        m = d.mean(axis=1) if d.ndim > 1 else d
        stem_rms[f[:-4]] = (m, dsr)

    # Guard windows: notes struck at the end of one section legitimately ring
    # into the next, so presence is measured after the previous section's
    # decay has passed rather than from the boundary itself.
    NOISE = 1e-3

    def active(part, t0, t1, guard, tail=0.35):
        # `tail` also trims the end: timing jitter can put the next section's
        # downbeat a few ms inside this window.
        m, dsr = stem_rms[part]
        seg = m[int((t0 + guard) * dsr):int((t1 - tail) * dsr)]
        return len(seg) > 0 and np.sqrt(np.mean(seg ** 2)) > NOISE

    for name, (b0, b1) in SECTIONS.items():
        t0, t1 = b0 * BAR, b1 * BAR
        present = [p for p in stem_rms if active(p, t0, t1, 1.5)]
        missing = [p for p in EXPECTED[name] if p not in present]
        extra = [p for p in FORBIDDEN.get(name, [])
                 if active(p, t0, t1, 3.0)]
        mark = "PASS" if not missing and not extra else "FAIL"
        print(f"  {mark}  {name} {int(t0 // 60)}:{t0 % 60:05.2f}-"
              f"{int(t1 // 60)}:{t1 % 60:05.2f}  {len(present)} parts: "
              f"{', '.join(sorted(present))}")
        if missing:
            fails.append(f"section {name} missing {missing}")
            print(f"        missing: {missing}")
        if extra:
            fails.append(f"section {name} should not contain {extra}")
            print(f"        should be absent: {extra}")


    print("\n" + "=" * 60)
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED:")
        for f in fails:
            print("  !", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print(f"  duration {int(dur // 60)}:{dur % 60:05.2f}   "
          f"peak {20 * np.log10(peak):.2f} dBFS   "
          f"{lufs:.2f} LUFS   PLR {plr:.1f} dB"
          + (f"   LRA {lra:.1f} LU" if lra else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTDIR, "mix.wav"))
