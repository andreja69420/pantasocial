"""Render each humanized MIDI part through its real sampled library."""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIDI = os.path.join(ROOT, "output", "midi")
STEMS = os.path.join(ROOT, "output", "stems")
S = "/home/user/samples"
os.makedirs(STEMS, exist_ok=True)

VPO = f"{S}/vpo/Virtual-Playing-Orchestra3/Strings"

# part -> (sfz library, human-readable source)
PARTS = {
    "piano": (
        f"{S}/SalamanderGrandPianoV3_48khz24bit/SalamanderGrandPianoV3.sfz",
        "Salamander Grand Piano V3 (Yamaha C5, 16 vel layers)"),
    "acoustic_guitar": (
        f"{S}/freepats/FSS-SteelStringGuitar-SFZ-20200521/"
        f"FSS-SteelStringGuitar-20200521.sfz",
        "FreePats FS Seagull steel-string acoustic"),
    "drums_kick": (
        f"{S}/avl/AVL_Drumkits_1.0/Black_Pearl_5pc.sfz",
        "AVL Drumkits Black Pearl - Pearl 22\" kick (5 vel layers)"),
    "drums_kit": (
        f"{S}/avl/AVL_Drumkits_1.0/Black_Pearl_5pc.sfz",
        "AVL Drumkits Black Pearl 5pc - snare/toms/cymbals"),
    "bass_sub": (
        f"{S}/karoryfer/Programs/04-babyblue_warm.sfz",
        "Karoryfer Black & Blue Basses - blue solidbody, pick"),
    "bass_electric": (
        f"{S}/karoryfer/Programs/05-darkblack_pluck.sfz",
        "Karoryfer Black & Blue Basses - black hollowbody, fingers"),
    "strings_violin": (
        f"{VPO}/1st-violin-SEC-sustain.sfz",
        "Virtual Playing Orchestra - 1st violin section sustain"),
    "strings_cello": (
        f"{VPO}/cello-SEC-sustain.sfz",
        "Virtual Playing Orchestra - cello section sustain"),
    "electric_clean": (
        f"{S}/karoryfer/Programs/04-green_twang.sfz",
        "Karoryfer Black & Green Guitars - green Gretsch, clean"),
    "electric_power": (
        f"{S}/karoryfer/Programs/07-black_twang.sfz",
        "Karoryfer Black & Green Guitars - black Hofner Club"),
    "electric_power2": (
        f"{S}/karoryfer/Programs/04-green_twang.sfz",
        "Karoryfer Black & Green Guitars - green Gretsch (double-track)"),
}

# Each part legitimately stops at a different bar (the outro drops instruments
# one by one), so the minimum length is checked per part, not globally.
EXPECT_MIN = {
    "piano": 238.0,
    "acoustic_guitar": 232.0,
    "drums_kick": 227.0,
    "drums_kit": 227.0,
    "bass_sub": 229.0,
    "bass_electric": 218.0,
    "strings_violin": 232.0,
    "strings_cello": 235.0,
    "electric_clean": 218.0,
    "electric_power": 218.0,
    "electric_power2": 218.0,
}


def main():
    fails = []
    print(f"{'part':<17}{'dur':>9}{'peak':>9}{'rms':>10}  library")
    print("-" * 96)
    for part, (sfz, desc) in PARTS.items():
        mid = os.path.join(MIDI, f"{part}.mid")
        wav = os.path.join(STEMS, f"{part}.wav")
        if not os.path.exists(sfz):
            fails.append(f"{part}: missing SFZ {sfz}")
            continue
        r = subprocess.run(
            ["sfizz_render", "--sfz", sfz, "--midi", mid, "--wav", wav,
             "-s", "48000", "--polyphony", "512", "-q", "10", "--use-eot"],
            capture_output=True, text=True)
        if r.returncode != 0:
            fails.append(f"{part}: sfizz_render failed\n{r.stderr[-800:]}")
            continue

        d, sr = sf.read(wav)
        mono = d.mean(axis=1) if d.ndim > 1 else d
        dur = len(mono) / sr
        peak = float(np.max(np.abs(mono)))
        rms = float(np.sqrt(np.mean(mono ** 2)))
        print(f"{part:<17}{dur:>8.2f}s{peak:>9.4f}{rms:>10.5f}  {desc}")

        if peak < 1e-3:
            fails.append(f"{part}: SILENT stem (peak={peak:.6f})")
        want = EXPECT_MIN[part]
        if dur < want:
            fails.append(f"{part}: too short ({dur:.2f}s < {want}s)")
        if sr != 48000:
            fails.append(f"{part}: wrong sample rate {sr}")

    print()
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  !", f)
        sys.exit(1)
    print("All stems rendered, non-silent, correct length and sample rate.")


if __name__ == "__main__":
    main()
