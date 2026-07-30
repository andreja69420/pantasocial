"""Remap the FreePats Upright Piano KW for exposed solo playing.

The shipped mapping has two properties that hurt an exposed piano part:

1. It stretches some samples up to +3 semitones from their recorded pitch.
   Piano strings are inharmonic, and that inharmonicity does not transpose, so
   a 3-semitone stretch reads as the instrument being out of tune. Remapping
   every key to its *nearest* recorded pitch brings the worst case to +/-2.

2. The low samples use loop_mode=loop_continuous. They do not decay to silence
   (still ~0.03 RMS after 12 s), so a held note sustains indefinitely rather
   than dying away. no_loop lets them decay naturally; the samples run 7-12 s
   in the low register, longer than any note in this arrangement.

Writes UprightPianoKW-remapped.sfz next to the original. Run once after
downloading the library; render.py points at the remapped file.
"""
import os
import re
import sys

SRC = ("/home/user/samples/upright/UprightPianoKW-SFZ-20220221/"
       "UprightPianoKW-20220221.sfz")
DST = os.path.join(os.path.dirname(SRC), "UprightPianoKW-remapped.sfz")
LOKEY, HIKEY = 21, 108


def main(src=SRC, dst=DST):
    if not os.path.exists(src):
        sys.exit(f"not found: {src}")
    text = open(src, errors="ignore").read()

    # each <group> carries the velocity split; regions carry pitch + sample
    layers = {}
    for block in re.split(r"<group>", text)[1:]:
        head = block.split("<region>")[0]
        lo = re.search(r"lovel=(\d+)", head)
        hi = re.search(r"hivel=(\d+)", head)
        lo = int(lo.group(1)) if lo else 1
        hi = int(hi.group(1)) if hi else 127
        pitches = layers.setdefault((lo, hi), {})
        for region in block.split("<region>")[1:]:
            centre = re.search(r"pitch_keycenter=(\d+)", region)
            sample = re.search(r"sample=(\S+)", region)
            if centre and sample:
                pitches[int(centre.group(1))] = sample.group(1)
    if not layers:
        sys.exit("could not parse any regions")

    out = [
        "// Upright Piano KW - remapped by tools/track/remap_upright.py",
        "//   1. every key uses its NEAREST recorded pitch (the shipped map",
        "//      stretched some keys +3 semitones, which detunes audibly)",
        "//   2. loop_mode=no_loop throughout (the low samples looped forever",
        "//      instead of decaying)",
        "<global>", " ampeg_release=0.9", " width=100", "",
    ]
    worst = 0
    for (lo, hi), pitches in sorted(layers.items()):
        centres = sorted(pitches)
        out.append(f"<group>\n lovel={lo}\n hivel={hi}\n loop_mode=no_loop")
        for key in range(LOKEY, HIKEY + 1):
            centre = min(centres, key=lambda c: (abs(c - key), c))
            worst = max(worst, abs(centre - key))
            out.append(f"<region>\n key={key}\n pitch_keycenter={centre}\n"
                       f" sample={pitches[centre]}")
        out.append("")

    open(dst, "w").write("\n".join(out) + "\n")
    print(f"wrote {dst}")
    for (lo, hi), pitches in sorted(layers.items()):
        print(f"  velocity {lo}-{hi}: {len(pitches)} recorded pitches")
    print(f"  worst stretch now +/-{worst} semitones (was +3)")


if __name__ == "__main__":
    main()
