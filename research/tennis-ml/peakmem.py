"""Peak memory of a command including all its child processes, sampled every 0.2 s: sum of PSS over the process tree
(pages shared by forked workers are split between them instead of counted once per process as RSS would).

usage: peakmem.py COMMAND [ARGS ...]   -> prints 'peak PSS X.XX GB, T s' on stderr, exits with the command's code
"""
import os
import subprocess
import sys
import time


def tree_pss(root):
    kids = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                with open(f"/proc/{p}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(p))
            except OSError:
                pass
    total, stack = 0, [root]
    while stack:
        p = stack.pop()
        try:
            with open(f"/proc/{p}/smaps_rollup") as f:
                total += next(int(x.split()[1]) for x in f if x.startswith("Pss:")) * 1024
        except (OSError, StopIteration):
            pass
        stack += kids.get(p, [])
    return total


t0 = time.time()
proc = subprocess.Popen(sys.argv[1:])
peak = 0
while proc.poll() is None:
    peak = max(peak, tree_pss(proc.pid))
    time.sleep(0.2)
print(f"peak PSS {peak / 2 ** 30:.2f} GB, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
sys.exit(proc.returncode)
