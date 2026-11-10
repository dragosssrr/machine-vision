"""check_layout.py -- report overfull boxes per frame.

Usage (from the project root, after a normal build):  python3 code/check_layout.py
It re-runs pdflatex once with the debug hook of preamble.tex and groups every
'Overfull' / 'Underfull \\vbox' warning under the title of the frame it belongs to.
"""
import re
import subprocess
import sys

# Use a separate job name so that main.pdf / main.aux of the normal build are never touched.
import glob, os, shutil
JOB = "layout_check"
for ext in ("aux", "bbl"):                      # citations and cross-references as in the real build
    if os.path.exists(f"main.{ext}"):
        shutil.copy(f"main.{ext}", f"{JOB}.{ext}")
cmd = ["pdflatex", "-interaction=nonstopmode", f"-jobname={JOB}", r"\def\debugframes{1}\input{main.tex}"]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
log = open(f"{JOB}.log", encoding="utf-8", errors="replace").read().splitlines()
for f in glob.glob(f"{JOB}.*"):
    if not f.endswith(".log"):
        os.remove(f)
os.remove(f"{JOB}.log")
title, hits, pending_h = "(before first frame)", {}, []
for line in log:
    m = re.match(r"\[FRAMETITLE\] (.*)", line)
    if m:
        title = m.group(1).strip()
        for l in pending_h:                      # \hbox warnings are emitted while the frame body is built,
            hits.setdefault(title, []).append(l)  # i.e. *before* its title is typeset -> belong to this frame
        pending_h = []
        continue
    if line.startswith("Overfull \\vbox") or line.startswith("Overfull \\hbox"):
        amount = float(re.search(r"\(([\d.]+)pt", line).group(1))
        if amount > 2.0:                          # ignore sub-2pt fuzz
            if line.startswith("Overfull \\hbox"):
                pending_h.append(line[:70])
            else:                                 # \vbox warnings come at shipout, after the title
                hits.setdefault(title, []).append(line[:70])
if not hits:
    print("No overfull boxes larger than 2pt.")
for t, ls in hits.items():
    print(f"- {t}")
    for l in ls:
        print("     ", l)
sys.exit(1 if hits else 0)
