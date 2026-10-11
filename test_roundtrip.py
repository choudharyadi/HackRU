"""Check the hand-off between the two halves: the report solve.py saves is the map writer.py walks.

solve.report() writes out exactly what it printed (last_report.txt); writer.py reads that back,
works out the sheet it was measured on, and tours every mistake down and up. Gemini is stubbed
out and the arm is simulated, but the layout, the listing and the tour are the real code.

    python test_roundtrip.py
"""
import os
import subprocess
import sys
import tempfile

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

import solve
import writer
import make_test_page as fixture

lines = fixture.LINES
mistakes = [
    solve.Mistake(original=word, correction=corr, type=kind, explanation="test",
                  context=lines[line - 1], line=line, char=lines[line - 1].index(word) + 1)
    for line, word, corr, kind in fixture.MISTAKES
]
review = solve.Review(text="\n".join(lines), mistakes=mistakes,
                      corrected="", spoken="found them")

solve.check_text = lambda *a, **k: review  # no Gemini call
image = open(os.path.join(PROJECT, "page_test.jpg"), "rb").read()

tmp = tempfile.mkdtemp()
os.chdir(tmp)
out = solve.report(image, "image/jpeg", "unused-key", speaker=None)
assert out is review, out

path = os.path.join(tmp, solve.REPORT_FILE)
listing = open(path).read()
print(f"\n--- {solve.REPORT_FILE} written by solve.py ({len(listing)} bytes) ---")
print(listing)

size, label = writer.choose_page(writer.parse_report(listing))
# Jog point for the dry run: top-right corner, sheet centred in the arm's reach.
ref = f"{size[0] + 0.5:.3f},{-size[1] / 2:.3f},0"
print(f"page {label} {size[0]:.3f} x {size[1]:.3f} in, calibrating at {ref}")

result = subprocess.run(
    [sys.executable, os.path.join(PROJECT, "writer.py"), path,
     "--ref", ref, "--tour", "--sim", "--fast"],
    capture_output=True, text=True, cwd=PROJECT)
print("--- writer.py on solve.py's output ---")
print(result.stdout)
sys.stdout.write(result.stderr)
assert result.returncode == 0, result.stderr
assert f"[tour] marked {len(mistakes)} of {len(mistakes)}" in result.stdout, result.stdout
assert os.path.exists(os.path.join(tmp, "last_marked.jpg")), "the marked page should exist"
print("round trip ok: solve.py -> last_report.txt -> writer.py tour")
