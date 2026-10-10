"""Render a passage with deliberate mistakes for testing solve.py (no printer needed).

    python make_test_text.py            # -> text_test.png

Ground truth - these mistakes are planted in the image:
    "i"         -> "I"           capitalization (sentence start)
    "its"       -> "it's"        punctuation (contraction needs an apostrophe)
    "recieve"   -> "receive"     spelling
    "run"       -> "runs"        grammar (subject-verb agreement)
    "teh"       -> "the"         spelling
    "alot"      -> "a lot"       spelling
    "should of" -> "should have" grammar
    missing "." at the end of line 3 (punctuation)
"""
import cv2
import numpy as np

LINES = [
    "i think its time to recieve the package.",
    "The dog run fast but teh cat is calm.",
    "There are alot of reasons to be happy",
    "We should of checked the weather.",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
MARGIN = 40
LINE_H = 70

width = 1000
height = MARGIN * 2 + LINE_H * len(LINES)
img = np.full((height, width, 3), 255, np.uint8)

for i, line in enumerate(LINES):
    y = MARGIN + i * LINE_H + 40
    cv2.putText(img, line, (MARGIN, y), FONT, 0.9, (0, 0, 0), 2)

cv2.imwrite("text_test.png", img)
print("wrote text_test.png")
