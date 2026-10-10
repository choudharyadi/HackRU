"""Render a word-search puzzle image for testing solve.py (no printer needed).

    python make_test_wordsearch.py            # -> wordsearch_test.png

Ground truth (1-indexed, from the top-left):
    REACT : row 1,  cols 1-5,  across (left to right)
    ROBOT : col 3,  rows 3-7,  down
    CODE  : diagonal (6,6) -> (9,9), down-right
    TREE  : row 10, cols 10 -> 7, across (right to left)
    WIN   : col 8,  rows 6 -> 4, up
"""
import random

import cv2
import numpy as np

SIZE = 10
CELL = 70
MARGIN = 40
WORDS = ["REACT", "ROBOT", "CODE", "TREE", "WIN"]

grid = [["" for _ in range(SIZE)] for _ in range(SIZE)]


def place(word, r, c, dr, dc):
    for i, ch in enumerate(word):
        grid[r + dr * i][c + dc * i] = ch


place("REACT", 0, 0, 0, 1)   # across
place("ROBOT", 2, 2, 1, 0)   # down
place("CODE", 5, 5, 1, 1)    # diagonal down-right
place("TREE", 9, 9, 0, -1)   # right to left
place("WIN", 5, 7, -1, 0)    # up

random.seed(7)
for r in range(SIZE):
    for c in range(SIZE):
        if not grid[r][c]:
            grid[r][c] = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

font = cv2.FONT_HERSHEY_SIMPLEX
width = MARGIN * 2 + CELL * SIZE
height = MARGIN * 2 + CELL * SIZE + 90
img = np.full((height, width, 3), 255, np.uint8)

cv2.putText(img, "WORD SEARCH", (MARGIN, MARGIN - 8), font, 0.9, (0, 0, 0), 2)

for r in range(SIZE):
    for c in range(SIZE):
        x = MARGIN + c * CELL + CELL // 2 - 14
        y = MARGIN + r * CELL + CELL // 2 + 14
        cv2.putText(img, grid[r][c], (x, y), font, 1.0, (0, 0, 0), 2)

y = MARGIN + CELL * SIZE + 40
cv2.putText(img, "Find: " + "  ".join(WORDS), (MARGIN, y), font, 0.8, (0, 0, 0), 2)

cv2.imwrite("wordsearch_test.png", img)
print("wrote wordsearch_test.png")
